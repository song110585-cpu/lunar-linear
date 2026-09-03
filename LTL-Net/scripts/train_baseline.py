"""
统一 baseline 训练脚本: 用 --model 一次切换 smp 各经典分割模型, 摸清最强 backbone 基线。

与 R57 (train_deeplab.py) 数据/损失/训练配置完全一致, 仅模型可切换。
用途: 在随机 8:1:1 (datasetv5_random811) 上跑齐卷积系模型, 锁定最强 backbone。

用法:
  python STDL-Net/scripts/train_baseline.py --model Unet
  python STDL-Net/scripts/train_baseline.py --model PSPNet
  python STDL-Net/scripts/train_baseline.py --model Linknet
  python STDL-Net/scripts/train_baseline.py --model DeepLabV3Plus   # 复现 R57 0.674

可选模型 (smp 现成, 均支持 encoder_name='resnet50'):
  Unet / UnetPlusPlus / DeepLabV3 / DeepLabV3Plus / PSPNet / Linknet / FPN / PAN / MAnet
"""
import os, sys, argparse, json, random, subprocess
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
for _sub in ['utils', 'models', 'datasets']:
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import time, datetime, numpy as np
import matplotlib
matplotlib.use('Agg')

import segmentation_models_pytorch as smp
import metrics
from MyDataset import MyDataset
from experiment_artifacts import save_training_history
from models.dlinknet import DLinkNet
from models.pidnet_multiclass import PIDNetSmall, load_pidnet_imagenet_weights


def set_seed(seed):
    """固定 Python/NumPy/PyTorch/DataLoader 随机性，便于配对重复实验。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint_state(checkpoint_path):
    """读取纯 state_dict 或常见训练 checkpoint，并兼容 DataParallel 前缀。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f'checkpoint 不是可加载的 state_dict: {type(checkpoint).__name__}')
    if checkpoint and all(str(key).startswith('module.') for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint

# =========================
# 工具函数 (复用 train_deeplab.py)
# =========================
def net_test(model, test_iter, loss, record_path, num_classes,
             epoch='000', save=False, vis_mean=None, vis_std=None, max_steps=0,
             return_metrics=False):
    device = next(model.parameters()).device
    model.eval()
    test_epoch_loss = []
    hist_total = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    with torch.no_grad(), torch.cuda.amp.autocast():
        for t_step, (optical, label, img_name) in enumerate(tqdm(test_iter, desc=f'test     Epoch {epoch}    :', unit='img')):
            if max_steps > 0 and t_step >= max_steps:
                break
            optical, label = optical.to(device), label.to(device)
            logits = model(optical)
            l = loss(logits, label)
            test_epoch_loss.append(l.item())
            pred = logits.argmax(dim=1)
            hist_total += metrics.multiclass_confusion(pred, label, num_classes).double()

    test_epoch_loss = float(np.average(test_epoch_loss)) if test_epoch_loss else 0.0
    m = metrics.metrics_from_hist(hist_total)
    print('- ' * 30)
    print('test_loss: {:.6f}  acc: {:.4f}  mIoU: {:.4f}  mF1: {:.4f}  mPrec: {:.4f}  mRec: {:.4f}'.format(
        test_epoch_loss, m['accuracy'], m['miou'], m['mf1'], m['mprecision'], m['mrecall']))
    print('per-class IoU: ' + ', '.join(f'{v:.4f}' for v in m['iou_per_class']))
    print('per-class F1 : ' + ', '.join(f'{v:.4f}' for v in m['f1_per_class']))
    print('- ' * 30)
    if return_metrics:
        result = dict(m)
        result['loss'] = test_epoch_loss
        result['miou_fg'] = float(np.mean(m['iou_per_class'][1:]))
        return result
    return m['miou']


def train(model, train_iter, val_iter, loss, opt, num_epochs, record_path, lr_scheduler,
          num_classes=5, accum_steps=1, max_steps=0, model_save_path=None):
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler()
    best_miou_fg = 0.0
    best_epoch = 0
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_epoch_loss = []
        hist_total = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        opt.zero_grad()
        for step, (optical, label, img_name) in enumerate(tqdm(train_iter, desc=f'Epoch {epoch}/{num_epochs} ', unit='img')):
            if max_steps > 0 and step >= max_steps:
                break
            optical, label = optical.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                logits = model(optical)
                l = loss(logits, label) / accum_steps
            scaler.scale(l).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_iter):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            train_epoch_loss.append(l.item() * accum_steps)
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                hist_total += metrics.multiclass_confusion(pred, label, num_classes).double()

        train_epoch_loss = float(np.average(train_epoch_loss)) if train_epoch_loss else 0.0
        m = metrics.metrics_from_hist(hist_total)
        print('- ' * 30)
        print('train_loss: {:.6f}  acc: {:.4f}  mIoU: {:.4f}  mF1: {:.4f}  mPrec: {:.4f}  mRec: {:.4f}'.format(
            train_epoch_loss, m['accuracy'], m['miou'], m['mf1'], m['mprecision'], m['mrecall']))
        print('per-class IoU: ' + ', '.join(f'{v:.4f}' for v in m['iou_per_class']))
        print('- ' * 30)

        current_lr = float(opt.param_groups[0]['lr'])
        val_result = net_test(model=model, test_iter=val_iter, loss=loss,
                              record_path=record_path, num_classes=num_classes,
                              epoch=str(epoch), save=False, max_steps=max_steps,
                              return_metrics=True)
        val_miou_fg = val_result['miou_fg']

        is_best = val_miou_fg > best_miou_fg
        if is_best:
            best_miou_fg = val_miou_fg
            best_epoch = epoch
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
            print(f'save best model at epoch {epoch}, val_mIoU_fg={best_miou_fg:.4f}')

        history.append({
            'epoch': epoch,
            'learning_rate': current_lr,
            'train_loss': train_epoch_loss,
            'train_accuracy': m['accuracy'],
            'train_miou': m['miou'],
            'train_miou_fg': float(np.mean(m['iou_per_class'][1:])),
            'train_mf1': m['mf1'],
            'val_loss': val_result['loss'],
            'val_accuracy': val_result['accuracy'],
            'val_miou': val_result['miou'],
            'val_miou_fg': val_result['miou_fg'],
            'val_mf1': val_result['mf1'],
            'is_best': int(is_best),
        })
        save_training_history(history, record_path)
        lr_scheduler.step()
    return {'best_epoch': best_epoch, 'best_val_miou_fg': best_miou_fg, 'history': history}


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Unet',
                        help='模型名: DLinkNet/PIDNet-S，或smp的Unet/UnetPlusPlus/DeepLabV3/DeepLabV3Plus/PSPNet/Linknet/FPN/PAN/MAnet')
    parser.add_argument('--encoder', default='resnet50', help='backbone, 默认 resnet50')
    parser.add_argument('--data-dir', default=None,
                        help='数据集根目录(含 train/val/test 子目录), 默认自动检测 Kaggle/本地')
    parser.add_argument('--output-dir', default=None,
                        help='结果根目录；每次运行会在其中创建 result_<run-name>，AutoDL建议使用/root/autodl-tmp/outputs')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader进程数；Windows/Kaggle默认0，AutoDL建议4')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='训练物理batch size；DLinkNet-ResNet50显存占用大，建议2')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='梯度累积步数；有效batch size=batch-size*accum-steps')
    parser.add_argument('--seed', type=int, default=42, help='训练随机种子')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='每个 train/val/test 阶段最多运行的 batch 数；0 表示完整运行，仅用于冒烟测试')
    parser.add_argument('--run-name', default=None, help='输出目录名称；默认由模型和seed生成')
    parser.add_argument('--eval-only', action='store_true',
                        help='只加载 checkpoint 并评估 test，不进行训练')
    parser.add_argument('--skip-test-evaluation', action='store_true',
                        help='训练完成后不读取Test；用于先锁定Val模型、稍后统一评价Test')
    parser.add_argument('--checkpoint', default=None,
                        help='--eval-only 使用的模型权重路径；扩展名可以是 .pth/.pt/.zip')
    parser.add_argument('--pretrained-checkpoint', default=None,
                        help='PIDNet-S训练使用的官方ImageNet预训练权重；其他模型忽略')
    args = parser.parse_args()

    if args.eval_only and not args.checkpoint:
        parser.error('--eval-only 必须同时提供 --checkpoint')
    if args.eval_only and args.skip_test_evaluation:
        parser.error('--eval-only 与 --skip-test-evaluation 不能同时使用')

    set_seed(args.seed)

    NUM_CLASSES = 5
    IN_CHANNELS = 5
    IMG_SIZE = 512

    # ---- 路径: 命令行 --data-dir 优先, 否则自动检测 Kaggle/本地 ----
    on_kaggle = os.path.isdir('/kaggle')
    if args.data_dir:
        DATA_ROOT = args.data_dir
    elif on_kaggle:
        DATA_ROOT = '/kaggle/input/datasets/yuanssy/v5data/datasetv5_random811'
    else:
        DATA_ROOT = r'E:\月球_dataset\dataset\datasetv5_random811'

    run_name = args.run_name or f'{args.model}_{args.encoder}_seed{args.seed}'
    if args.output_dir:
        RECORD_PATH = os.path.join(
            os.path.abspath(os.path.expanduser(args.output_dir)), f'result_{run_name}'
        )
    elif on_kaggle:
        RECORD_PATH = f'/kaggle/working/result_{run_name}'
    elif os.path.isdir('/root/autodl-tmp'):
        RECORD_PATH = f'/root/autodl-tmp/outputs/result_{run_name}'
    else:
        RECORD_PATH = os.path.join(r'E:\月球_dataset\output', f'result_{run_name}')

    class HyperParameter:
        def __init__(self):
            curr_time = datetime.datetime.now()
            curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
            self.name = "_result" + curr_time_str
            self.num_epochs = args.epochs
            self.max_steps = args.max_steps
            self.learning_rate = 5e-5
            self.train_batchsize = args.batch_size
            self.test_batchsize = 1
            self.accum_steps = args.accum_steps
            self.train_image_dir = os.path.join(DATA_ROOT, 'train', 'image')
            self.train_mask_dir  = os.path.join(DATA_ROOT, 'train', 'mask')
            self.val_image_dir   = os.path.join(DATA_ROOT, 'val',   'image')
            self.val_mask_dir    = os.path.join(DATA_ROOT, 'val',   'mask')
            self.test_image_dir  = os.path.join(DATA_ROOT, 'test',  'image')
            self.test_mask_dir   = os.path.join(DATA_ROOT, 'test',  'mask')
            self.record_path = RECORD_PATH
            self.model_save_path = os.path.join(RECORD_PATH, 'best_model.pth')

    hp = HyperParameter()
    os.makedirs(hp.record_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}     GPU available:{torch.cuda.is_available()}')

    # ---- 模型 (D-LinkNet或SMP baseline) ----
    normalized_model = args.model.lower().replace('-', '').replace('_', '')
    encoder_identity = args.encoder
    pretrained_loading = None
    if normalized_model == 'dlinknet':
        model = DLinkNet(
            encoder_name=args.encoder,
            encoder_weights=None if args.eval_only else "imagenet",
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
        ).to(device)
    elif normalized_model in ('pidnet', 'pidnets'):
        encoder_identity = 'native_pidnet_s'
        model = PIDNetSmall(in_channels=IN_CHANNELS, classes=NUM_CLASSES).to(device)
        if not args.eval_only:
            if not args.pretrained_checkpoint:
                parser.error('PIDNet-S正式训练必须提供 --pretrained-checkpoint')
            pretrained_path = os.path.abspath(args.pretrained_checkpoint)
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f'PIDNet-S预训练权重不存在: {pretrained_path}')
            pretrained_loading = load_pidnet_imagenet_weights(model, pretrained_path)
            print(f'PIDNet-S ImageNet weights: {pretrained_loading}')
    else:
        assert hasattr(smp, args.model), f'smp 没有模型 {args.model}'
        model_cls = getattr(smp, args.model)
        model = model_cls(
            encoder_name=args.encoder,
            encoder_weights=None if args.eval_only else "imagenet",
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
        ).to(device)
    print(f'Model: {args.model} ({encoder_identity}), params: {sum(p.numel() for p in model.parameters()):,}')

    # ---- 数据 ----
    if args.eval_only:
        test_data = MyDataset(hp.test_image_dir, hp.test_mask_dir)
        test_iter = DataLoader(test_data, batch_size=hp.test_batchsize, shuffle=False,
                               drop_last=False, num_workers=args.num_workers, pin_memory=True)
        train_iter = val_iter = None
        print(f'Eval-only  Test: {len(test_data)}')
    else:
        train_data = MyDataset(hp.train_image_dir, hp.train_mask_dir)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        train_iter = DataLoader(train_data, batch_size=hp.train_batchsize, shuffle=True,
                                drop_last=False, num_workers=args.num_workers, pin_memory=True,
                                worker_init_fn=seed_worker, generator=loader_generator)
        val_data = MyDataset(hp.val_image_dir, hp.val_mask_dir)
        val_iter = DataLoader(val_data, batch_size=hp.test_batchsize, shuffle=False,
                              drop_last=False, num_workers=args.num_workers, pin_memory=True)
        if args.skip_test_evaluation:
            test_iter = None
            print(f'Train: {len(train_data)}  Val: {len(val_data)}  Test: locked')
        else:
            test_data = MyDataset(hp.test_image_dir, hp.test_mask_dir)
            test_iter = DataLoader(test_data, batch_size=hp.test_batchsize, shuffle=False,
                                   drop_last=False, num_workers=args.num_workers, pin_memory=True)
            print(f'Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}')

    # ---- 损失 ----
    class_weights = torch.tensor([0.15, 1.0, 2.73, 1.98, 2.12], dtype=torch.float32).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    if args.eval_only:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'checkpoint 不存在: {checkpoint_path}')
        print(f'加载 checkpoint: {checkpoint_path}')
        model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
    else:
        # ---- 优化器与训练 ----
        opt = optim.AdamW(model.parameters(), lr=hp.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
        train_summary = train(model=model, train_iter=train_iter, val_iter=val_iter, loss=loss, opt=opt,
                              num_epochs=hp.num_epochs, record_path=hp.record_path, lr_scheduler=scheduler,
                              num_classes=NUM_CLASSES, accum_steps=hp.accum_steps, max_steps=hp.max_steps,
                              model_save_path=hp.model_save_path)
        checkpoint_path = hp.model_save_path
    if args.eval_only:
        train_summary = None

    # ---- Test只在显式允许时评价 ----
    if args.skip_test_evaluation:
        print('\nTest evaluation is locked and was not executed.')
        final_metrics = None
    else:
        print('\n' + '=' * 60)
        print('最终测试集评估 (Test Set Evaluation)')
        print('=' * 60)
        if not args.eval_only:
            model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
        final_metrics = net_test(model=model, test_iter=test_iter, loss=loss,
                                 record_path=hp.record_path, num_classes=NUM_CLASSES,
                                 epoch='final', save=False, max_steps=hp.max_steps,
                                 return_metrics=True)
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_root, text=True
        ).strip()
    except Exception:
        git_commit = 'unknown'
    result = {
        'model': args.model,
        'encoder': encoder_identity,
        'seed': args.seed,
        'epochs': args.epochs,
        'data_root': DATA_ROOT,
        'git_commit': git_commit,
        'class_weights': class_weights.detach().cpu().tolist(),
        'num_workers': args.num_workers,
        'batch_size': hp.train_batchsize,
        'accum_steps': hp.accum_steps,
        'effective_batch_size': hp.train_batchsize * hp.accum_steps,
        'max_steps': hp.max_steps,
        'selection_metric': 'val_mIoU_fg',
        'automatic_test_evaluation': not args.skip_test_evaluation,
        'best_epoch': None if train_summary is None else train_summary['best_epoch'],
        'best_val_miou_fg': None if train_summary is None else train_summary['best_val_miou_fg'],
        'history_file': None if train_summary is None else 'history.csv',
        'eval_only': args.eval_only,
        'checkpoint': checkpoint_path,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'pretrained_loading': pretrained_loading,
        'test': final_metrics,
    }
    metrics_path = os.path.join(hp.record_path, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已保存: {metrics_path}')
