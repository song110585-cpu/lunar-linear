"""
LTL-Net 训练脚本: ResNet50 + 细线保持解码器 (TPD)

与 train_baseline.py 数据/损失/训练配置完全一致, 唯一变量是模型结构 (TPD 多接入 f1 细节特征)。
用于在随机 8:1:1 (datasetv5_random811) 上与 DeepLabV3+ (前景 mIoU 0.674) 公平对比。

用法:
  python LTL-Net/scripts/train_ltl.py --data-dir <数据集根目录>
  python LTL-Net/scripts/train_ltl.py --detail-channels 32   # 消融: 调 f1 细节通道数
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

import metrics
from MyDataset import MyDataset
from models.ltl_net import LTLNet


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

# =========================
# 工具函数 (复用 train_baseline.py)
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
    best_miou = 0.0
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

        lr_scheduler.step()
        val_miou = net_test(model=model, test_iter=val_iter, loss=loss,
                            record_path=record_path, num_classes=num_classes,
                            epoch=str(epoch), save=False, max_steps=max_steps)

        if val_miou > best_miou:
            best_miou = val_miou
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
            print(f'save best model at epoch {epoch}, mIoU={best_miou:.4f}')


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='resnet50', help='backbone, 默认 resnet50')
    parser.add_argument('--detail-channels', type=int, default=16,
                        help='f1(1/2 细节特征) 降维后的通道数, 消融用 (默认 16)')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='每个 train/val/test 阶段最多运行的 batch 数；0 表示完整运行，仅用于冒烟测试')
    parser.add_argument('--data-dir', default=None,
                        help='数据集根目录(含 train/val/test 子目录), 默认自动检测 Kaggle/本地')
    parser.add_argument('--seed', type=int, default=42, help='训练随机种子')
    parser.add_argument('--run-name', default=None, help='输出目录名称；默认由模型和seed生成')
    args = parser.parse_args()

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

    tag = args.run_name or f'LTLNet_{args.encoder}_detail{args.detail_channels}_seed{args.seed}'
    if on_kaggle:
        RECORD_PATH = f'/kaggle/working/result_{tag}'
    else:
        RECORD_PATH = rf'E:\月球_dataset\output\{tag}'

    class HyperParameter:
        def __init__(self):
            curr_time = datetime.datetime.now()
            curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
            self.name = "_result" + curr_time_str
            self.num_epochs = args.epochs
            self.max_steps = args.max_steps
            self.learning_rate = 5e-5
            self.train_batchsize = 4
            self.test_batchsize = 1
            self.accum_steps = 1
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

    # ---- 模型 (LTL-Net = ResNet50 + TPD) ----
    model = LTLNet(
        encoder_name=args.encoder,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        highres_detail_channels=args.detail_channels,
    ).to(device)
    print(f'Model: LTLNet ({args.encoder}, detail_channels={args.detail_channels}), '
          f'params: {sum(p.numel() for p in model.parameters()):,}')

    # ---- 数据 ----
    train_data = MyDataset(hp.train_image_dir, hp.train_mask_dir)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_iter = DataLoader(train_data, batch_size=hp.train_batchsize, shuffle=True,
                            drop_last=False, num_workers=0, pin_memory=True,
                            worker_init_fn=seed_worker, generator=loader_generator)
    val_data = MyDataset(hp.val_image_dir, hp.val_mask_dir)
    val_iter = DataLoader(val_data, batch_size=hp.test_batchsize, shuffle=False,
                          drop_last=False, num_workers=0, pin_memory=True)
    test_data = MyDataset(hp.test_image_dir, hp.test_mask_dir)
    test_iter = DataLoader(test_data, batch_size=hp.test_batchsize, shuffle=False,
                           drop_last=False, num_workers=0, pin_memory=True)
    print(f'Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}')

    # ---- 损失 (与 baseline 完全一致) ----
    class_weights = torch.tensor([0.15, 1.0, 2.73, 1.98, 2.12], dtype=torch.float32).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    # ---- 优化器 (与 baseline 完全一致) ----
    opt = optim.AdamW(model.parameters(), lr=hp.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

    # ---- 训练 ----
    train(model=model, train_iter=train_iter, val_iter=val_iter, loss=loss, opt=opt,
          num_epochs=hp.num_epochs, record_path=hp.record_path, lr_scheduler=scheduler,
          num_classes=NUM_CLASSES, accum_steps=hp.accum_steps, max_steps=hp.max_steps,
          model_save_path=hp.model_save_path)

    # ---- 最终测试 ----
    print('\n' + '=' * 60)
    print('最终测试集评估 (Test Set Evaluation)')
    print('=' * 60)
    best_state = torch.load(hp.model_save_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state)
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
        'model': 'LTLNet',
        'encoder': args.encoder,
        'detail_channels': args.detail_channels,
        'seed': args.seed,
        'epochs': args.epochs,
        'data_root': DATA_ROOT,
        'git_commit': git_commit,
        'class_weights': class_weights.detach().cpu().tolist(),
        'max_steps': hp.max_steps,
        'selection_metric': 'val_mIoU_all',
        'test': final_metrics,
    }
    metrics_path = os.path.join(hp.record_path, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已保存: {metrics_path}')
