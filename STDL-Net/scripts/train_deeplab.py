"""
R57: DeepLabV3+ 基线 (ResNet50 backbone)
与 R56 数据/配置完全一致, 仅模型不同.
用法: python STDL-Net/scripts/train_deeplab.py
"""
import os, sys
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

# =========================
# 工具函数 (复用 train.py)
# =========================
def net_test(model, test_iter, loss, record_path, num_classes,
             epoch='000', save=False, vis_mean=None, vis_std=None, max_steps=0):
    """多类别分割评估."""
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
    return m['miou']


def train(model, train_iter, val_iter, loss, opt, num_epochs, record_path, lr_scheduler,
          num_classes=5, accum_steps=1, max_steps=0, model_save_path=None):
    """多类别分割训练循环."""
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
    NUM_CLASSES = 5
    IN_CHANNELS = 5
    IMG_SIZE = 512

    # ---- Kaggle / 本地 路径自动检测 ----
    if os.path.isdir('/kaggle'):
        DATA_ROOT = '/kaggle/input/datasets/yuanssy/v5data/datasetv5_random811'
        RECORD_PATH = '/kaggle/working/result'
    else:
        DATA_ROOT = r'E:\月球_dataset\dataset\datasetv5_random811'
        RECORD_PATH = r'E:\月球_dataset\output\result57'

    class HyperParameter:
        def __init__(self):
            curr_time = datetime.datetime.now()
            curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
            self.name = "_result" + curr_time_str
            self.num_epochs = 80
            self.max_steps = 0
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
            self.model_save_path = os.path.join(RECORD_PATH, self.name + '.pth')

    hp = HyperParameter()
    os.makedirs(hp.record_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}     GPU available:{torch.cuda.is_available()}')

    # ---- DeepLabV3+ (ResNet50) ----
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    ).to(device)
    print(f'Model: DeepLabV3+ (ResNet50), params: {sum(p.numel() for p in model.parameters()):,}')

    # ---- 数据 ----
    train_data = MyDataset(hp.train_image_dir, hp.train_mask_dir)
    train_iter = DataLoader(train_data, batch_size=hp.train_batchsize, shuffle=True,
                            drop_last=False, num_workers=0, pin_memory=True)
    val_data = MyDataset(hp.val_image_dir, hp.val_mask_dir)
    val_iter = DataLoader(val_data, batch_size=hp.test_batchsize, shuffle=False,
                          drop_last=False, num_workers=0, pin_memory=True)
    test_data = MyDataset(hp.test_image_dir, hp.test_mask_dir)
    test_iter = DataLoader(test_data, batch_size=hp.test_batchsize, shuffle=False,
                           drop_last=False, num_workers=0, pin_memory=True)
    print(f'Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}')

    # ---- 损失 ----
    class_weights = torch.tensor([0.15, 1.0, 1.3, 1.8, 1.5], dtype=torch.float32).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    # ---- 优化器 ----
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
    best_state = torch.load(hp.model_save_path, map_location='cpu')
    model.load_state_dict(best_state)
    net_test(model=model, test_iter=test_iter, loss=loss,
             record_path=hp.record_path, num_classes=NUM_CLASSES,
             epoch='final', save=False, max_steps=hp.max_steps)
