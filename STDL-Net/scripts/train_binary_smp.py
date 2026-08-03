"""
DeepLabV3+ 二分类对比实验
用 SMP 预训练模型，同样数据、同样任务，对比 Swin-UNet 效果
"""
import os, sys, csv, argparse, yaml, datetime
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'datasets'))
sys.path.insert(0, os.path.join(_root, 'utils'))

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
import segmentation_models_pytorch as smp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def label_to_binary(mask):
    return (mask > 0).long()


class AugmentedDataset(torch.utils.data.Dataset):
    """轻量增强: 只做翻转+旋转, 不依赖复杂逻辑"""
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        import random
        img, mask, name = self.base[idx]
        if random.random() > 0.5: img = img.flip(-1); mask = mask.flip(-1)
        if random.random() > 0.5: img = img.flip(-2); mask = mask.flip(-2)
        k = random.randint(0, 3)
        if k > 0: img = torch.rot90(img, k, [-2, -1]); mask = torch.rot90(mask, k, [-2, -1])
        return img, mask, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder', type=str, default='resnet50')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}, BS: {args.batch_size}, LR: {args.lr}")

    # ---- 数据 (复用 STDL-Net 的 Kaggle 路径逻辑) ----
    if os.path.isdir('/kaggle'):
        data_root = '/kaggle/input/datasets/yuanssy/dataset5/datasetv5'
        pretrain_path = None  # SMP 自动下载预训练
        record_path = '/kaggle/working/result'
    else:
        data_root = r'E:\月球_dataset\dataset\datasetv5'
        record_path = r'E:\月球_dataset\result'

    train_img = os.path.join(data_root, 'train', 'image')
    train_msk = os.path.join(data_root, 'train', 'mask')
    test_img = os.path.join(data_root, 'test', 'image')
    test_msk = os.path.join(data_root, 'test', 'mask')

    train_raw = MyDataset(train_img, train_msk)
    test_data = MyDataset(test_img, test_msk)

    # Val: 5% 随机 (和 R_binary 一致)
    import random as _rng; _rng.seed(42)
    indices = list(range(len(train_raw))); _rng.shuffle(indices)
    val_n = max(1, int(len(indices) * 0.05))
    train_subset = torch.utils.data.Subset(AugmentedDataset(train_raw), indices[val_n:])
    val_subset = torch.utils.data.Subset(train_raw, indices[:val_n])

    train_iter = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2 if use_cuda else 0, pin_memory=use_cuda, drop_last=True)
    val_iter = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                          num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    print(f"Train: {len(indices)-val_n}, Val: {val_n}, Test: {len(test_data)}")

    # ---- 模型 ----
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=5,
        classes=2,
    ).to(device)

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n:.1f}M")

    # ---- Loss ----
    class_weights = torch.tensor([0.5, 1.0], device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(logits, targets, smooth=1.0):
        probs = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(1, logits.shape[1]):
            p = probs[:, c]; g = (targets == c).float()
            inter = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice += (1 - (2 * inter + smooth) / (union + smooth)).mean()
        return dice / max(1, logits.shape[1] - 1)

    def loss_fn(logits, targets):
        return criterion_ce(logits, targets) + 0.5 * dice_loss(logits, targets)

    # ---- 优化器 ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()
    best_iou = 0.0; no_improve = 0
    result_dir = os.path.join(record_path, f'dlv3p_{args.encoder}_binary_{datetime.datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(result_dir, exist_ok=True)

    # ---- CSV ----
    csv_path = os.path.join(result_dir, 'epoch_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_iou_0', 'val_iou_0', 'train_iou_1', 'val_iou_1'])

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        train_losses = []; hist_train = torch.zeros(2, 2, dtype=torch.float64)
        for img, label, _ in tqdm(train_iter, desc=f'Epoch {epoch}', unit='batch'):
            img, label = img.to(device), label_to_binary(label).to(device)
            with amp_ctx:
                logits = model(img)
                loss = loss_fn(logits, label)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            train_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            hist_train += metrics.multiclass_confusion(pred, label, 2).double()

        scheduler.step()
        avg_loss = np.mean(train_losses)
        tr_m = metrics.metrics_from_hist(hist_train)
        tr_iou = tr_m['iou_per_class']

        # --- Val ---
        model.eval(); val_losses = []; hist_val = torch.zeros(2, 2, dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, _ in val_iter:
                img = img.to(device); label = label_to_binary(label).to(device)
                logits = model(img)
                val_losses.append(loss_fn(logits, label).item())
                pred = logits.argmax(dim=1)
                hist_val += metrics.multiclass_confusion(pred, label, 2).double()

        val_avg = np.mean(val_losses)
        vm = metrics.metrics_from_hist(hist_val)
        val_iou = vm['iou_per_class']

        print(f"[E{epoch}] train loss={avg_loss:.4f} IoU=[{tr_iou[0]:.4f},{tr_iou[1]:.4f}]  "
              f"val loss={val_avg:.4f} IoU=[{val_iou[0]:.4f},{val_iou[1]:.4f}]")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{avg_loss:.6f}', f'{val_avg:.6f}',
                                     f'{tr_iou[0]:.6f}', f'{val_iou[0]:.6f}', f'{tr_iou[1]:.6f}', f'{val_iou[1]:.6f}'])

        if val_iou[1] > best_iou:
            best_iou = val_iou[1]; no_improve = 0
            torch.save(model.state_dict(), os.path.join(result_dir, f'best_{args.encoder}.pth'))
            print(f"  >>> Best! val_line_IoU={best_iou:.4f}")
        else:
            no_improve += 1
            if no_improve >= 12:
                print(f"Early stop at epoch {epoch}"); break

    # --- Test ---
    best_path = os.path.join(result_dir, f'best_{args.encoder}.pth')
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval(); hist_test = torch.zeros(2, 2, dtype=torch.float64)
    with torch.no_grad(), amp_ctx:
        for img, label, _ in tqdm(test_iter, desc='Test'):
            img = img.to(device); label = label_to_binary(label).to(device)
            pred = model(img).argmax(dim=1)
            hist_test += metrics.multiclass_confusion(pred, label, 2).double()

    tm = metrics.metrics_from_hist(hist_test)
    test_iou = tm['iou_per_class']
    print(f"\n{'='*50}")
    print(f"  DeepLabV3+ ({args.encoder}) Test")
    print(f"  bg IoU={test_iou[0]:.4f}  line IoU={test_iou[1]:.4f}  mIoU={tm['miou']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
