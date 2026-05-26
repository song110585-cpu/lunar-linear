"""
STDL-Net 月球线性构造多类别分割训练
通道顺序: [WAC, DEM, Slope, TPI, 剖面曲率]
类别: 0=背景, 1=皱脊, 2=月溪, 3=断层, 4=地堑
"""
import os
import csv
import json
import shutil
import datetime
import random
import inspect
import argparse
import yaml
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================================
# 环境检测
# ============================================================================

def detect_env():
    """自动检测 Kaggle vs 本地环境, 返回路径配置和预训练目录."""
    if os.path.isdir('/kaggle'):
        # ---- Kaggle 环境 ----
        data_root = '/kaggle/input/datasets/changyasong/datasetv5/lunar-dataset/dataset'
        # if not os.path.isdir(data_root):
        #     data_root = '/kaggle/input/lunar-data-v2/datasetv5/dataset'

        # 自动查找 valid_tiles_*.txt: 优先 dataset 根目录, 然后逐级向上找
        def _find_valid_list(filename):
            candidates = [
                os.path.join(data_root, filename),
                os.path.join(os.path.dirname(data_root), filename),
                os.path.join(os.path.dirname(os.path.dirname(data_root)), filename),
            ]
            for p in candidates:
                if os.path.isfile(p):
                    return p
            return None

        return {
            'is_kaggle': True,
            'train_image_dir': os.path.join(data_root, 'train', 'image'),
            'train_mask_dir':  os.path.join(data_root, 'train', 'mask'),
            'test_image_dir':  os.path.join(data_root, 'test', 'image'),
            'test_mask_dir':   os.path.join(data_root, 'test', 'mask'),
            'pretrain_dir':    os.path.join(data_root, 'pretrain'),
            'record_path':     '/kaggle/working/result',
            'train_valid_list': _find_valid_list('valid_tiles_train.txt'),
            'test_valid_list':  _find_valid_list('valid_tiles_test.txt'),
        }
    else:
        # ---- 本地环境 ----
        _filter_dir = r'E:\月球_dataset\Research area\dataset_analysis'
        return {
            'is_kaggle': False,
            'train_image_dir': r'E:\月球_dataset\Research area\train\dataset_v5\image',
            'train_mask_dir':  r'E:\月球_dataset\Research area\train\dataset_v5\mask',
            'test_image_dir':  r'E:\月球_dataset\Research area\test\dataset_v5\image',
            'test_mask_dir':   r'E:\月球_dataset\Research area\test\dataset_v5\mask',
            'pretrain_dir':    None,  # 使用 swinv2unet.py 内置路径
            'record_path':     r'E:\月球_dataset\Research area\result',
            'train_valid_list': os.path.join(_filter_dir, 'valid_tiles_train.txt'),
            'test_valid_list':  os.path.join(_filter_dir, 'valid_tiles_test.txt'),
        }


# ============================================================================
# 配置
# ============================================================================


def load_yaml_config(path):
    """加载 YAML 配置文件, 返回 dict. 文件不存在时抛出错误."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class HyperParameter:
    def __init__(self, env=None, config=None):
        if env is None:
            env = detect_env()

        # 加载 YAML config (如果提供), 否则用空 dict
        if isinstance(config, str):
            config = load_yaml_config(config)
        elif config is None:
            config = {}

        self.is_kaggle = env['is_kaggle']
        self.pretrain_dir = env['pretrain_dir']

        curr_time = datetime.datetime.now()
        curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
        self.name = "_result" + curr_time_str

        # ---- 核心参数 (YAML 覆盖默认值) ----
        self.num_classes    = config.get('num_classes', 5)
        self.in_channels    = config.get('in_channels', 5)
        self.model_size     = config.get('model_size', 'small')
        self.freeze_stages  = config.get('freeze_stages', 1)

        self.num_epochs     = config.get('num_epochs', 60)
        self.max_steps      = config.get('max_steps', 0)
        self.batch_size     = config.get('batch_size', 4)
        self.accum_steps    = config.get('accum_steps', 1)
        self.learning_rate  = config.get('learning_rate', 5e-5)

        # ---- 模块开关 ----
        self.use_augment        = config.get('use_augment', True)
        self.use_copypaste      = config.get('use_copypaste', True)
        self.copypaste_p        = config.get('copypaste_p', 0.5)
        self.use_strip_pooling  = config.get('use_strip_pooling', False)
        self.use_coord_attention = config.get('use_coord_attention', False)
        self.use_boundary_loss  = config.get('use_boundary_loss', False)
        self.use_dem_guided     = config.get('use_dem_guided', True)
        self.terrain_channels   = config.get('terrain_channels', 4)

        # ---- Early stopping ----
        self.early_stop = config.get('early_stop', True)
        self.patience   = config.get('patience', 12)

        # ---- 导出开关 ----
        self.save_all_test_on_best = config.get('save_all_test_on_best', True)
        self.save_pred_mask_png    = config.get('save_pred_mask_png', True)
        self.save_pred_vis_png     = config.get('save_pred_vis_png', True)

        # ---- 类别权重 ----
        self.class_weights = config.get('class_weights', [0.15, 1.0, 1.3, 1.8, 2.5])

        # ---- 数据路径 ----
        self.train_image_dir = env['train_image_dir']
        self.train_mask_dir  = env['train_mask_dir']
        self.test_image_dir  = env['test_image_dir']
        self.test_mask_dir   = env['test_mask_dir']

        # ---- 数据过滤清单 (None 表示不过滤) ----
        self.train_valid_list = env['train_valid_list']
        self.test_valid_list  = env['test_valid_list']

        # ---- 输出路径 ----
        self.record_path = env['record_path']
        self.model_save_path = os.path.join(self.record_path, self.name + '.pth')
        self.result_dir = os.path.join(self.record_path, self.name)


# ============================================================================
# 工具函数
# ============================================================================

CLASS_NAMES = ['背景', '皱脊', '月溪', '断层', '地堑']
CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 100, 255],
    [0, 200, 0],
    [255, 255, 0],
], dtype=np.uint8)
CLASS_COLORS_PLT = ['black', 'red', 'dodgerblue', 'green', 'orange']


def denormalize(tensor_image, mean=None, std=None):
    """多通道反归一化, 返回用于可视化的 uint8 图像 (H,W) 或 (H,W,3)."""
    C = tensor_image.shape[0]
    img = tensor_image.clone().float().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(C, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(C, 1, 1)
        img = img * std_t + mean_t
    if C >= 3:
        img = img[:3]
    img_np = img.numpy()
    mn, mx = img_np.min(), img_np.max()
    if mx - mn > 1e-8:
        img_np = (img_np - mn) / (mx - mn)
    else:
        img_np = np.zeros_like(img_np)
    img_np = (img_np * 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        return img_np[0]
    return img_np.transpose(1, 2, 0)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(mask.astype(np.int64), 0, 4)]


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP=绿, FN=红, FP=橙"""
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    gt_fg, pr_fg = gt > 0, pred > 0
    out[gt_fg & pr_fg]    = [0, 200, 0]
    out[gt_fg & (~pr_fg)] = [255, 0, 0]
    out[(~gt_fg) & pr_fg] = [255, 165, 0]
    return out


def showTensor(data, title="data"):
    data = data.cpu()
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(data.detach().numpy(), cmap='gray')
    plt.title(title)
    plt.show()
    plt.close(fig)


# ============================================================================
# 数据增强 Dataset (CopyPaste + 几何 + 像素增强)
# ============================================================================

class AugmentedDataset(torch.utils.data.Dataset):
    """翻转 + 旋转 + 高斯噪声 + 亮度扰动 + Cutout + CopyPaste"""
    def __init__(self, base_dataset, copypaste=False, copypaste_p=0.5):
        self.base = base_dataset
        self.copypaste = copypaste
        self.copypaste_p = copypaste_p

        if self.copypaste:
            self.rare_indices = []
            print('CopyPaste: 预索引少数类样本...')
            for i in range(len(self.base)):
                _, mask, _ = self.base[i]
                classes_present = set(mask.unique().tolist())
                if 3 in classes_present or 4 in classes_present:
                    self.rare_indices.append(i)
            print(f'CopyPaste: 找到 {len(self.rare_indices)} 张含 Fault/Graben 的样本')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask, name = self.base[idx]

        if self.copypaste and self.rare_indices and random.random() < self.copypaste_p:
            donor_idx = random.choice(self.rare_indices)
            donor_img, donor_mask, _ = self.base[donor_idx]
            rare_fg = (donor_mask == 3) | (donor_mask == 4)
            if rare_fg.any():
                fg_mask = rare_fg.unsqueeze(0).expand_as(img)
                img = torch.where(fg_mask, donor_img, img)
                mask = torch.where(rare_fg, donor_mask, mask)

        # 几何增强
        if random.random() > 0.5:
            img = img.flip(-1)
            mask = mask.flip(-1)
        if random.random() > 0.5:
            img = img.flip(-2)
            mask = mask.flip(-2)
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        # 像素增强
        if random.random() > 0.5:
            noise = torch.randn_like(img) * 0.02
            img = img + noise
        if random.random() > 0.5:
            C = img.shape[0]
            shift = (torch.rand(C, 1, 1) - 0.5) * 0.1
            img = img + shift
        if random.random() > 0.7:
            _, H, W = img.shape
            n_holes = random.randint(1, 3)
            for _ in range(n_holes):
                size = random.randint(32, 64)
                y = random.randint(0, H - size)
                x = random.randint(0, W - size)
                img[:, y:y+size, x:x+size] = 0.0

        return img, mask, name


# ============================================================================
# 损失函数
# ============================================================================

def dice_loss(logits, targets, smooth=1.0):
    """前景类 Dice Loss"""
    probs = torch.softmax(logits, dim=1)
    dice = 0.0
    for c in range(1, logits.shape[1]):
        p = probs[:, c]
        g = (targets == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        dice += (1 - (2 * inter + smooth) / (union + smooth)).mean()
    return dice / (logits.shape[1] - 1)


def combined_loss(ce_loss_fn, logits, targets):
    return ce_loss_fn(logits, targets) + 0.5 * dice_loss(logits, targets)


# ============================================================================
# 导出 & 可视化
# ============================================================================

def export_all_test(model, test_iter, save_root, epoch, num_classes, device, use_cuda=True):
    mask_dir = os.path.join(save_root, 'pred_mask')
    vis_dir  = os.path.join(save_root, 'pred_vis')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()

    model.eval()
    with torch.no_grad(), amp_ctx:
        for img, label, name in tqdm(test_iter, desc=f'Export@E{epoch}', unit='img'):
            img, label = img.to(device), label.to(device)
            pred = model(img).argmax(dim=1)

            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            gt_np   = label[0].cpu().numpy().astype(np.uint8)
            stem    = name[0]

            Image.fromarray(pred_np, mode='L').save(
                os.path.join(mask_dir, f'{stem}.png'))

            x = img[0].cpu().numpy()
            wac = x[0] * CHANNEL_STD[0] + CHANNEL_MEAN[0]
            wac = np.clip(wac, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(wac, cmap='gray');          axes[0].set_title(f'WAC - {stem}', fontsize=8)
            axes[1].imshow(mask_to_color(gt_np));      axes[1].set_title('GT')
            axes[2].imshow(mask_to_color(pred_np));    axes[2].set_title('Pred')
            axes[3].imshow(error_map(gt_np, pred_np)); axes[3].set_title('Error(G=TP,R=FN,O=FP)')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{stem}.png'), dpi=120)
            plt.close(fig)

    print(f'已导出 {len(os.listdir(mask_dir))} 张 pred_mask + {len(os.listdir(vis_dir))} 张 pred_vis')


def plot_training_curves(history, save_path, num_classes):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ep = history['epoch']

    axes[0, 0].plot(ep, history['train_loss'], 'o-', ms=3, label='Train')
    axes[0, 0].plot(ep, history['test_loss'],  's-', ms=3, label='Test')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history['train_miou'], 'o-', ms=3, label='Train')
    axes[0, 1].plot(ep, history['test_miou'],  's-', ms=3, label='Test')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('mIoU Curve'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    arr = np.array(history['train_iou_per_class'])
    for c in range(num_classes):
        axes[1, 0].plot(ep, arr[:, c], 'o-', ms=2, lw=1.5,
                        color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Train Per-Class IoU'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    arr = np.array(history['test_iou_per_class'])
    for c in range(num_classes):
        axes[1, 1].plot(ep, arr[:, c], 's-', ms=2, lw=1.5,
                        color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Test Per-Class IoU'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


# ============================================================================
# 训练
# ============================================================================

def train(hp: HyperParameter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f'Device: {device}  CUDA: {use_cuda}')
    if not use_cuda:
        print('WARNING: 本地 CPU 模式, 仅用于调试流程 (无法真正训练). '
              '正式训练请在 Kaggle GPU 环境运行 notebook.')

    os.makedirs(hp.result_dir, exist_ok=True)

    # ---- Kaggle: 替换 swinv2unet 内置预训练路径 ----
    if hp.is_kaggle and hp.pretrain_dir:
        import swinv2unet
        swinv2unet._SWIN_V2_PRETRAINED = {
            'tiny':  os.path.join(hp.pretrain_dir, 'swinv2_tiny_patch4_window16_256.pth'),
            'small': os.path.join(hp.pretrain_dir, 'swinv2_small_patch4_window16_256.pth'),
            'base':  os.path.join(hp.pretrain_dir, 'swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'),
        }
        print('Kaggle: 已注入预训练路径')

    # ---- 模型 ----
    init_sig = inspect.signature(Swin_LCSRB_DeformablePSP_FPNPAN.__init__)
    init_params = set(init_sig.parameters.keys())

    model_kwargs = dict(
        size=hp.model_size,
        num_classes=hp.num_classes,
        in_channels=hp.in_channels,
        pretrained=True,
    )
    if 'use_strip_pooling' in init_params and hp.use_strip_pooling:
        model_kwargs['use_strip_pooling'] = True
    if 'use_coord_attention' in init_params and hp.use_coord_attention:
        model_kwargs['use_coord_attention'] = True
    if 'use_dem_guided' in init_params and hp.use_dem_guided:
        model_kwargs['use_dem_guided'] = True
        model_kwargs['terrain_channels'] = hp.terrain_channels

    if hp.use_dem_guided and 'use_dem_guided' not in init_params:
        raise RuntimeError(
            '当前 swinv2unet.py 不支持 use_dem_guided 参数! '
            f'可用参数: {sorted(init_params - {"self"})}'
        )

    print(f'Model kwargs: {list(model_kwargs.keys())}')
    print(f'Available init params: {sorted(init_params - {"self"})}')

    model = Swin_LCSRB_DeformablePSP_FPNPAN(**model_kwargs).to(device)

    if hp.freeze_stages > 0:
        model.freeze_backbone_stages(hp.freeze_stages)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Total params: {n_params:.1f}M, Trainable: {n_train:.1f}M')

    # ---- 数据集 ----
    train_data_raw = MyDataset(
        images_dir=hp.train_image_dir,
        masks_dir=hp.train_mask_dir,
        valid_list_file=hp.train_valid_list,
    )
    test_data = MyDataset(
        images_dir=hp.test_image_dir,
        masks_dir=hp.test_mask_dir,
        valid_list_file=hp.test_valid_list,
    )

    if hp.use_augment:
        train_data = AugmentedDataset(
            train_data_raw, copypaste=hp.use_copypaste, copypaste_p=hp.copypaste_p)
        aug_str = '翻转 + 旋转 + 高斯噪声 + 亮度扰动 + Cutout'
        if hp.use_copypaste:
            aug_str += f' + CopyPaste(p={hp.copypaste_p})'
        print(f'数据增强: {aug_str}')
    else:
        train_data = train_data_raw

    train_iter = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True,
                            num_workers=2 if use_cuda else 0,
                            pin_memory=use_cuda, drop_last=True)
    test_iter  = DataLoader(test_data,  batch_size=1, shuffle=False,
                            num_workers=2 if use_cuda else 0,
                            pin_memory=use_cuda)

    print(f'训练集: {len(train_data)} 张, 测试集: {len(test_data)} 张')
    print(f'训练 batches/epoch: {len(train_iter)}')

    # ---- 损失 ----
    class_weights = torch.tensor(hp.class_weights, dtype=torch.float32).to(device)
    print(f'Class weights: {class_weights.tolist()}')
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def loss_fn(logits, targets):
        return combined_loss(ce_loss_fn, logits, targets)

    # ---- 优化器 & 调度器 ----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.num_epochs, eta_min=1e-6)

    # ---- 训练状态 ----
    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    best_miou = 0.0
    best_export_dir = None
    no_improve_count = 0

    history = {
        'epoch': [],
        'train_loss': [], 'train_miou': [], 'train_acc': [],
        'test_loss':  [], 'test_miou':  [], 'test_acc':  [],
        'train_iou_per_class': [], 'test_iou_per_class': [],
    }

    csv_path = os.path.join(hp.result_dir, 'epoch_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['epoch', 'train_loss', 'test_loss', 'train_miou', 'test_miou',
                  'train_acc', 'test_acc']
        for c in range(hp.num_classes):
            header += [f'train_iou_{c}', f'test_iou_{c}']
        w.writerow(header)

    # ---- 训练循环 ----
    for epoch in range(1, hp.num_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        hist = torch.zeros(hp.num_classes, hp.num_classes, dtype=torch.float64)
        optimizer.zero_grad()

        pbar = tqdm(train_iter, desc=f'Epoch {epoch}/{hp.num_epochs}', unit='batch')
        for step, (img, label, name) in enumerate(pbar):
            if hp.max_steps > 0 and step >= hp.max_steps:
                break
            img, label = img.to(device), label.to(device)

            with amp_ctx:
                logits = model(img)
                l = loss_fn(logits, label) / hp.accum_steps

            if scaler is not None:
                scaler.scale(l).backward()
                if (step + 1) % hp.accum_steps == 0 or (step + 1) == len(train_iter):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                l.backward()
                if (step + 1) % hp.accum_steps == 0 or (step + 1) == len(train_iter):
                    optimizer.step()
                    optimizer.zero_grad()

            train_losses.append(l.item() * hp.accum_steps)
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                hist += metrics.multiclass_confusion(pred, label, hp.num_classes).double()

            if step % 50 == 0:
                pbar.set_postfix(loss=f'{np.mean(train_losses[-50:]):.4f}')

        m = metrics.metrics_from_hist(hist)
        avg_loss = float(np.mean(train_losses))
        print(f'\n[Train] loss: {avg_loss:.4f}  mIoU: {m["miou"]:.4f}  acc: {m["accuracy"]:.4f}')
        print('  IoU:', ' '.join(f'{v:.4f}' for v in m['iou_per_class']))

        scheduler.step()

        # --- Test ---
        model.eval()
        test_losses = []
        test_hist = torch.zeros(hp.num_classes, hp.num_classes, dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, name in tqdm(test_iter, desc='Testing', unit='img'):
                if hp.max_steps > 0 and len(test_losses) >= hp.max_steps:
                    break
                img, label = img.to(device), label.to(device)
                logits = model(img)
                test_losses.append(loss_fn(logits, label).item())
                pred = logits.argmax(dim=1)
                test_hist += metrics.multiclass_confusion(pred, label, hp.num_classes).double()

        tm = metrics.metrics_from_hist(test_hist)
        test_avg_loss = float(np.mean(test_losses))
        print(f'[Test]  loss: {test_avg_loss:.4f}  mIoU: {tm["miou"]:.4f}  acc: {tm["accuracy"]:.4f}')
        print('  IoU:', ' '.join(f'{v:.4f}' for v in tm['iou_per_class']))

        # --- 记录 ----
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['train_miou'].append(float(m['miou']))
        history['train_acc'].append(float(m['accuracy']))
        history['test_loss'].append(test_avg_loss)
        history['test_miou'].append(float(tm['miou']))
        history['test_acc'].append(float(tm['accuracy']))
        history['train_iou_per_class'].append([float(v) for v in m['iou_per_class']])
        history['test_iou_per_class'].append([float(v) for v in tm['iou_per_class']])

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            row = [epoch, f'{avg_loss:.6f}', f'{test_avg_loss:.6f}',
                   f'{float(m["miou"]):.6f}', f'{float(tm["miou"]):.6f}',
                   f'{float(m["accuracy"]):.6f}', f'{float(tm["accuracy"]):.6f}']
            for c in range(hp.num_classes):
                row += [f'{m["iou_per_class"][c]:.6f}', f'{tm["iou_per_class"][c]:.6f}']
            csv.writer(f).writerow(row)

        with open(os.path.join(hp.result_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # --- 保存最优 ----
        if tm['miou'] > best_miou:
            best_miou = float(tm['miou'])
            no_improve_count = 0
            torch.save(model.state_dict(), hp.model_save_path)
            print(f'>>> Best model saved! mIoU={best_miou:.4f}')

            if hp.save_all_test_on_best:
                if best_export_dir and os.path.isdir(best_export_dir):
                    shutil.rmtree(best_export_dir, ignore_errors=True)
                best_export_dir = os.path.join(hp.result_dir,
                    f'best_epoch_{epoch:02d}_miou_{best_miou:.4f}')
                export_all_test(model, test_iter, best_export_dir, epoch,
                                hp.num_classes, device, use_cuda=use_cuda)
        else:
            no_improve_count += 1
            print(f'  No improvement ({no_improve_count}/{hp.patience})')

        # --- Checkpoint ----
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(hp.result_dir, f'ckpt_epoch{epoch}.pth'))
            print(f'Checkpoint saved at epoch {epoch}')

        # --- Early stopping ---
        if hp.early_stop and no_improve_count >= hp.patience:
            print(f'\n*** Early stopping at epoch {epoch} '
                  f'(no improvement for {hp.patience} epochs) ***')
            break

    # ---- 保存最终模型 ----
    final_path = os.path.join(hp.result_dir, f'final_{hp.model_size}.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved: {final_path}')

    # ---- 训练曲线 ----
    plot_training_curves(history,
                         os.path.join(hp.result_dir, 'training_curves.png'),
                         hp.num_classes)

    print(f'\nTraining done! Best mIoU = {best_miou:.4f}')
    print(f'Epoch metrics: {csv_path}')
    return model, best_miou


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STDL-Net 月球线性构造分割训练')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径 (如 configs/R24.yaml)')
    args = parser.parse_args()

    env = detect_env()
    hp = HyperParameter(env, config=args.config)

    print(f'Environment: {"Kaggle" if hp.is_kaggle else "Local"}')
    print(f'Data: {hp.train_image_dir}')
    if hp.train_valid_list:
        print(f'Train filter: {hp.train_valid_list}')
        print(f'Test  filter: {hp.test_valid_list}')
    print(f'Model: {hp.model_size}, Epochs: {hp.num_epochs}, '
          f'BS: {hp.batch_size}x{hp.accum_steps}(eff={hp.batch_size * hp.accum_steps}), '
          f'LR: {hp.learning_rate}')
    print(f'Freeze stages: {hp.freeze_stages}')
    print(f'Augmentation: {hp.use_augment}, CopyPaste: {hp.use_copypaste} (p={hp.copypaste_p})')
    print(f'StripPooling: {hp.use_strip_pooling}, CoordAttention: {hp.use_coord_attention}')
    print(f'BoundaryLoss: {hp.use_boundary_loss}, DEM-Guided: {hp.use_dem_guided} '
          f'(terrain_channels={hp.terrain_channels})')
    print(f'EarlyStop: {hp.early_stop} (patience={hp.patience})')
    print(f'Result dir: {hp.result_dir}')

    train(hp)
