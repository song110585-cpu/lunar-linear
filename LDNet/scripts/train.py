"""
LDNet 月球线性构造双分支分割训练

分支1 (Binary):  "线在哪？"  → BCE + Dice
分支2 (4-Class): "什么类型？" → CE (仅前景像素)
推理: final = argmax(b2) × (b1 > 0.5)

通道顺序: [WAC, DEM, Slope, TPI, 剖面曲率]
类别: 0=背景, 1=皱脊, 2=月溪, 3=断层, 4=地堑
"""
import os
import csv
import json
import shutil
import datetime
import random
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

# 确保能从 LDNet 目录 import
import sys as _sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models', 'losses']:
    _p = os.path.join(_root, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
from ldnet import LDNet, merge_branches
from dual_loss import DualBranchLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================================
# 环境检测 (复用 STDL-Net 逻辑)
# ============================================================================

def _generate_scene_split(image_dir):
    """根据文件名规则生成 Scene-Level train/val split."""
    import re
    SPLIT_ROW = 6447
    VAL_SCENE = 'Mare Serenitatis'

    def _extract_scene(fname):
        basename = fname.replace('.tif', '').replace('.tiff', '')
        m = re.match(r'^(.+?)_5ch_', basename)
        if m: return m.group(1)
        m = re.match(r'^train_(.+?)_r\d+', basename)
        if m: return m.group(1)
        m = re.match(r'^(?:train_)?(.+?)_r\d+', basename)
        if m: return m.group(1)
        return basename

    tiles = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    train_tiles, val_tiles = [], []
    for t in tiles:
        scene = _extract_scene(t)
        if scene == VAL_SCENE:
            m = re.search(r'_r(\d+)_', t)
            r = int(m.group(1)) if m else 0
            if r < SPLIT_ROW:
                val_tiles.append(t)
            else:
                train_tiles.append(t)
        else:
            train_tiles.append(t)
    return train_tiles, val_tiles


def detect_env():
    """自动检测 Kaggle vs 本地环境, 返回路径配置."""
    if os.path.isdir('/kaggle'):
        data_root_v5  = '/kaggle/input/datasets/yuanssy/dataset5/datasetv5'
        data_root_v10 = '/kaggle/input/datasets/yuanssy/dataset10/datasetv10'
        data_root_v9  = '/kaggle/input/datasets/changyasong/dataset9/datasetv9'
        data_root_v8  = '/kaggle/input/datasets/changyasong/dataset8/datasetv8'

        # v5 优先 (当前主力, 512 tile, 需自动生成 5% 分层 val)
        if os.path.isdir(data_root_v5):
            data_root = data_root_v5
            pretrain_dir = os.path.join(data_root, 'pretrain')
            os.makedirs('/kaggle/working', exist_ok=True)
            _tl = '/kaggle/working/v5_train_list.txt'
            _vl = '/kaggle/working/v5_val_list.txt'
            if not os.path.isfile(_vl):
                print('[Env] Kaggle v5: 生成 5% 分层随机 val...')
                import random as _rnd, numpy as _np, rasterio
                from collections import defaultdict
                _rnd.seed(42); _np.random.seed(42)
                _cls_tiles = defaultdict(list)
                _train_img = os.path.join(data_root, 'train', 'image')
                for t in sorted(f for f in os.listdir(_train_img)
                               if f.endswith('.tif')):
                    with rasterio.open(os.path.join(data_root, 'train', 'mask', t)) as src:
                        m = src.read(1).astype(_np.int64)
                    m[(m > 4) | (m < 0)] = 0
                    dom = max(range(1, 5), key=lambda c: int(_np.sum(m == c))) \
                        if _np.sum(m > 0) > 0 else 0
                    _cls_tiles[dom].append(t)
                _tr, _va = [], []
                for c in range(5):
                    ts = _cls_tiles[c]; _rnd.shuffle(ts)
                    nv = max(1, int(len(ts) * 0.05))
                    _va += ts[:nv]; _tr += ts[nv:]
                for path, tiles in [(_tl, _tr), (_vl, _va)]:
                    with open(path, 'w', encoding='utf-8') as f:
                        for t in sorted(tiles): f.write(t + '\n')
                print(f'[Env] Kaggle v5 Train={len(_tr)}, Val={len(_va)}')
            print(f'[Env] Kaggle v5 (5% Stratified Val): {data_root}')
            return {
                'is_kaggle': True,
                'train_image_dir': os.path.join(data_root, 'train', 'image'),
                'train_mask_dir':  os.path.join(data_root, 'train', 'mask'),
                'val_image_dir':   os.path.join(data_root, 'train', 'image'),
                'val_mask_dir':    os.path.join(data_root, 'train', 'mask'),
                'test_image_dir':  os.path.join(data_root, 'test', 'image'),
                'test_mask_dir':   os.path.join(data_root, 'test', 'mask'),
                'pretrain_dir':    pretrain_dir,
                'record_path':     '/kaggle/working/result',
                'train_valid_list': _tl,
                'val_valid_list':   _vl,
                'test_valid_list':  None,
            }

        # fallback: v10/v9/v8 (独立 train/val/test 目录)
        for _dr in [data_root_v10, data_root_v9, data_root_v8]:
            if os.path.isdir(_dr):
                data_root = _dr
                break
        else:
            raise FileNotFoundError('Kaggle: 未找到任何数据集 (v5/v10/v9/v8)')

        pretrain_dir = os.path.join(data_root, 'pretrain')
        train_list = os.path.join(pretrain_dir, 'train_list.txt')
        val_list   = os.path.join(pretrain_dir, 'val_list.txt')
        test_list  = os.path.join(pretrain_dir, 'test_list.txt')
        _has_val_dir = os.path.isdir(os.path.join(data_root, 'val'))

        print(f'[Env] Kaggle (pre-split): {data_root}')
        return {
            'is_kaggle': True,
            'train_image_dir': os.path.join(data_root, 'train', 'image'),
            'train_mask_dir':  os.path.join(data_root, 'train', 'mask'),
            'val_image_dir':   os.path.join(data_root, 'val' if _has_val_dir else 'train', 'image'),
            'val_mask_dir':    os.path.join(data_root, 'val' if _has_val_dir else 'train', 'mask'),
            'test_image_dir':  os.path.join(data_root, 'test', 'image'),
            'test_mask_dir':   os.path.join(data_root, 'test', 'mask'),
            'pretrain_dir':    pretrain_dir,
            'record_path':     '/kaggle/working/result',
            'train_valid_list': train_list if os.path.isfile(train_list) else None,
            'val_valid_list':   val_list   if os.path.isfile(val_list)   else None,
            'test_valid_list':  test_list  if os.path.isfile(test_list)  else None,
        }

    # ---- 本地环境 ----
    _base = r'E:\月球_dataset\dataset'
    _filter_dir = os.path.join(_base, 'dataset_analysis')
    os.makedirs(_filter_dir, exist_ok=True)

    # v5 (512 tile, 5% 分层随机 val — 当前主力)
    _v5 = os.path.join(_base, 'datasetv5')
    if os.path.isdir(_v5):
        _tl = os.path.join(_filter_dir, 'valid_tiles_train_v5.txt')
        _vl = os.path.join(_filter_dir, 'valid_tiles_val_v5.txt')
        if not os.path.isfile(_vl):
            print('[Env] v5: 生成 5% 分层随机 val...')
            import random as _rnd, numpy as _np, rasterio
            from collections import defaultdict
            _rnd.seed(42); _np.random.seed(42)
            _cls_tiles = defaultdict(list)
            for t in sorted(f for f in os.listdir(os.path.join(_v5, 'train', 'image'))
                           if f.endswith('.tif')):
                with rasterio.open(os.path.join(_v5, 'train', 'mask', t)) as src:
                    m = src.read(1).astype(_np.int64)
                m[(m > 4) | (m < 0)] = 0
                dom = max(range(1, 5), key=lambda c: int(_np.sum(m == c))) \
                    if _np.sum(m > 0) > 0 else 0
                _cls_tiles[dom].append(t)
            _tr, _va = [], []
            for c in range(5):
                ts = _cls_tiles[c]; _rnd.shuffle(ts)
                nv = max(1, int(len(ts) * 0.05))
                _va += ts[:nv]; _tr += ts[nv:]
            for path, tiles in [(_tl, _tr), (_vl, _va)]:
                with open(path, 'w', encoding='utf-8') as f:
                    for t in sorted(tiles): f.write(t + '\n')
            print(f'[Env] v5 Train={len(_tr)}, Val={len(_va)}')
        print(f'[Env] 本地 v5 (5% Stratified Val): {_v5}')
        return {
            'is_kaggle': False,
            'train_image_dir': os.path.join(_v5, 'train', 'image'),
            'train_mask_dir':  os.path.join(_v5, 'train', 'mask'),
            'val_image_dir':   os.path.join(_v5, 'train', 'image'),
            'val_mask_dir':    os.path.join(_v5, 'train', 'mask'),
            'test_image_dir':  os.path.join(_v5, 'test', 'image'),
            'test_mask_dir':   os.path.join(_v5, 'test', 'mask'),
            'pretrain_dir':    None,
            'record_path':     os.path.join(_base, 'result'),
            'train_valid_list': _tl,
            'val_valid_list':   _vl,
            'test_valid_list':  None,
        }

    # fallback: v10
    _v10 = os.path.join(_base, 'datasetv10')
    if os.path.isdir(_v10):
        _pt = os.path.join(_v10, 'pretrain')
        print(f'[Env] 本地 v10: {_v10}')
        return {
            'is_kaggle': False,
            'train_image_dir': os.path.join(_v10, 'train', 'image'),
            'train_mask_dir':  os.path.join(_v10, 'train', 'mask'),
            'val_image_dir':   os.path.join(_v10, 'val', 'image'),
            'val_mask_dir':    os.path.join(_v10, 'val', 'mask'),
            'test_image_dir':  os.path.join(_v10, 'test', 'image'),
            'test_mask_dir':   os.path.join(_v10, 'test', 'mask'),
            'pretrain_dir':    None,
            'record_path':     os.path.join(_base, 'result'),
            'train_valid_list': os.path.join(_pt, 'train_list.txt') if os.path.isfile(os.path.join(_pt, 'train_list.txt')) else None,
            'val_valid_list':   os.path.join(_pt, 'val_list.txt')   if os.path.isfile(os.path.join(_pt, 'val_list.txt'))   else None,
            'test_valid_list':  os.path.join(_pt, 'test_list.txt')  if os.path.isfile(os.path.join(_pt, 'test_list.txt'))  else None,
        }

    raise FileNotFoundError(f'未找到任何数据集 (v5/v10), base={_base}')


# ============================================================================
# 配置
# ============================================================================

def load_yaml_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class HyperParameter:
    def __init__(self, env=None, config=None):
        if env is None:
            env = detect_env()

        if isinstance(config, str):
            config = load_yaml_config(config)
        elif config is None:
            config = {}

        self.is_kaggle = env['is_kaggle']
        self.pretrain_dir = env['pretrain_dir']

        curr_time = datetime.datetime.now()
        curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
        self.name = "_result" + curr_time_str

        # ---- 核心参数 ----
        self.num_classes    = config.get('num_classes', 5)
        self.in_channels    = config.get('in_channels', 5)
        self.img_size       = config.get('img_size', 512)
        self.model_size     = config.get('model_size', 'small')
        self.freeze_stages  = config.get('freeze_stages', 1)

        self.num_epochs     = config.get('num_epochs', 100)
        self.max_steps      = config.get('max_steps', 0)
        self.batch_size     = config.get('batch_size', 4)
        self.accum_steps    = config.get('accum_steps', 1)
        self.learning_rate  = config.get('learning_rate', 5e-5)

        # ---- 损失函数 ----
        self.bce_weight   = config.get('bce_weight', 1.0)
        self.dice_weight  = config.get('dice_weight', 1.0)
        self.ce_weight    = config.get('ce_weight', 1.0)
        self.class_weights_4 = config.get('class_weights_4', [1.0, 1.3, 1.8, 2.5])
        # WR/Rille/Fault/Graben 的四分类权重 (不含背景)

        # ---- 优化器正则 ----
        self.weight_decay   = config.get('weight_decay', 0.01)
        self.grad_clip_norm = config.get('grad_clip_norm', 0.0)

        # ---- 数据增强 ----
        self.use_augment   = config.get('use_augment', True)
        self.use_copypaste = config.get('use_copypaste', True)
        self.copypaste_p   = config.get('copypaste_p', 0.5)
        self.use_scale_aug = config.get('use_scale_aug', False)
        self.scale_range   = config.get('scale_range', [0.9, 1.1])

        # ---- Tile 采样 ----
        self.use_tile_sampling = config.get('use_tile_sampling', True)
        self.tile_sample_exp  = config.get('tile_sample_exp', 0.3)

        # ---- Early stopping ----
        self.use_val    = config.get('use_val', True)
        self.early_stop = config.get('early_stop', True)
        self.patience   = config.get('patience', 12)

        # ---- 导出开关 ----
        self.save_val_on_best     = config.get('save_val_on_best', True)
        self.save_all_test_on_best = config.get('save_all_test_on_best', True)
        self.save_pred_mask_png   = config.get('save_pred_mask_png', True)
        self.save_pred_vis_png    = config.get('save_pred_vis_png', True)

        # ---- 数据路径 ----
        self.train_image_dir = env['train_image_dir']
        self.train_mask_dir  = env['train_mask_dir']
        self.val_image_dir   = env.get('val_image_dir', env['train_image_dir'])
        self.val_mask_dir    = env.get('val_mask_dir',  env['train_mask_dir'])
        self.test_image_dir  = env['test_image_dir']
        self.test_mask_dir   = env['test_mask_dir']

        self.train_valid_list = env['train_valid_list']
        self.val_valid_list   = env.get('val_valid_list', None)
        self.test_valid_list  = env['test_valid_list']

        # ---- 输出路径 ----
        self.record_path = env['record_path']
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


# ============================================================================
# 数据增强 Dataset
# ============================================================================

class AugmentedDataset(torch.utils.data.Dataset):
    """翻转 + 旋转 + 随机缩放 + 高斯噪声 + 亮度扰动 + Cutout + CopyPaste"""
    def __init__(self, base_dataset, copypaste=False, copypaste_p=0.5,
                 scale_aug=False, scale_range=(0.8, 1.2)):
        self.base = base_dataset
        self.copypaste = copypaste
        self.copypaste_p = copypaste_p
        self.scale_aug = scale_aug
        self.scale_range = scale_range

        if self.copypaste:
            self.rare_indices = []
            print('CopyPaste: 预索引少数类样本...')
            for i in tqdm(range(len(self.base)), desc='CopyPaste索引', unit='img'):
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

        if self.scale_aug:
            _, H, W = img.shape
            scale = random.uniform(*self.scale_range)
            new_h, new_w = int(H * scale), int(W * scale)
            if scale < 1.0:
                img_small = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                          mode='bilinear', align_corners=False)
                mask_small = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                           size=(new_h, new_w), mode='nearest')
                img = F.pad(img_small.squeeze(0), (0, W - new_w, 0, H - new_h), value=0)
                mask = F.pad(mask_small.squeeze(0).squeeze(0),
                             (0, W - new_w, 0, H - new_h), value=0).long()
            else:
                img_big = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                        mode='bilinear', align_corners=False)
                mask_big = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                         size=(new_h, new_w), mode='nearest')
                y0 = random.randint(0, new_h - H)
                x0 = random.randint(0, new_w - W)
                img = img_big[:, :, y0:y0+H, x0:x0+W].squeeze(0)
                mask = mask_big[:, :, y0:y0+H, x0:x0+W].squeeze(0).squeeze(0).long()

        # 几何增强
        if random.random() > 0.5:
            img = img.flip(-1); mask = mask.flip(-1)
        if random.random() > 0.5:
            img = img.flip(-2); mask = mask.flip(-2)
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        # 像素增强
        if random.random() > 0.5:
            img = img + torch.randn_like(img) * 0.02
        if random.random() > 0.5:
            C = img.shape[0]
            img = img + (torch.rand(C, 1, 1) - 0.5) * 0.1
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
# Weighted Tile Sampler (与 STDL-Net R50 一致)
# ============================================================================

def build_weighted_sampler(dataset, exp=0.3, num_samples=None):
    """按前景占比加权采样: weight = fg_ratio ** exp."""
    weights = []
    for i in range(len(dataset)):
        _, mask, _ = dataset[i]
        fg = (mask > 0).float().mean().item()
        weights.append((fg + 1e-8) ** exp)
    w = torch.tensor(weights, dtype=torch.float64)
    if num_samples is None:
        num_samples = len(dataset)
    return torch.utils.data.WeightedRandomSampler(w, num_samples, replacement=True)


# ============================================================================
# 导出 & 可视化
# ============================================================================

def export_all_val(model, data_iter, save_root, epoch, num_classes, device,
                   use_cuda=True, save_mask=True, save_vis=True):
    if not save_mask and not save_vis:
        return

    mask_dir = os.path.join(save_root, 'pred_mask')
    vis_dir  = os.path.join(save_root, 'pred_vis')
    if save_mask: os.makedirs(mask_dir, exist_ok=True)
    if save_vis:  os.makedirs(vis_dir, exist_ok=True)

    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()

    model.eval()
    with torch.no_grad(), amp_ctx:
        for img, label, name in tqdm(data_iter, desc=f'Export@E{epoch}', unit='img'):
            img, label = img.to(device), label.to(device)
            pred = model.predict(img)

            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            gt_np   = label[0].cpu().numpy().astype(np.uint8)
            stem    = name[0]

            if save_mask:
                Image.fromarray(pred_np, mode='L').save(
                    os.path.join(mask_dir, f'{stem}.png'))

            if save_vis:
                x = img[0].cpu().numpy()
                wac = x[0] * CHANNEL_STD[0] + CHANNEL_MEAN[0]
                wac = np.clip(wac, 0, 1)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(wac, cmap='gray')
                axes[0].set_title(f'WAC - {stem}', fontsize=8)
                axes[1].imshow(mask_to_color(gt_np))
                axes[1].set_title('GT')
                axes[2].imshow(mask_to_color(pred_np))
                axes[2].set_title('Pred')
                axes[3].imshow(error_map(gt_np, pred_np))
                axes[3].set_title('Error(G=TP,R=FN,O=FP)')
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'{stem}.png'), dpi=120)
                plt.close(fig)

    saved_mask = len(os.listdir(mask_dir)) if save_mask else 0
    saved_vis = len(os.listdir(vis_dir)) if save_vis else 0
    print(f'已导出 {saved_mask} 张 pred_mask + {saved_vis} 张 pred_vis 到 {save_root}')


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
            pred = model.predict(img)

            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            gt_np   = label[0].cpu().numpy().astype(np.uint8)
            stem    = name[0]

            Image.fromarray(pred_np, mode='L').save(
                os.path.join(mask_dir, f'{stem}.png'))

            x = img[0].cpu().numpy()
            wac = x[0] * CHANNEL_STD[0] + CHANNEL_MEAN[0]
            wac = np.clip(wac, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(wac, cmap='gray')
            axes[0].set_title(f'WAC - {stem}', fontsize=8)
            axes[1].imshow(mask_to_color(gt_np))
            axes[1].set_title('GT')
            axes[2].imshow(mask_to_color(pred_np))
            axes[2].set_title('Pred')
            axes[3].imshow(error_map(gt_np, pred_np))
            axes[3].set_title('Error(G=TP,R=FN,O=FP)')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{stem}.png'), dpi=120)
            plt.close(fig)

    print(f'已导出 {len(os.listdir(mask_dir))} 张 pred_mask + '
          f'{len(os.listdir(vis_dir))} 张 pred_vis')


def plot_training_curves(history, save_path, num_classes):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ep = history['epoch']

    axes[0, 0].plot(ep, history['train_loss'], 'o-', ms=3, label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(ep, history['val_loss'], '^-', ms=3, label='Val')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history['train_miou'], 'o-', ms=3, label='Train')
    if 'val_miou' in history:
        axes[0, 1].plot(ep, history['val_miou'], '^-', ms=3, label='Val')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('mIoU Curve'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    arr = np.array(history['train_iou_per_class'])
    for c in range(num_classes):
        axes[1, 0].plot(ep, arr[:, c], 'o-', ms=2, lw=1.5,
                        color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Train Per-Class IoU')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    if 'val_iou_per_class' in history:
        arr_val = np.array(history['val_iou_per_class'])
        for c in range(num_classes):
            axes[1, 1].plot(ep, arr_val[:, c], '^-', ms=2, lw=1.5,
                            color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Val Per-Class IoU')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

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
        print('WARNING: 本地 CPU 模式, 仅用于调试流程.')

    os.makedirs(hp.result_dir, exist_ok=True)

    # ---- Kaggle: 替换 backbone 内置预训练路径 ----
    if hp.is_kaggle and hp.pretrain_dir:
        import models.backbone as _bb
        _bb._SWIN_V2_PRETRAINED = {
            'tiny':  os.path.join(hp.pretrain_dir, 'swinv2_tiny_patch4_window16_256.pth'),
            'small': os.path.join(hp.pretrain_dir, 'swinv2_small_patch4_window16_256.pth'),
            'base':  os.path.join(hp.pretrain_dir,
                                  'swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'),
        }
        print('Kaggle: 已注入预训练路径')

    # ---- 模型 ----
    model = LDNet(
        size=hp.model_size,
        img_size=hp.img_size,
        in_channels=hp.in_channels,
        pretrained=True,
    ).to(device)

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
    if getattr(hp, 'use_val', True):
        val_data = MyDataset(
            images_dir=hp.val_image_dir,
            masks_dir=hp.val_mask_dir,
            valid_list_file=hp.val_valid_list,
        )
    else:
        import tempfile
        _tmp = tempfile.mkdtemp()
        val_data = MyDataset(_tmp, _tmp)
        hp.val_valid_list = None
    test_data = MyDataset(
        images_dir=hp.test_image_dir,
        masks_dir=hp.test_mask_dir,
        valid_list_file=hp.test_valid_list,
    )

    if hp.use_augment:
        train_data = AugmentedDataset(
            train_data_raw, copypaste=hp.use_copypaste, copypaste_p=hp.copypaste_p,
            scale_aug=hp.use_scale_aug, scale_range=hp.scale_range)
    else:
        train_data = train_data_raw

    # Weighted sampler
    if hp.use_tile_sampling:
        sampler = build_weighted_sampler(train_data_raw, exp=hp.tile_sample_exp)
        shuffle = False
        print(f'Tile sampling: exp={hp.tile_sample_exp}, '
              f'{len(train_data_raw)} → {len(sampler)} samples/epoch')
    else:
        sampler = None
        shuffle = True

    train_iter = DataLoader(train_data, batch_size=hp.batch_size,
                            shuffle=shuffle if sampler is None else None,
                            sampler=sampler,
                            num_workers=2 if use_cuda else 0,
                            pin_memory=use_cuda, drop_last=True)
    val_iter   = DataLoader(val_data,   batch_size=1, shuffle=False,
                            num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    test_iter  = DataLoader(test_data,  batch_size=1, shuffle=False,
                            num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

    print(f'训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张, '
          f'测试集: {len(test_data)} 张')

    # ---- 损失 ----
    loss_fn = DualBranchLoss(
        bce_weight=hp.bce_weight,
        dice_weight=hp.dice_weight,
        ce_weight=hp.ce_weight,
        class_weights=hp.class_weights_4 if hasattr(hp, 'class_weights_4') else None,
    ).to(device)
    print(f'Loss: {hp.bce_weight}*BCE + {hp.dice_weight}*Dice (B1) + '
          f'{hp.ce_weight}*CE (B2, 4-class weights={hp.class_weights_4})')

    # ---- 优化器 & 调度器 ----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp.num_epochs, eta_min=1e-6)
    print(f'Optimizer: AdamW(lr={hp.learning_rate}, wd={hp.weight_decay})')

    # ---- 训练状态 ----
    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    best_miou = 0.0
    best_export_dir = None
    no_improve_count = 0

    history = {
        'epoch': [],
        'train_loss': [], 'train_miou': [], 'train_acc': [],
        'val_loss':   [], 'val_miou':   [], 'val_acc':   [],
        'train_iou_per_class': [], 'val_iou_per_class': [],
    }

    csv_path = os.path.join(hp.result_dir, 'epoch_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['epoch', 'train_loss', 'val_loss',
                  'train_miou', 'val_miou', 'train_acc', 'val_acc']
        for c in range(hp.num_classes):
            header += [f'train_iou_{c}', f'val_iou_{c}']
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
                b1_logit, b2_logit = model(img)
                total_loss, loss_dict = loss_fn(b1_logit, b2_logit, label)
                l = total_loss / hp.accum_steps

            if scaler is not None:
                scaler.scale(l).backward()
                if (step + 1) % hp.accum_steps == 0 or (step + 1) == len(train_iter):
                    if hp.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), hp.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                l.backward()
                if (step + 1) % hp.accum_steps == 0 or (step + 1) == len(train_iter):
                    if hp.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), hp.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            train_losses.append(l.item() * hp.accum_steps)
            with torch.no_grad():
                pred = merge_branches(b1_logit, b2_logit)
                hist += metrics.multiclass_confusion(
                    pred, label, hp.num_classes).double()

            if step % 50 == 0:
                pbar.set_postfix(loss=f'{np.mean(train_losses[-50:]):.4f}')

        m = metrics.metrics_from_hist(hist)
        avg_loss = float(np.mean(train_losses))
        print(f'\n[Train] loss: {avg_loss:.4f}  mIoU: {m["miou"]:.4f}  '
              f'acc: {m["accuracy"]:.4f}')
        print('  IoU:', ' '.join(f'{v:.4f}' for v in m['iou_per_class']))

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_losses = []
        val_hist = torch.zeros(hp.num_classes, hp.num_classes, dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, name in tqdm(val_iter, desc='Validating', unit='img'):
                if hp.max_steps > 0 and len(val_losses) >= hp.max_steps:
                    break
                img, label = img.to(device), label.to(device)
                b1_logit, b2_logit = model(img)
                _, ld = loss_fn(b1_logit, b2_logit, label)
                val_losses.append(ld['total'])
                pred = merge_branches(b1_logit, b2_logit)
                val_hist += metrics.multiclass_confusion(
                    pred, label, hp.num_classes).double()

        vm = metrics.metrics_from_hist(val_hist)
        val_avg_loss = float(np.mean(val_losses))
        print(f'[Val]   loss: {val_avg_loss:.4f}  mIoU: {vm["miou"]:.4f}  '
              f'acc: {vm["accuracy"]:.4f}')
        print('  IoU:', ' '.join(f'{v:.4f}' for v in vm['iou_per_class']))

        # --- Record ---
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['train_miou'].append(float(m['miou']))
        history['train_acc'].append(float(m['accuracy']))
        history['val_loss'].append(val_avg_loss)
        history['val_miou'].append(float(vm['miou']))
        history['val_acc'].append(float(vm['accuracy']))
        history['train_iou_per_class'].append([float(v) for v in m['iou_per_class']])
        history['val_iou_per_class'].append([float(v) for v in vm['iou_per_class']])

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            row = [epoch, f'{avg_loss:.6f}', f'{val_avg_loss:.6f}',
                   f'{float(m["miou"]):.6f}', f'{float(vm["miou"]):.6f}',
                   f'{float(m["accuracy"]):.6f}', f'{float(vm["accuracy"]):.6f}']
            for c in range(hp.num_classes):
                row += [f'{m["iou_per_class"][c]:.6f}',
                        f'{vm["iou_per_class"][c]:.6f}']
            csv.writer(f).writerow(row)

        with open(os.path.join(hp.result_dir, 'history.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # --- Save best ---
        no_val = (len(val_data) == 0)
        if no_val:
            if epoch % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(hp.result_dir, f'best_{hp.model_size}.pth'))
        elif vm['miou'] > best_miou:
            best_miou = float(vm['miou'])
            no_improve_count = 0
            torch.save(model.state_dict(),
                       os.path.join(hp.result_dir, f'best_{hp.model_size}.pth'))
            print(f'>>> Best model saved! val_mIoU={best_miou:.4f}')
            if hp.save_val_on_best:
                if best_export_dir and os.path.isdir(best_export_dir):
                    shutil.rmtree(best_export_dir, ignore_errors=True)
                best_export_dir = os.path.join(
                    hp.result_dir, f'best_epoch_{epoch:02d}_miou_{best_miou:.4f}')
                export_all_val(
                    model, val_iter, best_export_dir, epoch,
                    hp.num_classes, device, use_cuda=use_cuda,
                    save_mask=hp.save_pred_mask_png, save_vis=hp.save_pred_vis_png)
        else:
            no_improve_count += 1
            print(f'  No improvement ({no_improve_count}/{hp.patience})')

        # --- Checkpoint ---
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(hp.result_dir, f'ckpt_epoch{epoch}.pth'))

        # --- Early stopping ---
        if not no_val and hp.early_stop and no_improve_count >= hp.patience:
            print(f'\n*** Early stopping at epoch {epoch} ***')
            break

    # ---- Final model ----
    torch.save(model.state_dict(),
               os.path.join(hp.result_dir, f'final_{hp.model_size}.pth'))
    print(f'Final model saved.')

    # ---- Final test evaluation ----
    if len(test_data) > 0:
        print('\n' + '=' * 72)
        print('>>> Final Test Evaluation (best val checkpoint)')
        print('=' * 72)

        best_path = os.path.join(hp.result_dir, f'best_{hp.model_size}.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f'Loaded best checkpoint: {best_path}')
        else:
            print('No best checkpoint found, using current model.')

        model.eval()
        test_losses = []
        test_hist = torch.zeros(hp.num_classes, hp.num_classes, dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, name in tqdm(test_iter, desc='Final Test', unit='img'):
                img, label = img.to(device), label.to(device)
                b1_logit, b2_logit = model(img)
                _, ld = loss_fn(b1_logit, b2_logit, label)
                test_losses.append(ld['total'])
                pred = merge_branches(b1_logit, b2_logit)
                test_hist += metrics.multiclass_confusion(
                    pred, label, hp.num_classes).double()

        tm = metrics.metrics_from_hist(test_hist)
        test_avg_loss = float(np.mean(test_losses))
        print('\n' + '=' * 60)
        print(f'  Final Test | loss: {test_avg_loss:.4f}  '
              f'mIoU: {tm["miou"]:.4f}  acc: {tm["accuracy"]:.4f}')
        for c in range(hp.num_classes):
            print(f'  {CLASS_NAMES[c]:<15}: IoU={tm["iou_per_class"][c]:.4f}  '
                  f'P={tm["precision_per_class"][c]:.4f}  '
                  f'R={tm["recall_per_class"][c]:.4f}')
        print('=' * 60)

        if hp.save_all_test_on_best:
            best_export_dir = os.path.join(
                hp.result_dir, f'final_test_miou_{tm["miou"]:.4f}')
            export_all_test(model, test_iter, best_export_dir, epoch,
                            hp.num_classes, device, use_cuda=use_cuda)

    # ---- Training curves ----
    plot_training_curves(history,
                         os.path.join(hp.result_dir, 'training_curves.png'),
                         hp.num_classes)

    # ---- Kaggle zip ----
    if hp.is_kaggle:
        import zipfile
        zip_path = '/kaggle/working/result.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(hp.result_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    zf.write(fpath, os.path.relpath(fpath, '/kaggle/working'))
        print(f'Result zip: {zip_path} '
              f'({os.path.getsize(zip_path)/1024/1024:.1f} MB)')

    print(f'\nTraining done! Best mIoU = {best_miou:.4f}')
    return model, best_miou


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDNet 双分支分割训练')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径 (如 configs/R1.yaml)')
    args = parser.parse_args()

    env = detect_env()
    hp = HyperParameter(env, config=args.config)

    print(f'Environment: {"Kaggle" if hp.is_kaggle else "Local"}')
    print(f'Data: {hp.train_image_dir}')
    print(f'Model: {hp.model_size}, Epochs: {hp.num_epochs}, '
          f'BS: {hp.batch_size}x{hp.accum_steps}='
          f'{hp.batch_size * hp.accum_steps}, LR: {hp.learning_rate}')
    print(f'Freeze: {hp.freeze_stages} stages')
    print(f'Aug: {hp.use_augment}, Scale: {hp.use_scale_aug}, '
          f'CopyPaste: {hp.use_copypaste}(p={hp.copypaste_p})')
    print(f'TileSampling: {hp.use_tile_sampling} (exp={hp.tile_sample_exp})')
    print(f'Result dir: {hp.result_dir}')

    train(hp)
