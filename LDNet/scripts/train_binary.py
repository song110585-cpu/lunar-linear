"""
LDNet-Binary 训练脚本 — Step 1: 线 vs 背景

输入: 5ch × 512×512
标签: 自动将 1-4 类合并为 1 (线性构造)
Loss: BCE + Dice
模型: LDNetBinary (复用 STDL-Net R50 decoder 架构)

用法: python scripts/train_binary.py --config configs/R2_binary.yaml
"""
import os, csv, json, shutil, datetime, random, argparse, yaml
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import sys as _sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models', 'losses']:
    _p = os.path.join(_root, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
from ldnet_binary import LDNetBinary
from binary_loss import BinaryLoss, label_to_binary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================================
# 环境检测 (与 train.py 相同)
# ============================================================================

def detect_env():
    if os.path.isdir('/kaggle'):
        data_root_v5 = '/kaggle/input/datasets/yuanssy/dataset5/datasetv5'
        data_root_v10 = '/kaggle/input/datasets/yuanssy/dataset10/datasetv10'
        data_root_v9 = '/kaggle/input/datasets/changyasong/dataset9/datasetv9'
        data_root_v8 = '/kaggle/input/datasets/changyasong/dataset8/datasetv8'

        if os.path.isdir(data_root_v5):
            data_root = data_root_v5
            pretrain_dir = os.path.join(data_root, 'pretrain')
            os.makedirs('/kaggle/working', exist_ok=True)
            _tl = '/kaggle/working/v5_train_list.txt'
            _vl = '/kaggle/working/v5_val_list.txt'
            if not os.path.isfile(_vl):
                print('[Env] Kaggle v5: 生成 5% 分层随机 val...')
                import random as _rnd, numpy as _np, rasterio
                _rnd.seed(42); _np.random.seed(42)
                _cls_tiles = defaultdict(list)
                _train_img = os.path.join(data_root, 'train', 'image')
                for t in sorted(f for f in os.listdir(_train_img) if f.endswith('.tif')):
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
            return _make_env(data_root, pretrain_dir, '/kaggle/working/result', _tl, _vl, None, True)

        for _dr in [data_root_v10, data_root_v9, data_root_v8]:
            if os.path.isdir(_dr):
                data_root = _dr; break
        else:
            raise FileNotFoundError('Kaggle: 未找到任何数据集')

        pretrain_dir = os.path.join(data_root, 'pretrain')
        _has_val = os.path.isdir(os.path.join(data_root, 'val'))
        return _make_env(data_root, pretrain_dir, '/kaggle/working/result',
                         os.path.join(pretrain_dir, 'train_list.txt'),
                         os.path.join(pretrain_dir, 'val_list.txt'),
                         os.path.join(pretrain_dir, 'test_list.txt'), True, _has_val)

    # ---- 本地 ----
    _base = r'E:\月球_dataset\dataset'
    _filter_dir = os.path.join(_base, 'dataset_analysis')
    os.makedirs(_filter_dir, exist_ok=True)

    _v5 = os.path.join(_base, 'datasetv5')
    if os.path.isdir(_v5):
        _tl = os.path.join(_filter_dir, 'valid_tiles_train_v5.txt')
        _vl = os.path.join(_filter_dir, 'valid_tiles_val_v5.txt')
        if not os.path.isfile(_vl):
            print('[Env] v5: 生成 5% 分层随机 val...')
            import random as _rnd, numpy as _np, rasterio
            _rnd.seed(42); _np.random.seed(42)
            _cls_tiles = defaultdict(list)
            for t in sorted(f for f in os.listdir(os.path.join(_v5, 'train', 'image')) if f.endswith('.tif')):
                with rasterio.open(os.path.join(_v5, 'train', 'mask', t)) as src:
                    m = src.read(1).astype(_np.int64)
                m[(m > 4) | (m < 0)] = 0
                dom = max(range(1, 5), key=lambda c: int(_np.sum(m == c))) if _np.sum(m > 0) > 0 else 0
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
        return _make_env(_v5, None, os.path.join(_base, 'result'), _tl, _vl, None, False)

    raise FileNotFoundError(f'未找到数据集')


def _make_env(data_root, pretrain_dir, record_path, tl, vl, test_l, is_kaggle, has_val=True):
    """构建 env dict."""
    val_img = os.path.join(data_root, 'val' if has_val else 'train', 'image')
    val_msk = os.path.join(data_root, 'val' if has_val else 'train', 'mask')
    def _exists(p): return p if (p and os.path.isfile(p)) else None
    return {
        'is_kaggle': is_kaggle,
        'train_image_dir': os.path.join(data_root, 'train', 'image'),
        'train_mask_dir':  os.path.join(data_root, 'train', 'mask'),
        'val_image_dir':   val_img,
        'val_mask_dir':    val_msk,
        'test_image_dir':  os.path.join(data_root, 'test', 'image'),
        'test_mask_dir':   os.path.join(data_root, 'test', 'mask'),
        'pretrain_dir':    pretrain_dir,
        'record_path':     record_path,
        'train_valid_list': _exists(tl),
        'val_valid_list':   _exists(vl),
        'test_valid_list':  _exists(test_l),
    }


# ============================================================================
# 配置
# ============================================================================

def load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class HyperParameter:
    def __init__(self, env=None, config=None):
        if env is None: env = detect_env()
        if isinstance(config, str): config = load_yaml_config(config)
        elif config is None: config = {}

        self.is_kaggle = env['is_kaggle']
        self.pretrain_dir = env['pretrain_dir']

        curr_time = datetime.datetime.now()
        self.name = "_binary" + curr_time.strftime("_%Y%m%d_%H%M%S")

        self.img_size       = config.get('img_size', 512)
        self.in_channels    = config.get('in_channels', 5)
        self.model_size     = config.get('model_size', 'small')
        self.freeze_stages  = config.get('freeze_stages', 1)
        self.num_epochs     = config.get('num_epochs', 80)
        self.batch_size     = config.get('batch_size', 4)
        self.accum_steps    = config.get('accum_steps', 1)
        self.learning_rate  = config.get('learning_rate', 5e-5)
        self.weight_decay   = config.get('weight_decay', 0.01)
        self.grad_clip_norm = config.get('grad_clip_norm', 0.0)

        self.bce_weight   = config.get('bce_weight', 1.0)
        self.dice_weight  = config.get('dice_weight', 1.0)
        self.pos_weight   = config.get('pos_weight', 2.0)

        self.use_augment   = config.get('use_augment', True)
        self.use_copypaste = config.get('use_copypaste', True)
        self.copypaste_p   = config.get('copypaste_p', 0.5)
        self.use_scale_aug = config.get('use_scale_aug', False)
        self.scale_range   = config.get('scale_range', [0.9, 1.1])

        self.use_tile_sampling = config.get('use_tile_sampling', True)
        self.tile_sample_exp  = config.get('tile_sample_exp', 0.3)

        self.early_stop = config.get('early_stop', True)
        self.patience   = config.get('patience', 12)

        self.save_val_on_best      = config.get('save_val_on_best', True)
        self.save_all_test_on_best = config.get('save_all_test_on_best', True)
        self.save_pred_mask_png    = config.get('save_pred_mask_png', True)
        self.save_pred_vis_png     = config.get('save_pred_vis_png', True)

        self.train_image_dir = env['train_image_dir']
        self.train_mask_dir  = env['train_mask_dir']
        self.val_image_dir   = env.get('val_image_dir', env['train_image_dir'])
        self.val_mask_dir    = env.get('val_mask_dir',  env['train_mask_dir'])
        self.test_image_dir  = env['test_image_dir']
        self.test_mask_dir   = env['test_mask_dir']
        self.train_valid_list = env['train_valid_list']
        self.val_valid_list   = env.get('val_valid_list', None)
        self.test_valid_list  = env['test_valid_list']

        self.record_path = env['record_path']
        self.result_dir = os.path.join(self.record_path, self.name)


# ============================================================================
# 可视化
# ============================================================================

CLASS_COLORS = np.array([[0,0,0],[255,255,255]], dtype=np.uint8)


def mask_to_color(mask):
    return CLASS_COLORS[np.clip(mask.astype(np.int64), 0, 1)]


def error_map(gt, pred):
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    gt_fg, pr_fg = gt > 0, pred > 0
    out[gt_fg & pr_fg]    = [0, 200, 0]
    out[gt_fg & (~pr_fg)] = [255, 0, 0]
    out[(~gt_fg) & pr_fg] = [255, 165, 0]
    return out


# ============================================================================
# 数据增强 Dataset (复用)
# ============================================================================

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, copypaste=False, copypaste_p=0.5,
                 scale_aug=False, scale_range=(0.8, 1.2)):
        self.base = base_dataset
        self.copypaste = copypaste
        self.copypaste_p = copypaste_p
        self.scale_aug = scale_aug
        self.scale_range = scale_range
        if self.copypaste:
            self.rare_indices = []
            for i in tqdm(range(len(self.base)), desc='CopyPaste索引', unit='img'):
                _, mask, _ = self.base[i]
                if 3 in mask.unique() or 4 in mask.unique():
                    self.rare_indices.append(i)
            print(f'CopyPaste: {len(self.rare_indices)} 张含 Fault/Graben')

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, mask, name = self.base[idx]
        if self.copypaste and self.rare_indices and random.random() < self.copypaste_p:
            donor_idx = random.choice(self.rare_indices)
            donor_img, donor_mask, _ = self.base[donor_idx]
            rare_fg = (donor_mask == 3) | (donor_mask == 4)
            if rare_fg.any():
                img = torch.where(rare_fg.unsqueeze(0).expand_as(img), donor_img, img)
                mask = torch.where(rare_fg, donor_mask, mask)

        if self.scale_aug:
            _, H, W = img.shape
            scale = random.uniform(*self.scale_range)
            new_h, new_w = int(H * scale), int(W * scale)
            if scale < 1.0:
                img_s = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
                mask_s = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w), mode='nearest')
                img = F.pad(img_s.squeeze(0), (0, W-new_w, 0, H-new_h), value=0)
                mask = F.pad(mask_s.squeeze(0).squeeze(0), (0, W-new_w, 0, H-new_h), value=0).long()
            else:
                img_b = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
                mask_b = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w), mode='nearest')
                y0, x0 = random.randint(0, new_h-H), random.randint(0, new_w-W)
                img = img_b[:,:,y0:y0+H,x0:x0+W].squeeze(0)
                mask = mask_b[:,:,y0:y0+H,x0:x0+W].squeeze(0).squeeze(0).long()

        if random.random() > 0.5: img = img.flip(-1); mask = mask.flip(-1)
        if random.random() > 0.5: img = img.flip(-2); mask = mask.flip(-2)
        k = random.randint(0, 3)
        if k > 0: img = torch.rot90(img, k, [-2,-1]); mask = torch.rot90(mask, k, [-2,-1])

        if random.random() > 0.5: img = img + torch.randn_like(img) * 0.02
        if random.random() > 0.5: img = img + (torch.rand(img.shape[0],1,1)-0.5)*0.1
        if random.random() > 0.7:
            _, H, W = img.shape
            for _ in range(random.randint(1,3)):
                s = random.randint(32,64); y = random.randint(0,H-s); x = random.randint(0,W-s)
                img[:,y:y+s,x:x+s] = 0.0
        return img, mask, name


# ============================================================================
# Weighted Sampler
# ============================================================================

def build_weighted_sampler(dataset, exp=0.3):
    weights = []
    for i in range(len(dataset)):
        _, mask, _ = dataset[i]
        fg = (mask > 0).float().mean().item()
        weights.append((fg + 1e-8) ** exp)
    return torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64), len(dataset), replacement=True)


# ============================================================================
# 二分类指标
# ============================================================================

def binary_metrics(hist_2x2):
    """从 2x2 混淆矩阵计算二分类指标."""
    tn, fp = hist_2x2[0, 0].item(), hist_2x2[0, 1].item()
    fn, tp = hist_2x2[1, 0].item(), hist_2x2[1, 1].item()
    eps = 1e-10
    return {
        'iou':  tp / (tp + fp + fn + eps),
        'prec': tp / (tp + fp + eps),
        'rec':  tp / (tp + fn + eps),
        'acc':  (tp + tn) / (tp + tn + fp + fn + eps),
    }


# ============================================================================
# 导出
# ============================================================================

def export_all_val(model, data_iter, save_root, epoch, device, use_cuda,
                   save_mask=True, save_vis=True):
    if not save_mask and not save_vis: return
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
            pred_np = pred[0].cpu().numpy().astype(np.uint8); gt_np = label_to_binary(label[0]).cpu().numpy().astype(np.uint8)
            stem = name[0]
            if save_mask: Image.fromarray(pred_np*255, mode='L').save(os.path.join(mask_dir, f'{stem}.png'))
            if save_vis:
                x = img[0].cpu().numpy(); wac = np.clip(x[0]*CHANNEL_STD[0]+CHANNEL_MEAN[0], 0, 1)
                fig, axes = plt.subplots(1, 4, figsize=(16,4))
                axes[0].imshow(wac, cmap='gray'); axes[0].set_title(f'WAC-{stem}', fontsize=8)
                axes[1].imshow(mask_to_color(gt_np)); axes[1].set_title('GT')
                axes[2].imshow(mask_to_color(pred_np)); axes[2].set_title('Pred')
                axes[3].imshow(error_map(gt_np, pred_np)); axes[3].set_title('Error')
                for ax in axes: ax.axis('off')
                plt.tight_layout(); plt.savefig(os.path.join(vis_dir, f'{stem}.png'), dpi=120); plt.close(fig)
    print(f'导出 {len(os.listdir(mask_dir))} masks + {len(os.listdir(vis_dir))} vis')


def export_all_test(model, test_iter, save_root, epoch, device, use_cuda):
    mask_dir = os.path.join(save_root, 'pred_mask'); vis_dir = os.path.join(save_root, 'pred_vis')
    os.makedirs(mask_dir, exist_ok=True); os.makedirs(vis_dir, exist_ok=True)
    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()
    model.eval()
    with torch.no_grad(), amp_ctx:
        for img, label, name in tqdm(test_iter, desc=f'Export@E{epoch}', unit='img'):
            img, label = img.to(device), label.to(device)
            pred = model.predict(img)
            pred_np = pred[0].cpu().numpy().astype(np.uint8); gt_np = label_to_binary(label[0]).cpu().numpy().astype(np.uint8)
            stem = name[0]
            Image.fromarray(pred_np*255, mode='L').save(os.path.join(mask_dir, f'{stem}.png'))
            x = img[0].cpu().numpy(); wac = np.clip(x[0]*CHANNEL_STD[0]+CHANNEL_MEAN[0], 0, 1)
            fig, axes = plt.subplots(1, 4, figsize=(16,4))
            axes[0].imshow(wac, cmap='gray'); axes[0].set_title(f'WAC-{stem}', fontsize=8)
            axes[1].imshow(mask_to_color(gt_np)); axes[1].set_title('GT')
            axes[2].imshow(mask_to_color(pred_np)); axes[2].set_title('Pred')
            axes[3].imshow(error_map(gt_np, pred_np)); axes[3].set_title('Error')
            for ax in axes: ax.axis('off')
            plt.tight_layout(); plt.savefig(os.path.join(vis_dir, f'{stem}.png'), dpi=120); plt.close(fig)
    print(f'导出 {len(os.listdir(mask_dir))} masks + {len(os.listdir(vis_dir))} vis')


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ep = history['epoch']
    axes[0,0].plot(ep, history['train_loss'], 'o-', ms=3, label='Train')
    if 'val_loss' in history: axes[0,0].plot(ep, history['val_loss'], '^-', ms=3, label='Val')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss'); axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    axes[0,1].plot(ep, history['train_iou'], 'o-', ms=3, label='Train')
    if 'val_iou' in history: axes[0,1].plot(ep, history['val_iou'], '^-', ms=3, label='Val')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('IoU'); axes[0,1].set_title('IoU (class=1)'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    axes[1,0].plot(ep, history['train_rec'], 'o-', ms=3, label='Train')
    if 'val_rec' in history: axes[1,0].plot(ep, history['val_rec'], '^-', ms=3, label='Val')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Recall'); axes[1,0].set_title('Recall'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    axes[1,1].plot(ep, history['train_prec'], 'o-', ms=3, label='Train')
    if 'val_prec' in history: axes[1,1].plot(ep, history['val_prec'], '^-', ms=3, label='Val')
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Precision'); axes[1,1].set_title('Precision'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f'Saved: {save_path}')


# ============================================================================
# 训练
# ============================================================================

def train(hp: HyperParameter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f'Device: {device}')
    os.makedirs(hp.result_dir, exist_ok=True)

    # Kaggle: 注入预训练路径
    if hp.is_kaggle and hp.pretrain_dir:
        import backbone as _bb
        _bb._SWIN_V2_PRETRAINED = {
            'tiny':  os.path.join(hp.pretrain_dir, 'swinv2_tiny_patch4_window16_256.pth'),
            'small': os.path.join(hp.pretrain_dir, 'swinv2_small_patch4_window16_256.pth'),
            'base':  os.path.join(hp.pretrain_dir, 'swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'),
        }

    # ---- 模型 ----
    model = LDNetBinary(size=hp.model_size, img_size=hp.img_size,
                        in_channels=hp.in_channels, pretrained=True).to(device)
    if hp.freeze_stages > 0:
        model.freeze_backbone_stages(hp.freeze_stages)
    n = sum(p.numel() for p in model.parameters())/1e6
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f'Params: {n:.1f}M total, {nt:.1f}M trainable')

    # ---- 数据 ----
    train_raw = MyDataset(hp.train_image_dir, hp.train_mask_dir, valid_list_file=hp.train_valid_list)
    val_data   = MyDataset(hp.val_image_dir, hp.val_mask_dir, valid_list_file=hp.val_valid_list)
    test_data  = MyDataset(hp.test_image_dir, hp.test_mask_dir, valid_list_file=hp.test_valid_list)

    if hp.use_augment:
        train_data = AugmentedDataset(train_raw, copypaste=hp.use_copypaste,
                                      copypaste_p=hp.copypaste_p, scale_aug=hp.use_scale_aug,
                                      scale_range=hp.scale_range)
    else:
        train_data = train_raw

    if hp.use_tile_sampling:
        sampler = build_weighted_sampler(train_raw, exp=hp.tile_sample_exp)
        shuffle = False
    else:
        sampler = None; shuffle = True

    train_iter = DataLoader(train_data, batch_size=hp.batch_size,
                            shuffle=shuffle if sampler is None else None,
                            sampler=sampler, num_workers=2 if use_cuda else 0,
                            pin_memory=use_cuda, drop_last=True)
    val_iter  = DataLoader(val_data,  batch_size=1, shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
    print(f'Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')

    # ---- Loss ----
    loss_fn = BinaryLoss(bce_weight=hp.bce_weight, dice_weight=hp.dice_weight,
                         pos_weight=hp.pos_weight).to(device)
    print(f'Loss: {hp.bce_weight}*BCE(pos_w={hp.pos_weight}) + {hp.dice_weight}*Dice')

    # ---- Optimizer ----
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.num_epochs, eta_min=1e-6)

    # ---- 训练状态 ----
    amp_ctx = torch.amp.autocast('cuda') if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    best_iou = 0.0; best_export_dir = None; no_improve = 0

    history = {'epoch': [], 'train_loss': [], 'train_iou': [], 'train_rec': [], 'train_prec': [],
               'val_loss': [], 'val_iou': [], 'val_rec': [], 'val_prec': []}

    csv_path = os.path.join(hp.result_dir, 'epoch_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_loss','train_iou','val_iou',
                                'train_rec','val_rec','train_prec','val_prec'])

    for epoch in range(1, hp.num_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        hist_2x2 = torch.zeros(2, 2, dtype=torch.float64)
        optimizer.zero_grad()
        pbar = tqdm(train_iter, desc=f'Epoch {epoch}/{hp.num_epochs}', unit='batch')
        for step, (img, label, _) in enumerate(pbar):
            img, label = img.to(device), label.to(device)
            binary_label = label_to_binary(label).float().unsqueeze(1)

            with amp_ctx:
                logits = model(img)
                total_loss, _ = loss_fn(logits, binary_label)
                l = total_loss / hp.accum_steps

            if scaler is not None:
                scaler.scale(l).backward()
                if (step+1)%hp.accum_steps==0 or (step+1)==len(train_iter):
                    if hp.grad_clip_norm>0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                l.backward()
                if (step+1)%hp.accum_steps==0 or (step+1)==len(train_iter):
                    if hp.grad_clip_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
                    optimizer.step(); optimizer.zero_grad()

            train_losses.append(l.item()*hp.accum_steps)
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
                hist_2x2 += metrics.multiclass_confusion(pred, binary_label.squeeze(1).long(), 2).double()

            if step%50==0: pbar.set_postfix(loss=f'{np.mean(train_losses[-50:]):.4f}')

        bm = binary_metrics(hist_2x2)
        avg_loss = float(np.mean(train_losses))
        print(f'\n[Train] loss:{avg_loss:.4f} IoU:{bm["iou"]:.4f} Rec:{bm["rec"]:.4f} Prec:{bm["prec"]:.4f}')
        scheduler.step()

        # --- Val ---
        model.eval()
        val_losses = []; val_hist_2x2 = torch.zeros(2,2,dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, _ in tqdm(val_iter, desc='Val', unit='img'):
                img, label = img.to(device), label.to(device)
                binary_label = label_to_binary(label).float().unsqueeze(1)
                logits = model(img)
                _, ld = loss_fn(logits, binary_label)
                val_losses.append(ld['total'])
                pred = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
                val_hist_2x2 += metrics.multiclass_confusion(pred, binary_label.squeeze(1).long(), 2).double()

        vm = binary_metrics(val_hist_2x2)
        val_avg_loss = float(np.mean(val_losses))
        print(f'[Val]   loss:{val_avg_loss:.4f} IoU:{vm["iou"]:.4f} Rec:{vm["rec"]:.4f} Prec:{vm["prec"]:.4f}')

        # --- Record ---
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss); history['train_iou'].append(bm['iou'])
        history['train_rec'].append(bm['rec']); history['train_prec'].append(bm['prec'])
        history['val_loss'].append(val_avg_loss); history['val_iou'].append(vm['iou'])
        history['val_rec'].append(vm['rec']); history['val_prec'].append(vm['prec'])

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, f'{avg_loss:.6f}', f'{val_avg_loss:.6f}',
                                    f'{bm["iou"]:.6f}', f'{vm["iou"]:.6f}',
                                    f'{bm["rec"]:.6f}', f'{vm["rec"]:.6f}',
                                    f'{bm["prec"]:.6f}', f'{vm["prec"]:.6f}'])

        with open(os.path.join(hp.result_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # --- Best ---
        if vm['iou'] > best_iou:
            best_iou = vm['iou']; no_improve = 0
            torch.save(model.state_dict(), os.path.join(hp.result_dir, f'best_{hp.model_size}.pth'))
            print(f'>>> Best! val_IoU={best_iou:.4f}')
            if hp.save_val_on_best:
                if best_export_dir and os.path.isdir(best_export_dir): shutil.rmtree(best_export_dir, ignore_errors=True)
                best_export_dir = os.path.join(hp.result_dir, f'best_ep{epoch:02d}_iou{best_iou:.4f}')
                export_all_val(model, val_iter, best_export_dir, epoch, device, use_cuda,
                               save_mask=hp.save_pred_mask_png, save_vis=hp.save_pred_vis_png)
        else:
            no_improve += 1
            print(f'  No improvement ({no_improve}/{hp.patience})')

        if epoch%5==0:
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),'best_iou':best_iou},
                       os.path.join(hp.result_dir, f'ckpt_epoch{epoch}.pth'))

        if hp.early_stop and no_improve >= hp.patience:
            print(f'\n*** Early stop at epoch {epoch} ***'); break

    # ---- Final model ----
    torch.save(model.state_dict(), os.path.join(hp.result_dir, f'final_{hp.model_size}.pth'))

    # ---- Test ----
    if len(test_data) > 0:
        print('\n' + '='*50)
        print('>>> Final Test')
        best_path = os.path.join(hp.result_dir, f'best_{hp.model_size}.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        test_losses = []; test_hist_2x2 = torch.zeros(2,2,dtype=torch.float64)
        with torch.no_grad(), amp_ctx:
            for img, label, _ in tqdm(test_iter, desc='Test', unit='img'):
                img, label = img.to(device), label.to(device)
                binary_label = label_to_binary(label).float().unsqueeze(1)
                logits = model(img); _, ld = loss_fn(logits, binary_label)
                test_losses.append(ld['total'])
                pred = (torch.sigmoid(logits) > 0.5).squeeze(1).long()
                test_hist_2x2 += metrics.multiclass_confusion(pred, binary_label.squeeze(1).long(), 2).double()

        tm = binary_metrics(test_hist_2x2)
        print(f'  Test | IoU:{tm["iou"]:.4f} Rec:{tm["rec"]:.4f} Prec:{tm["prec"]:.4f} Acc:{tm["acc"]:.4f}')
        print('='*50)
        if hp.save_all_test_on_best:
            export_all_test(model, test_iter,
                            os.path.join(hp.result_dir, f'final_test_iou_{tm["iou"]:.4f}'),
                            epoch, device, use_cuda)

    plot_training_curves(history, os.path.join(hp.result_dir, 'training_curves.png'))

    if hp.is_kaggle:
        import zipfile
        zp = '/kaggle/working/result.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(hp.result_dir):
                for f in files: zf.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), '/kaggle/working'))
        print(f'Result zip: {zp}')

    print(f'Done! Best IoU = {best_iou:.4f}')
    return model, best_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDNet-Binary 二分类训练')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    env = detect_env()
    hp = HyperParameter(env, config=args.config)
    print(f'Env: {"Kaggle" if hp.is_kaggle else "Local"}')
    print(f'Model: {hp.model_size}, Epochs: {hp.num_epochs}, BS: {hp.batch_size}, LR: {hp.learning_rate}')
    print(f'Result: {hp.result_dir}')
    train(hp)
