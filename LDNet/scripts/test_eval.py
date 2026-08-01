"""
LDNet 双分支推理评估脚本

与 STDL-Net test_eval.py 接口兼容, 自动推断 img_size/model_size,
过滤无效 checkpoint key, 处理 tuple 输出, 支持 TTA, 导出预测图.

用法:
  python scripts/test_eval.py "E:\月球_dataset\双分支output\R1" --split test
  python scripts/test_eval.py "E:\月球_dataset\双分支output\R1" --split test --tta
  python scripts/test_eval.py . --weights best_small.pth --data-dir "E:\月球_dataset\dataset\datasetv5\test"
"""
import os
import sys as _sys
import argparse
from contextlib import nullcontext

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models', 'losses']:
    _p = os.path.join(_root, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
from ldnet import LDNet, merge_branches

CLASS_NAMES = ['背景', '皱脊', '月溪', '断层', '地堑']


# ============================================================================
# 工具
# ============================================================================

def infer_model_size(ckpt_path):
    """从权重文件名推断 model_size: best_small.pth → small"""
    fname = os.path.basename(ckpt_path).lower()
    for size in ['tiny', 'small', 'base']:
        if size in fname:
            return size
    return 'small'


def infer_img_size(state_dict):
    """从 checkpoint 的 attn_mask 形状推断 img_size.

    attn_mask 形状: (num_windows, Wh*Ww, Wh*Ww)
    Wh = Ww = window_size, num_windows = (H / window_size) * (W / window_size)
    对于 512×512, window_size=16: 32×32=1024 windows
    """
    for k, v in state_dict.items():
        if 'attn_mask' in k and v.dim() >= 1:
            num_windows = v.shape[0]
            # num_windows = (img_size / 4 / window_size) ** 2
            # img_size = sqrt(num_windows) * window_size * 4
            import math
            side_windows = int(math.sqrt(num_windows))
            return side_windows * 16 * 4  # window_size=16, patch_size=4
    return 512  # fallback


def filter_state(state_dict, prefix_blacklist=('sp2.', 'sp3.')):
    """过滤 checkpoint 中无效的 key (如旧模块残留)."""
    return {k: v for k, v in state_dict.items()
            if not k.startswith(prefix_blacklist)}


# ============================================================================
# TTA
# ============================================================================

def tta_inference(model, img, device, use_amp=True):
    """4 方向 TTA: 原始 + H-flip + V-flip + Rot90."""
    amp_ctx = torch.amp.autocast('cuda') if use_amp and torch.cuda.is_available() else nullcontext()

    with torch.no_grad(), amp_ctx:
        b1, b2 = model(img)
        b1_prob = torch.sigmoid(b1)
        b2_prob = F.softmax(b2, dim=1)

        b1f, b2f = model(torch.flip(img, [-1]))
        b1_prob += torch.flip(torch.sigmoid(b1f), [-1])
        b2_prob += torch.flip(F.softmax(b2f, dim=1), [-1])

        b1f, b2f = model(torch.flip(img, [-2]))
        b1_prob += torch.flip(torch.sigmoid(b1f), [-2])
        b2_prob += torch.flip(F.softmax(b2f, dim=1), [-2])

        b1f, b2f = model(torch.rot90(img, 1, [-2, -1]))
        b1_prob += torch.rot90(torch.sigmoid(b1f), -1, [-2, -1])
        b2_prob += torch.rot90(F.softmax(b2f, dim=1), -1, [-2, -1])

    b1_prob /= 4.0
    b2_prob /= 4.0
    binary_mask = (b1_prob > 0.5).squeeze(1)
    class_pred = b2_prob.argmax(dim=1) + 1
    return (class_pred * binary_mask).long()


# ============================================================================
# 主逻辑
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LDNet 双分支推理评估')
    parser.add_argument('result_dir', type=str, nargs='?', default='.',
                        help='结果目录 (自动找 best_*.pth, 支持 result/R1/)')
    parser.add_argument('--weights', type=str, default=None,
                        help='直接指定权重路径 (覆盖 result_dir)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val'],
                        help='评估哪个 split')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='test/val 数据根目录 (如 E:\\...\\datasetv5\\test)')
    parser.add_argument('--model_size', type=str, default=None,
                        help='模型规模 (不指定则从文件名推断)')
    parser.add_argument('--tta', action='store_true',
                        help='Test-Time Augmentation')
    parser.add_argument('--export', action='store_true',
                        help='导出预测 PNG')
    parser.add_argument('--export-dir', type=str, default=None,
                        help='导出目录')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f'Device: {device}')

    # ---- 1. 找权重 ----
    if args.weights:
        ckpt_path = args.weights
    else:
        import glob
        candidates = glob.glob(os.path.join(args.result_dir, 'best_*.pth'))
        if not candidates:
            # 尝试子目录
            candidates = glob.glob(
                os.path.join(args.result_dir, '**', 'best_*.pth'), recursive=True)
        if not candidates:
            raise FileNotFoundError(
                f'在 {args.result_dir} 中未找到 best_*.pth 权重文件')
        ckpt_path = candidates[0]
    print(f'Weights: {ckpt_path}')

    state = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    state = filter_state(state)

    # ---- 2. 推断参数 ----
    model_size = args.model_size or infer_model_size(ckpt_path)
    img_size = infer_img_size(state)
    print(f'Inferred: model_size={model_size}, img_size={img_size}')

    # ---- 3. 模型 ----
    model = LDNet(
        size=model_size, img_size=img_size,
        in_channels=5, pretrained=False,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[WARN] Missing keys ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')
    model.eval()

    # ---- 4. 数据 ----
    if args.data_dir:
        img_dir = os.path.join(args.data_dir, 'image')
        mask_dir = os.path.join(args.data_dir, 'mask')
    else:
        _base = r'E:\月球_dataset\dataset\datasetv5'
        if not os.path.isdir(_base):
            # Kaggle fallback
            _base = '/kaggle/input/datasets/yuanssy/dataset5/datasetv5'
        img_dir = os.path.join(_base, args.split, 'image')
        mask_dir = os.path.join(_base, args.split, 'mask')
    print(f'Data: {img_dir}')

    eval_data = MyDataset(img_dir, mask_dir)
    eval_iter = DataLoader(eval_data, batch_size=1, shuffle=False,
                           num_workers=2 if use_amp else 0,
                           pin_memory=use_amp)
    print(f'Tiles: {len(eval_data)}')

    # ---- 5. 推理 + 指标 ----
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
    eval_hist = torch.zeros(5, 5, dtype=torch.float64)

    with torch.no_grad():
        for img, label, name in tqdm(eval_iter, desc='Inference', unit='img'):
            img, label = img.to(device), label.to(device)

            if args.tta:
                pred = tta_inference(model, img, device, use_amp)
            else:
                with amp_ctx:
                    pred = model.predict(img)

            eval_hist += metrics.multiclass_confusion(pred, label, 5).double()

    m = metrics.metrics_from_hist(eval_hist)

    # ---- 6. 输出 ----
    tta_str = ' (TTA)' if args.tta else ''
    print(f'\n{"=" * 55}')
    print(f'  {args.split.upper()} Results{tta_str}')
    print(f'  mIoU:      {m["miou"]:.4f}')
    print(f'  Accuracy:  {m["accuracy"]:.4f}')
    print(f'  mPrecision:{m["mprecision"]:.4f}')
    print(f'  mRecall:   {m["mrecall"]:.4f}')
    print(f'  mF1:       {m["mf1"]:.4f}')
    print(f'  {"-" * 45}')
    for c in range(5):
        print(f'  {CLASS_NAMES[c]:<12}  '
              f'IoU={m["iou_per_class"][c]:.4f}  '
              f'P={m["precision_per_class"][c]:.4f}  '
              f'R={m["recall_per_class"][c]:.4f}  '
              f'F1={m["f1_per_class"][c]:.4f}')
    print(f'{"=" * 55}')

    # ---- 7. 导出 ----
    if args.export:
        export_dir = args.export_dir or os.path.join(
            os.path.dirname(ckpt_path), f'{args.split}_preds')
        os.makedirs(export_dir, exist_ok=True)

        for img, label, name in tqdm(eval_iter, desc='Export', unit='img'):
            img, label = img.to(device), label.to(device)

            if args.tta:
                pred = tta_inference(model, img, device, use_amp)
            else:
                with torch.no_grad(), amp_ctx:
                    pred = model.predict(img)

            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            stem = name[0]
            Image.fromarray(pred_np, mode='L').save(
                os.path.join(export_dir, f'{stem}.png'))

        print(f'Exported {len(os.listdir(export_dir))} to {export_dir}')


if __name__ == '__main__':
    main()
