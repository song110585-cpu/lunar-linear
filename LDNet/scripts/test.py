r"""
LDNet 双分支推理脚本

用法:
  python scripts/test.py --weights result/best_small.pth --data v5
  python scripts/test.py --weights result/best_small.pth --data v5 --tta
  python scripts/test.py --weights result/best_small.pth --data_dir E:\月球_dataset\dataset\datasetv5\test
"""
import os
import sys as _sys
import argparse

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models', 'losses']:
    _p = os.path.join(_root, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics
from MyDataset import MyDataset, CHANNEL_MEAN, CHANNEL_STD
from ldnet import LDNet, merge_branches

CLASS_NAMES = ['背景', '皱脊', '月溪', '断层', '地堑']


def tta_inference(model, img, device, use_amp=True):
    """Test-Time Augmentation: 原始 + 水平翻转 + 垂直翻转 + 旋转90°, 取平均."""
    amp_ctx = torch.amp.autocast('cuda') if use_amp and torch.cuda.is_available() else nullcontext()

    with torch.no_grad(), amp_ctx:
        # original
        b1, b2 = model(img)
        b1_prob = torch.sigmoid(b1)
        b2_prob = F.softmax(b2, dim=1)

        # H-flip
        b1f, b2f = model(torch.flip(img, [-1]))
        b1_prob += torch.flip(torch.sigmoid(b1f), [-1])
        b2_prob += torch.flip(F.softmax(b2f, dim=1), [-1])

        # V-flip
        b1f, b2f = model(torch.flip(img, [-2]))
        b1_prob += torch.flip(torch.sigmoid(b1f), [-2])
        b2_prob += torch.flip(F.softmax(b2f, dim=1), [-2])

        # Rot90
        b1f, b2f = model(torch.rot90(img, 1, [-2, -1]))
        b1_prob += torch.rot90(torch.sigmoid(b1f), -1, [-2, -1])
        b2_prob += torch.rot90(F.softmax(b2f, dim=1), -1, [-2, -1])

    # Average
    b1_prob /= 4.0
    b2_prob /= 4.0

    # Merge
    binary_mask = (b1_prob > 0.5).squeeze(1)
    class_pred = b2_prob.argmax(dim=1) + 1
    return (class_pred * binary_mask).long()


def main():
    parser = argparse.ArgumentParser(description='LDNet 双分支推理')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径 (.pth)')
    parser.add_argument('--data', type=str, default='v5',
                        choices=['v5', 'v8', 'v9', 'v10'],
                        help='数据集版本')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='自定义 test 数据目录 (覆盖 --data)')
    parser.add_argument('--model_size', type=str, default='small',
                        help='模型规模 (small/base)')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--in_channels', type=int, default=5)
    parser.add_argument('--tta', action='store_true',
                        help='使用 Test-Time Augmentation')
    parser.add_argument('--export', action='store_true',
                        help='导出预测 PNG')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='导出目录 (默认: 权重同目录)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f'Device: {device}')

    # ---- 数据路径 ----
    if args.data_dir:
        test_image_dir = os.path.join(args.data_dir, 'image')
        test_mask_dir  = os.path.join(args.data_dir, 'mask')
    else:
        _base = r'E:\月球_dataset\dataset'
        _d = os.path.join(_base, f'dataset{args.data}')
        test_image_dir = os.path.join(_d, 'test', 'image')
        test_mask_dir  = os.path.join(_d, 'test', 'mask')

    print(f'Test data: {test_image_dir}')

    # ---- 模型 ----
    model = LDNet(
        size=args.model_size,
        img_size=args.img_size,
        in_channels=args.in_channels,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'[WARN] Missing keys ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')
    model.eval()
    print(f'Loaded: {args.weights}')

    # ---- 数据集 ----
    test_data = MyDataset(test_image_dir, test_mask_dir)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False,
                           num_workers=0, pin_memory=use_amp)
    print(f'Test tiles: {len(test_data)}')

    # ---- 推理 ----
    hist = torch.zeros(5, 5, dtype=torch.float64)
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()

    for img, label, name in tqdm(test_iter, desc='Inference', unit='img'):
        img, label = img.to(device), label.to(device)

        if args.tta:
            pred = tta_inference(model, img, device, use_amp)
        else:
            with torch.no_grad(), amp_ctx:
                pred = model.predict(img)

        hist += metrics.multiclass_confusion(pred, label, 5).double()

    # ---- 结果 ----
    m = metrics.metrics_from_hist(hist)
    tta_str = ' (TTA)' if args.tta else ''
    print(f'\n{"=" * 55}')
    print(f'  Test Results{tta_str}')
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

    # ---- 导出 ----
    if args.export:
        from PIL import Image

        export_dir = args.export_dir or os.path.join(
            os.path.dirname(args.weights), 'test_preds')
        os.makedirs(export_dir, exist_ok=True)

        for img, label, name in tqdm(test_iter, desc='Export', unit='img'):
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

        print(f'Exported {len(os.listdir(export_dir))} predictions to {export_dir}')


if __name__ == '__main__':
    main()
