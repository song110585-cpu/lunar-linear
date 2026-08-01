r"""
LDNet-Binary 推理评估 — Step 1: 线 vs 背景

用法:
  python scripts/test_binary.py "E:\月球_dataset\双分支output\R2" --split test
  python scripts/test_binary.py "E:\月球_dataset\双分支output\R2" --split test --tta
"""
import os, sys as _sys, argparse, glob
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
from MyDataset import MyDataset
from ldnet_binary import LDNetBinary
from binary_loss import label_to_binary


# ============================================================================
# 参数推断
# ============================================================================

def infer_model_size(ckpt_path):
    fname = os.path.basename(ckpt_path).lower()
    for s in ['tiny', 'small', 'base']:
        if s in fname: return s
    return 'small'


def infer_img_size(state_dict):
    for k, v in state_dict.items():
        if 'attn_mask' in k and v.dim() >= 1:
            import math
            side_w = int(math.sqrt(v.shape[0]))
            return side_w * 16 * 4
    return 512


def filter_state(state, prefix_blacklist=('sp2.', 'sp3.')):
    return {k: v for k, v in state.items() if not k.startswith(prefix_blacklist)}


# ============================================================================
# TTA
# ============================================================================

def tta_inference(model, img, device, use_amp=True):
    amp_ctx = torch.amp.autocast('cuda') if use_amp and torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), amp_ctx:
        prob = torch.sigmoid(model(img))
        prob += torch.flip(torch.sigmoid(model(torch.flip(img, [-1]))), [-1])
        prob += torch.flip(torch.sigmoid(model(torch.flip(img, [-2]))), [-2])
        prob += torch.rot90(torch.sigmoid(model(torch.rot90(img, 1, [-2,-1]))), -1, [-2,-1])
    return (prob / 4.0 > 0.5).squeeze(1).long()


# ============================================================================
# 主逻辑
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LDNet-Binary 二分类评估')
    parser.add_argument('result_dir', type=str, nargs='?', default='.',
                        help='结果目录 (自动找 best_*.pth)')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--model_size', type=str, default=None)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--export-dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f'Device: {device}')

    # ---- 权重 ----
    if args.weights:
        ckpt_path = args.weights
    else:
        cand = glob.glob(os.path.join(args.result_dir, 'best_*.pth'))
        if not cand:
            cand = glob.glob(os.path.join(args.result_dir, '**', 'best_*.pth'), recursive=True)
        if not cand:
            raise FileNotFoundError(f'未找到 best_*.pth: {args.result_dir}')
        ckpt_path = cand[0]
    print(f'Weights: {ckpt_path}')

    state = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in state: state = state['model_state_dict']
    state = filter_state(state)
    model_size = args.model_size or infer_model_size(ckpt_path)
    img_size = infer_img_size(state)
    print(f'model_size={model_size}, img_size={img_size}')

    # ---- 模型 ----
    model = LDNetBinary(size=model_size, img_size=img_size, in_channels=5, pretrained=False).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:  print(f'[WARN] Missing: {missing[:3]}...')
    if unexpected: print(f'[WARN] Unexpected: {unexpected[:3]}...')
    model.eval()

    # ---- 数据 ----
    if args.data_dir:
        img_dir = os.path.join(args.data_dir, 'image')
        mask_dir = os.path.join(args.data_dir, 'mask')
    else:
        _base = r'E:\月球_dataset\dataset\datasetv5'
        if not os.path.isdir(_base):
            _base = '/kaggle/input/datasets/yuanssy/dataset5/datasetv5'
        img_dir = os.path.join(_base, args.split, 'image')
        mask_dir = os.path.join(_base, args.split, 'mask')
    print(f'Data: {img_dir}')

    eval_data = MyDataset(img_dir, mask_dir)
    eval_iter = DataLoader(eval_data, batch_size=1, shuffle=False,
                           num_workers=2 if use_amp else 0, pin_memory=use_amp)
    print(f'Tiles: {len(eval_data)}')

    # ---- 推理 ----
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
    eval_hist = torch.zeros(2, 2, dtype=torch.float64)

    with torch.no_grad():
        for img, label, _ in tqdm(eval_iter, desc='Inference', unit='img'):
            img, label = img.to(device), label.to(device)
            binary_label = label_to_binary(label)

            if args.tta:
                pred = tta_inference(model, img, device, use_amp)
            else:
                with amp_ctx:
                    pred = model.predict(img)

            eval_hist += metrics.multiclass_confusion(pred, binary_label, 2).double()

    # ---- 结果 ----
    tn, fp = eval_hist[0,0].item(), eval_hist[0,1].item()
    fn, tp = eval_hist[1,0].item(), eval_hist[1,1].item()
    eps = 1e-10
    iou  = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2*prec*rec/(prec+rec+eps)
    acc  = (tp+tn)/(tp+tn+fp+fn+eps)

    tta_str = ' (TTA)' if args.tta else ''
    print(f'\n{"="*45}')
    print(f'  BINARY {args.split.upper()} Results{tta_str}')
    print(f'  IoU:       {iou:.4f}')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1:        {f1:.4f}')
    print(f'  TP={tp:.0f}  FP={fp:.0f}  FN={fn:.0f}  TN={tn:.0f}')
    print(f'{"="*45}')

    # ---- 导出 ----
    if args.export:
        export_dir = args.export_dir or os.path.join(
            os.path.dirname(ckpt_path), f'{args.split}_preds')
        os.makedirs(export_dir, exist_ok=True)
        for img, label, _ in tqdm(eval_iter, desc='Export', unit='img'):
            img = img.to(device)
            if args.tta:
                pred = tta_inference(model, img, device, use_amp)
            else:
                with torch.no_grad(), amp_ctx: pred = model.predict(img)
            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(pred_np*255, mode='L').save(
                os.path.join(export_dir, f'{os.path.splitext(_[0])[0]}.png'))
        print(f'Exported {len(os.listdir(export_dir))} to {export_dir}')


if __name__ == '__main__':
    main()
