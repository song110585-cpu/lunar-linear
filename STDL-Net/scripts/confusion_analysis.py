"""
从 pred_mask PNG 和 GT 计算混淆矩阵，聚焦 Fault (断层) 的误分来源。
"""
import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import rasterio

PRED_DIR = r"E:\月球_dataset\baseline模型结果\result29\result\result\best_epoch_58_val_miou_0.6002\pred_mask"
GT_DIR   = r"E:\月球_dataset\Research area\train\dataset_v6\mask"
VAL_LIST = r"E:\月球_dataset\Research area\dataset_analysis\valid_tiles_val.txt"

NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Wrinkle', 'Rille', 'Fault', 'Graben']
CLASS_NAMES_CN = ['背景', '皱脊', '月溪', '断层', '地堑']


def main():
    # 读取 val tile 列表 (去掉 .tif/.png 后缀)
    with open(VAL_LIST, 'r') as f:
        val_tiles = set(os.path.splitext(line.strip())[0] for line in f if line.strip())

    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.png') and os.path.splitext(f)[0] in val_tiles])
    print(f'Val tiles in pred_mask: {len(pred_files)}')

    # 混淆矩阵: hist[gt, pred]
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fname in tqdm(pred_files, desc='计算混淆矩阵'):
        stem = os.path.splitext(fname)[0]
        pred = np.array(Image.open(os.path.join(PRED_DIR, fname)))

        gt_path = os.path.join(GT_DIR, f'{stem}.tif')
        with rasterio.open(gt_path) as src:
            gt = src.read(1).astype(np.int64)
        gt[(gt > 4) | (gt < 0)] = 0

        for g in range(NUM_CLASSES):
            for p in range(NUM_CLASSES):
                hist[g, p] += ((gt == g) & (pred == p)).sum()

    print(f'\n{"="*70}')
    print('混淆矩阵 (行=GT, 列=Pred):')
    print(f'{"":>12}', end='')
    for c in range(NUM_CLASSES):
        print(f'{CLASS_NAMES[c]:>10}', end='')
    print()
    for g in range(NUM_CLASSES):
        print(f'{CLASS_NAMES[g]:>12}', end='')
        for p in range(NUM_CLASSES):
            print(f'{hist[g, p]:>10,}', end='')
        print()

    # 行归一化
    print(f'\n{"="*70}')
    print('归一化混淆矩阵 (行=100%, GT → Pred %):')
    for g in range(NUM_CLASSES):
        total = hist[g].sum()
        print(f'{CLASS_NAMES[g]:>12}', end='')
        for p in range(NUM_CLASSES):
            ratio = hist[g, p] / total * 100 if total > 0 else 0
            print(f'{ratio:>9.1f}%', end='')
        print()

    # ---- Fault 专项分析 ----
    fault_idx = 3
    gt_fault_total = hist[fault_idx].sum()
    print(f'\n{"="*70}')
    print(f'🔍 Fault (断层) 专项分析:')
    print(f'  总 GT 像素: {gt_fault_total:,}')
    print(f'  正确预测:   {hist[3, 3]:,} ({hist[3,3]/gt_fault_total*100:.1f}%)')
    print(f'  漏检 (FN):   {gt_fault_total - hist[3, 3]:,}')

    print(f'\n  Fault 像素被误分为:')
    for p in range(NUM_CLASSES):
        if p != fault_idx and hist[fault_idx, p] > 0:
            ratio = hist[fault_idx, p] / gt_fault_total * 100
            print(f'    → {CLASS_NAMES_CN[p]:<6}: {hist[fault_idx, p]:>10,} px ({ratio:>5.1f}%)')

    print(f'\n  Fault 预测的来源 (哪些 GT 被标为 Fault):')
    pred_fault_total = hist[:, fault_idx].sum()
    for g in range(NUM_CLASSES):
        if g != fault_idx and hist[g, fault_idx] > 0:
            ratio = hist[g, fault_idx] / pred_fault_total * 100
            print(f'    ← {CLASS_NAMES_CN[g]:<6}: {hist[g, fault_idx]:>10,} px ({ratio:>5.1f}%)')

    # 各类 Recall / Precision
    print(f'\n{"="*70}')
    print(f'{"Class":<15} {"Recall":>8} {"Precision":>10} {"IoU":>8}')
    print('-'*45)
    for c in range(NUM_CLASSES):
        recall = hist[c, c] / hist[c].sum() * 100 if hist[c].sum() > 0 else 0
        prec   = hist[c, c] / hist[:, c].sum() * 100 if hist[:, c].sum() > 0 else 0
        iou    = hist[c, c] / (hist[c].sum() + hist[:, c].sum() - hist[c, c]) * 100
        print(f'{CLASS_NAMES[c]:<15} {recall:>7.1f}% {prec:>9.1f}% {iou:>7.1f}%')


if __name__ == '__main__':
    main()
