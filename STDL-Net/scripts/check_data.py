"""
数据集质量检查: 逐 tile 检查 image/mask 配对、像素值范围、前景占比、异常通道.

用法: python STDL-Net/scripts/check_data.py --data E:\月球_dataset\dataset\datasetv5

输出:
  - 每 tile 一行: filename | image_shape | mask_classes | fg_pixel% | issues
  - 末尾汇总: 各类异常统计
"""
import os, sys, argparse
import numpy as np
import rasterio
from collections import Counter
from tqdm import tqdm


def check_tile(img_path, mask_path):
    """检查单个 tile, 返回 issues 列表."""
    issues = []

    # 1. 读 image
    try:
        with rasterio.open(img_path) as src:
            img = src.read()
            c, h, w = img.shape
    except Exception as e:
        return [f'IMAGE_READ_ERROR: {e}'], None, None, None, None

    # 2. 读 mask
    try:
        with rasterio.open(mask_path) as src:
            msk = src.read(1).astype(np.int64)
            mh, mw = msk.shape
    except Exception as e:
        return [f'MASK_READ_ERROR: {e}'], (c, h, w), None, None, None

    # 3. 尺寸一致性
    if h != mh or w != mw:
        issues.append(f'SHAPE_MISMATCH: img=({h},{w}) mask=({mh},{mw})')

    # 4. image 通道异常
    img_nan = np.sum(np.isnan(img))
    img_inf = np.sum(np.isinf(img))
    if img_nan > 0:
        issues.append(f'IMAGE_NAN: {img_nan} pixels')
    if img_inf > 0:
        issues.append(f'IMAGE_INF: {img_inf} pixels')
    for ch in range(c):
        ch_range = (img[ch].min(), img[ch].max())
        if ch_range[1] - ch_range[0] < 1e-6:
            issues.append(f'IMAGE_CH{ch}_CONSTANT: {ch_range}')

    # 5. mask 值域
    unique_vals = np.unique(msk)
    bad_vals = [v for v in unique_vals if v not in [0, 1, 2, 3, 4]]
    if bad_vals:
        issues.append(f'MASK_BAD_VALUES: {bad_vals}')

    # 6. 前景像素占比
    total = msk.size
    fg_pixels = int(np.sum(msk > 0))
    fg_pct = fg_pixels / total * 100 if total > 0 else 0

    # 7. 各类别像素统计
    class_pixels = {int(v): int(np.sum(msk == v)) for v in unique_vals}

    return issues, (c, h, w), unique_vals, fg_pct, class_pixels


def main():
    parser = argparse.ArgumentParser(description='数据集质量检查')
    parser.add_argument('--data', required=True, help='数据集根目录 (含 train/val/test 子目录)')
    parser.add_argument('--split', default='train', choices=['train','val','test','all'],
                        help='检查哪个 split (default: train)')
    parser.add_argument('--out', default=None, help='输出 CSV 路径 (default: data_dir/quality_report.csv)')
    args = parser.parse_args()

    data_dir = args.data
    out_csv = args.out or os.path.join(data_dir, f'quality_report_{args.split}.csv')

    # 收集所有 image/mask 目录对: 兼容两种结构
    #   A) data_dir/train/image + data_dir/train/mask
    #   B) data_dir/image + data_dir/mask  (没有 split 子目录)
    pairs = []  # [(label, img_dir, msk_dir)]

    if args.split == 'all':
        targets = ['train', 'val', 'test']
    else:
        targets = [args.split]

    for s in targets:
        img_d = os.path.join(data_dir, s, 'image')
        msk_d = os.path.join(data_dir, s, 'mask')
        if os.path.isdir(img_d) and os.path.isdir(msk_d):
            pairs.append((s, img_d, msk_d))

    # 结构 B: data_dir 直接含 image/ 和 mask/
    img_d = os.path.join(data_dir, 'image')
    msk_d = os.path.join(data_dir, 'mask')
    if os.path.isdir(img_d) and os.path.isdir(msk_d):
        pairs.append(('dataset', img_d, msk_d))

    if not pairs:
        print(f'[错误] 在 {data_dir} 下未找到 image/mask 目录')
        print(f'  支持的结构:')
        print(f'    A) {data_dir}/train/image + {data_dir}/train/mask')
        print(f'    B) {data_dir}/image + {data_dir}/mask')
        return

    all_rows = []

    for label, img_dir, msk_dir in pairs:

        tiles = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.tif','.tiff')))
        print(f'[{label}] {len(tiles)} tiles  ({img_dir})')

        split_issues = Counter()

        for tile in tqdm(tiles, desc=f'检查 {label}'):
            img_path = os.path.join(img_dir, tile)
            msk_path = os.path.join(msk_dir, tile)

            issues, shape, vals, fg_pct, cls_px = check_tile(img_path, msk_path)

            row = {
                'split': label,
                'filename': tile,
                'shape': f'{shape[1]}x{shape[2]}x{shape[0]}' if shape else 'ERROR',
                'fg_pct': f'{fg_pct:.2f}' if fg_pct is not None else '',
                'mask_classes': ','.join(str(v) for v in vals) if vals is not None else '',
                'fg_pixels': cls_px.get(1, 0) + cls_px.get(2, 0) + cls_px.get(3, 0) + cls_px.get(4, 0) if cls_px else 0,
                'issues': '; '.join(issues) if issues else 'OK',
            }
            all_rows.append(row)

            if issues:
                for iss in issues:
                    tag = iss.split(':')[0] if ':' in iss else iss
                    split_issues[tag] += 1
            else:
                split_issues['OK'] += 1

        # 汇总
        print(f'\n[{label}] 汇总:')
        for tag, cnt in sorted(split_issues.items(), key=lambda x: -x[1]):
            pct = cnt / len(tiles) * 100
            print(f'  {tag}: {cnt} ({pct:.1f}%)')
        print()

    # 写 CSV
    import csv
    fieldnames = ['split', 'filename', 'shape', 'fg_pct', 'mask_classes', 'fg_pixels', 'issues']
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'报告: {out_csv} ({len(all_rows)} tiles)')


if __name__ == '__main__':
    main()
