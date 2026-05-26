"""
数据集质量分析 + 过滤脚本

功能:
1. 统计每个 tile 的:
   - 黑边/无效像素比例 (WAC 通道 nodata)
   - 各类 GT 像素数 / 前景比例
2. 输出:
   - 训练/测试集质量分布 (直方图 + 文本报告)
   - 建议过滤阈值
   - 写出 valid_tiles_train.txt / valid_tiles_test.txt (符合阈值的文件名清单)
3. 不修改原数据, 后续 MyDataset 可以选择只加载 valid 列表里的文件
"""
import os
import json
import numpy as np
import rasterio
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 配置
# ============================================================================
TRAIN_IMG_DIR  = r'E:\月球_dataset\Research area\train\dataset_v5\image'
TRAIN_MASK_DIR = r'E:\月球_dataset\Research area\train\dataset_v5\mask'
TEST_IMG_DIR   = r'E:\月球_dataset\Research area\test\dataset_v5\image'
TEST_MASK_DIR  = r'E:\月球_dataset\Research area\test\dataset_v5\mask'

OUTPUT_DIR = r'E:\月球_dataset\Research area\dataset_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 过滤阈值 (符合任一条件即剔除)
BAD_RATIO_THRESH = 0.20    # WAC 黑边像素比例 > 20% 剔除
# 注: 不按前景比例过滤 (允许纯背景图作为负例存在)

CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
NUM_CLASSES = 5


# ============================================================================
# 单张 tile 分析
# ============================================================================
def analyze_tile(img_path: str, mask_path: str) -> dict:
    """对一张 tile 计算所有质量指标."""
    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)  # (C, H, W)
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int64)    # (H, W)

    # 修正异常 mask 值
    mask[(mask > 4) | (mask < 0)] = 0

    H, W = mask.shape
    total = H * W

    # 黑边/无效像素 (使用 MyDataset 同款判定)
    bad = ~np.isfinite(image) | (image < -1e10)
    bad_wac = bad[0]                    # WAC 通道的无效像素
    bad_ratio = float(bad_wac.mean())

    # 前景比例
    fg_pixels = int((mask > 0).sum())
    fg_ratio = fg_pixels / total

    # 各类像素数
    class_counts = {c: int((mask == c).sum()) for c in range(NUM_CLASSES)}

    return {
        'bad_ratio': bad_ratio,
        'fg_ratio': fg_ratio,
        'fg_pixels': fg_pixels,
        'total_pixels': total,
        'class_counts': class_counts,
    }


# ============================================================================
# 整个数据集分析
# ============================================================================
def analyze_dataset(img_dir: str, mask_dir: str, split_name: str) -> dict:
    """遍历目录, 返回每个 tile 的统计 + 全局聚合."""
    files = sorted(f for f in os.listdir(mask_dir)
                   if f.lower().endswith(('.tif', '.tiff')))
    print(f'\n=== {split_name}: 共 {len(files)} 张 tile ===')

    per_tile = {}
    aggregate = defaultdict(int)

    for i, fname in enumerate(files):
        img_p = os.path.join(img_dir, fname)
        msk_p = os.path.join(mask_dir, fname)
        info = analyze_tile(img_p, msk_p)
        per_tile[fname] = info

        for c, n in info['class_counts'].items():
            aggregate[f'class_{c}_pixels'] += n
        aggregate['total_pixels'] += info['total_pixels']
        aggregate['bad_pixels'] += int(info['bad_ratio'] * info['total_pixels'])

        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(files)} 已分析')

    # 计算全局类别分布
    total = aggregate['total_pixels']
    print(f'\n  全局像素分布 ({split_name}):')
    for c, name in enumerate(CLASS_NAMES):
        n = aggregate[f'class_{c}_pixels']
        pct = 100 * n / total
        print(f'    {name:15s}: {n:>14,} ({pct:6.3f}%)')
    print(f'    黑边像素率      : {100*aggregate["bad_pixels"]/total:6.3f}%')

    return per_tile, dict(aggregate)


# ============================================================================
# 可视化分布
# ============================================================================
def plot_distributions(per_tile: dict, split_name: str, save_path: str):
    bad_ratios = [v['bad_ratio'] for v in per_tile.values()]
    fg_ratios  = [v['fg_ratio']  for v in per_tile.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 黑边分布
    axes[0].hist(bad_ratios, bins=50, color='steelblue', edgecolor='black')
    axes[0].axvline(BAD_RATIO_THRESH, color='red', linestyle='--',
                     label=f'阈值={BAD_RATIO_THRESH}')
    axes[0].set_xlabel('Bad pixel ratio (WAC)')
    axes[0].set_ylabel('Tile count')
    axes[0].set_title(f'{split_name}: 黑边/无效像素分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 前景比例分布
    axes[1].hist(fg_ratios, bins=50, color='coral', edgecolor='black')
    axes[1].set_xlabel('Foreground ratio (mask > 0)')
    axes[1].set_ylabel('Tile count')
    axes[1].set_title(f'{split_name}: 前景标注比例分布')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f'  分布图已保存: {save_path}')


# ============================================================================
# 过滤
# ============================================================================
def filter_tiles(per_tile: dict, split_name: str) -> tuple:
    """按阈值过滤, 返回 (valid_files, removed_files, stats)."""
    valid, removed = [], []
    removal_reasons = defaultdict(int)

    for fname, info in per_tile.items():
        reasons = []
        if info['bad_ratio'] > BAD_RATIO_THRESH:
            reasons.append(f'bad_ratio>{BAD_RATIO_THRESH}')

        if reasons:
            removed.append((fname, reasons, info))
            for r in reasons:
                removal_reasons[r] += 1
        else:
            valid.append(fname)

    print(f'\n=== {split_name} 过滤结果 ===')
    print(f'  保留: {len(valid)} / {len(per_tile)} ({100*len(valid)/len(per_tile):.1f}%)')
    print(f'  剔除: {len(removed)}')
    for r, c in removal_reasons.items():
        print(f'    - {r}: {c} 张')

    if removed:
        print(f'\n  剔除样本 (前 10):')
        for fname, reasons, info in removed[:10]:
            print(f'    {fname}: bad={info["bad_ratio"]:.2f}, fg={info["fg_ratio"]:.4f}')

    return valid, removed


# ============================================================================
# 主流程
# ============================================================================
def main():
    print(f'输出目录: {OUTPUT_DIR}\n')

    # 训练集
    train_per_tile, train_agg = analyze_dataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, 'TRAIN')
    plot_distributions(train_per_tile, 'TRAIN',
                       os.path.join(OUTPUT_DIR, 'train_dist.png'))
    train_valid, train_removed = filter_tiles(train_per_tile, 'TRAIN')

    # 测试集
    test_per_tile, test_agg = analyze_dataset(
        TEST_IMG_DIR, TEST_MASK_DIR, 'TEST')
    plot_distributions(test_per_tile, 'TEST',
                       os.path.join(OUTPUT_DIR, 'test_dist.png'))
    test_valid, test_removed = filter_tiles(test_per_tile, 'TEST')

    # 写入 valid 列表
    with open(os.path.join(OUTPUT_DIR, 'valid_tiles_train.txt'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(train_valid))
    with open(os.path.join(OUTPUT_DIR, 'valid_tiles_test.txt'),
              'w', encoding='utf-8') as f:
        f.write('\n'.join(test_valid))

    # 写入完整统计 JSON
    summary = {
        'config': {
            'bad_ratio_threshold': BAD_RATIO_THRESH,
        },
        'train': {
            'total_tiles': len(train_per_tile),
            'valid_tiles': len(train_valid),
            'removed_tiles': len(train_removed),
            'aggregate': train_agg,
        },
        'test': {
            'total_tiles': len(test_per_tile),
            'valid_tiles': len(test_valid),
            'removed_tiles': len(test_removed),
            'aggregate': test_agg,
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n=== 完成 ===')
    print(f'  valid_tiles_train.txt: {len(train_valid)} 张')
    print(f'  valid_tiles_test.txt : {len(test_valid)} 张')
    print(f'  summary.json: 完整统计')
    print(f'  train_dist.png / test_dist.png: 分布图')


if __name__ == '__main__':
    main()
