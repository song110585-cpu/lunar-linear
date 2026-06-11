"""
统计训练集 + 验证集各标签的像素总数与类别占比，并绘制可视化表格。
用法: python scripts/class_stats.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm

# ==================== 配置 ====================
MASK_DIR = r"E:\月球_dataset\Research area\train\dataset_v6\mask"
VALID_LIST = r"E:\月球_dataset\Research area\dataset_analysis\valid_tiles_train.txt"
VAL_LIST   = r"E:\月球_dataset\Research area\dataset_analysis\valid_tiles_val.txt"

NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
CLASS_NAMES_CN = ['背景', '皱脊', '月溪', '断层', '地堑']
CLASS_COLORS = ['#333333', '#FF0000', '#0064FF', '#00C800', '#FFFF00']


def count_pixels(mask_dir, valid_list=None):
    """扫描 mask 目录, 统计每类像素总数和总像素数"""
    all_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff')))

    if valid_list and os.path.exists(valid_list):
        with open(valid_list, 'r', encoding='utf-8') as f:
            valid_set = set(line.strip() for line in f if line.strip())
        files = [f for f in all_files if f in valid_set]
        print(f'过滤: {len(all_files)} -> {len(files)} ({os.path.basename(valid_list)})')
    else:
        files = all_files

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_valid = 0

    for fname in tqdm(files, desc=f'统计 {len(files)} 张mask'):
        with rasterio.open(os.path.join(mask_dir, fname)) as src:
            mask = src.read(1)

        # 过滤异常值
        mask[(mask > 4) | (mask < 0)] = 0

        for c in range(NUM_CLASSES):
            counts[c] += (mask == c).sum()
        total_valid += mask.size  # H * W

    return counts, total_valid


def main():
    # ---- 训练集 ----
    print('=== 训练集 ===')
    train_counts, train_total = count_pixels(MASK_DIR, VALID_LIST)

    # ---- 验证集 ----
    print('\n=== 验证集 ===')
    val_counts, val_total = count_pixels(MASK_DIR, VAL_LIST)

    # ---- 全部 ----
    all_counts = train_counts + val_counts
    all_total = train_total + val_total

    # ---- 打印结果 ----
    print(f'\n{"="*60}')
    print(f'{"Class":<20} {"Train Pixels":>15} {"Val Pixels":>15} {"Total Pixels":>15} {"Ratio":>8}')
    print(f'{"-"*60}')
    for c in range(NUM_CLASSES):
        ratio = all_counts[c] / all_total * 100
        print(f'{CLASS_NAMES[c]:<20} {train_counts[c]:>15,} {val_counts[c]:>15,} {all_counts[c]:>15,} {ratio:>7.2f}%')
    print(f'{"-"*60}')
    print(f'{"Total":<20} {train_total:>15,} {val_total:>15,} {all_total:>15,}')
    print(f'{"="*60}')

    # ---- 绘制可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 饼图
    axes[0].pie(all_counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
                autopct='%1.2f%%', startangle=140,
                explode=(0.02, 0.02, 0.02, 0.05, 0.05))
    axes[0].set_title('Class Distribution (Train + Val)', fontsize=14)

    # 柱状图 (不含背景)
    idx = [1, 2, 3, 4]  # Wrinkle, Rille, Fault, Graben
    fg_counts = [all_counts[i] for i in idx]
    fg_labels = [CLASS_NAMES[i] for i in idx]
    fg_colors = [CLASS_COLORS[i] for i in idx]
    fg_ratios = [all_counts[i] / all_total * 100 for i in idx]

    bars = axes[1].bar(fg_labels, fg_ratios, color=fg_colors, edgecolor='black')
    axes[1].set_ylabel('Ratio (%)')
    axes[1].set_title('Foreground Class Ratio (excl. Background)', fontsize=14)
    for bar, val, cnt in zip(bars, fg_ratios, fg_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}%\n({cnt:,} px)', ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_distribution.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\n图表已保存: {save_path}')

    # 打印逐类汇总表
    print(f'\n{"="*60}')
    print(f'类别占比总结')
    print(f'{"="*60}')
    for c in range(NUM_CLASSES):
        ratio = all_counts[c] / all_total * 100
        bar = '█' * int(ratio * 2)
        print(f'{CLASS_NAMES_CN[c]:<6} {ratio:>5.1f}%  {bar}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
