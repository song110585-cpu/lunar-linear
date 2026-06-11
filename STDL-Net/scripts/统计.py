"""
数据集统计脚本 — 完整版
输出到 dataset_analysis/ 文件夹:
  - Pixel_Statistics.csv, Class_Area_Statistics.csv (论文用)
  - 各类直方图 / Log直方图 / Boxplot / CDF (600 dpi)
  - report.txt 完整报告
"""
import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===========================
# 配置
# ===========================
MASK_DIR = r"E:\月球_dataset\Research area\train\dataset_v6\mask"
OUTPUT_DIR = r"E:\月球_dataset\Research area\dataset_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = {0: "Background", 1: "Wrinkle_Ridge", 2: "Rille", 3: "Fault", 4: "Graben"}
CLASS_CN = {0: "背景", 1: "皱脊", 2: "月溪", 3: "断层", 4: "地堑"}
FOREGROUND_CLASSES = [1, 2, 3, 4]
COLORS = ['#333333', '#FF0000', '#0064FF', '#00C800', '#FFFF00']

# ===========================
# 初始化
# ===========================
pixel_statistics = {k: 0 for k in CLASSES}
foreground_ratio = []
class_area_all = {c: [] for c in FOREGROUND_CLASSES}  # 所有 tile (含 0)
class_exist_count = {c: 0 for c in FOREGROUND_CLASSES}
tile_class_number = []
total_tiles = 0
pure_bg_tiles = 0

# ===========================
# 遍历 mask
# ===========================
files = [f for f in os.listdir(MASK_DIR) if f.endswith('.tif')]
print(f'扫描 {len(files)} 张 mask...')

for file in tqdm(files):
    total_tiles += 1
    path = os.path.join(MASK_DIR, file)
    with rasterio.open(path) as src:
        mask = src.read(1)

    total_px = mask.size
    fg_px = int(np.sum(mask != 0))
    fg_ratio_val = fg_px / total_px
    foreground_ratio.append(fg_ratio_val)

    if fg_px == 0:
        pure_bg_tiles += 1

    exist_num = 0
    for cls in CLASSES:
        px = int(np.sum(mask == cls))
        pixel_statistics[cls] += px
        if cls != 0:
            class_area_all[cls].append(px)
            if px > 0:
                class_exist_count[cls] += 1
                exist_num += 1

    tile_class_number.append(exist_num)

foreground_ratio = np.array(foreground_ratio, dtype=np.float64)
tile_class_number = np.array(tile_class_number)

# ===========================
# Helper: 所有 tile 和 仅存在 tile 的分位数统计
# ===========================
def compute_stats(arr_all, arr_exist):
    """返回两个 dict: all_stats, exist_stats"""
    def _s(a):
        if len(a) == 0:
            return {'Mean': 0, 'Std': 0, 'Min': 0, 'Max': 0,
                    'P5': 0, 'P25': 0, 'P50': 0, 'P75': 0, 'P95': 0}
        return {
            'Mean': float(np.mean(a)),
            'Std': float(np.std(a)),
            'Min': float(np.min(a)),
            'Max': float(np.max(a)),
            'P5': float(np.percentile(a, 5)),
            'P25': float(np.percentile(a, 25)),
            'P50': float(np.percentile(a, 50)),
            'P75': float(np.percentile(a, 75)),
            'P95': float(np.percentile(a, 95)),
        }
    return _s(arr_all), _s(arr_exist)

# ===========================
# 报告写入
# ===========================
def write_report(fp):
    def p(*args, **kwargs):
        print(*args, **kwargs, file=fp)

    p("=" * 70)
    p("STDL-Net 数据集统计报告")
    p(f"Mask 目录: {MASK_DIR}")
    p(f"总 tile 数: {total_tiles}")
    p("=" * 70)

    # --- 总体像素 ---
    total_px = sum(pixel_statistics.values())
    p("\n--- 总体像素统计 ---")
    for c in CLASSES:
        ratio = pixel_statistics[c] / total_px * 100
        p(f"  {CLASSES[c]:<20s} {pixel_statistics[c]:>15,d} px  {ratio:>7.2f}%")

    # --- 前景占比 ---
    p(f"\n--- 前景占比 (Foreground Ratio) ---")
    p(f"  Mean:   {foreground_ratio.mean():.4f}")
    p(f"  Median: {np.median(foreground_ratio):.4f}")
    p(f"  Std:    {foreground_ratio.std():.4f}")
    p(f"  Min:    {foreground_ratio.min():.4f}")
    p(f"  Max:    {foreground_ratio.max():.4f}")
    p(f"  P5:     {np.percentile(foreground_ratio, 5):.4f}")
    p(f"  P25:    {np.percentile(foreground_ratio, 25):.4f}")
    p(f"  P50:    {np.percentile(foreground_ratio, 50):.4f}")
    p(f"  P75:    {np.percentile(foreground_ratio, 75):.4f}")
    p(f"  P95:    {np.percentile(foreground_ratio, 95):.4f}")

    # --- 前景占比分段 ---
    thresholds = [('fg < 0.1%', 0.001), ('fg < 0.5%', 0.005), ('fg < 1%', 0.01),
                  ('fg < 5%', 0.05), ('fg > 20%', 0.20)]
    p(f"\n--- 前景占比分布 ---")
    p(f"  纯背景 tile (fg=0): {pure_bg_tiles} ({pure_bg_tiles/max(total_tiles,1)*100:.1f}%)")
    for label, th in thresholds:
        if '>' in label:
            cnt = int(np.sum(foreground_ratio > th))
        else:
            cnt = int(np.sum(foreground_ratio < th))
        p(f"  {label:<16s}: {cnt:>5d} ({cnt/max(total_tiles,1)*100:>5.1f}%)")

    # --- 每 tile 类别数 ---
    p(f"\n--- 每个 Tile 前景类别数 ---")
    for i in range(5):
        cnt = int(np.sum(tile_class_number == i))
        p(f"  {i} classes: {cnt:>5d} ({cnt/max(total_tiles,1)*100:>5.1f}%)")

    # --- 逐类面积统计 (All Tiles vs Exist Only) ---
    p(f"\n--- 逐类面积统计 ---")
    for c in FOREGROUND_CLASSES:
        arr_all = np.array(class_area_all[c], dtype=np.int64)
        arr_exist = arr_all[arr_all > 0]
        s_all, s_ex = compute_stats(arr_all, arr_exist)
        p(f"\n  [{CLASSES[c]}]")
        p(f"    Exist Tiles: {class_exist_count[c]} / {total_tiles} ({class_exist_count[c]/max(total_tiles,1)*100:.1f}%)")
        p(f"    {'':>10s} {'All Tiles':>30s} {'Exist Only':>30s}")
        for key in ['Mean', 'Std', 'Min', 'Max', 'P5', 'P25', 'P50', 'P75', 'P95']:
            p(f"    {key:>10s}: {s_all[key]:>30.1f}  {s_ex[key]:>30.1f}")

    total_fg_px = sum(pixel_statistics[c] for c in FOREGROUND_CLASSES)
    p(f"\n  Total Foreground: {total_fg_px:,} px ({total_fg_px/total_px*100:.2f}%)")
    p("=" * 70)


# ===========================
# 可视化
# ===========================
def make_plots():
    plt.rcParams['font.family'] = 'DeJavu Sans'
    plt.rcParams['font.size'] = 12

    # --- Fig 1: 前景占比分布 (直方图 + Log + CDF) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 直方图
    axes[0].hist(foreground_ratio, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(foreground_ratio), color='red', linestyle='--', label=f'Median={np.median(foreground_ratio):.3f}')
    axes[0].set_xlabel('Foreground Ratio'); axes[0].set_ylabel('Tile Count')
    axes[0].set_title('Foreground Ratio Distribution')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Log 直方图
    axes[1].hist(foreground_ratio[foreground_ratio > 0], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Foreground Ratio'); axes[1].set_ylabel('Tile Count (log)')
    axes[1].set_title('Foreground Ratio Distribution (Log Y)')
    axes[1].grid(True, alpha=0.3)

    # CDF
    sorted_fg = np.sort(foreground_ratio)
    cdf = np.arange(1, len(sorted_fg) + 1) / len(sorted_fg)
    axes[2].plot(sorted_fg, cdf, lw=2)
    axes[2].axvline(np.median(foreground_ratio), color='red', linestyle='--', label=f'Median={np.median(foreground_ratio):.3f}')
    axes[2].set_xlabel('Foreground Ratio'); axes[2].set_ylabel('CDF')
    axes[2].set_title('Foreground Ratio CDF')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Foreground_Ratio.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Fig 2: 箱线图 ---
    fg_by_class = {CLASSES[c]: np.array(class_area_all[c], dtype=np.int64) for c in FOREGROUND_CLASSES}
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([fg_by_class[CLASSES[c]] for c in FOREGROUND_CLASSES],
                    labels=[CLASSES[c] for c in FOREGROUND_CLASSES],
                    showfliers=False, patch_artist=True)
    for patch, color in zip(bp['boxes'], [COLORS[c] for c in FOREGROUND_CLASSES]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel('Pixels per Tile'); ax.set_title('Per-Class Area Distribution (Boxplot)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PerClass_Area_Boxplot.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Fig 3: 各类直方图 + Log 直方图 ---
    for c in FOREGROUND_CLASSES:
        arr_all = np.array(class_area_all[c], dtype=np.int64)
        arr_exist = arr_all[arr_all > 0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 线性直方图 (存在 tile 的)
        axes[0].hist(arr_exist, bins=40, color=COLORS[c], edgecolor='black', alpha=0.7)
        axes[0].axvline(np.median(arr_exist), color='red', linestyle='--', label=f'Median={np.median(arr_exist):.0f}')
        axes[0].set_xlabel('Pixels'); axes[0].set_ylabel('Tile Count')
        axes[0].set_title(f'{CLASSES[c]} — Area Distribution (Exist Tiles)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Log 直方图
        axes[1].hist(arr_exist, bins=40, color=COLORS[c], edgecolor='black', alpha=0.7)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Pixels'); axes[1].set_ylabel('Tile Count (log)')
        axes[1].set_title(f'{CLASSES[c]} — Area Distribution (Log Y)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{CLASSES[c]}_Distribution.png'), dpi=600, bbox_inches='tight')
        plt.close()

    # --- Fig 4: CDF 曲线 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in FOREGROUND_CLASSES:
        arr = np.array(class_area_all[c], dtype=np.int64)
        sorted_arr = np.sort(arr)
        cdf_vals = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf_vals, lw=2, color=COLORS[c], label=CLASSES[c])
    ax.set_xlabel('Pixels per Tile'); ax.set_ylabel('CDF')
    ax.set_title('Per-Class Area CDF (All Tiles)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PerClass_Area_CDF.png'), dpi=600, bbox_inches='tight')
    plt.close()


# ===========================
# CSV 输出
# ===========================
def save_csvs():
    total_px = sum(pixel_statistics.values())
    # 像素统计表
    rows = []
    for c in CLASSES:
        rows.append([CLASSES[c], CLASS_CN[c], pixel_statistics[c],
                     pixel_statistics[c] / total_px * 100])
    df = pd.DataFrame(rows, columns=['Class', 'Class_CN', 'Pixels', 'Ratio(%)'])
    df.to_csv(os.path.join(OUTPUT_DIR, 'Pixel_Statistics.csv'), index=False, encoding='utf-8-sig')

    # 逐类面积统计表
    rows2 = []
    for c in FOREGROUND_CLASSES:
        arr_all = np.array(class_area_all[c], dtype=np.int64)
        arr_exist = arr_all[arr_all > 0]
        s_all, s_ex = compute_stats(arr_all, arr_exist)
        rows2.append([
            CLASSES[c], CLASS_CN[c],
            class_exist_count[c],
            s_all['Mean'], s_all['Std'], s_all['Min'], s_all['Max'],
            s_all['P5'], s_all['P25'], s_all['P50'], s_all['P75'], s_all['P95'],
            s_ex['Mean'], s_ex['Std'], s_ex['Min'], s_ex['Max'],
            s_ex['P5'], s_ex['P25'], s_ex['P50'], s_ex['P75'], s_ex['P95'],
        ])
    cols = ['Class', 'Class_CN', 'Exist_Tiles',
            'All_Mean', 'All_Std', 'All_Min', 'All_Max',
            'All_P5', 'All_P25', 'All_P50', 'All_P75', 'All_P95',
            'Exist_Mean', 'Exist_Std', 'Exist_Min', 'Exist_Max',
            'Exist_P5', 'Exist_P25', 'Exist_P50', 'Exist_P75', 'Exist_P95']
    df2 = pd.DataFrame(rows2, columns=cols)
    df2.to_csv(os.path.join(OUTPUT_DIR, 'Class_Area_Statistics.csv'), index=False, encoding='utf-8-sig')

    # 前景占比表
    rows3 = []
    thresholds = [('fg < 0.1%', 0.001), ('fg < 0.5%', 0.005), ('fg < 1%', 0.01),
                  ('fg < 5%', 0.05), ('fg > 20%', 0.20)]
    for label, th in thresholds:
        if '>' in label:
            cnt = int(np.sum(foreground_ratio > th))
        else:
            cnt = int(np.sum(foreground_ratio < th))
        rows3.append([label, cnt, cnt / max(total_tiles, 1) * 100])
    rows3.append(['Pure BG (fg=0)', pure_bg_tiles, pure_bg_tiles / max(total_tiles, 1) * 100])
    df3 = pd.DataFrame(rows3, columns=['Condition', 'Tile_Count', 'Ratio(%)'])
    df3.to_csv(os.path.join(OUTPUT_DIR, 'Foreground_Ratio_Stats.csv'), index=False, encoding='utf-8-sig')


# ===========================
# 主流程
# ===========================
print('\n生成报告...')
with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w', encoding='utf-8') as fp:
    write_report(fp)

print('生成图表...')
make_plots()

print('保存 CSV...')
save_csvs()

# 终端摘要
total_px = sum(pixel_statistics.values())
print(f'\n{"="*60}')
print('Output Summary')
print(f'{"="*60}')
for c in CLASSES:
    print(f'{CLASSES[c]:<20s} {pixel_statistics[c]:>15,d} px  {pixel_statistics[c]/total_px*100:>6.2f}%')
print(f'{"="*60}')
print(f'全部输出保存至: {OUTPUT_DIR}')
print(f'{"="*60}')
