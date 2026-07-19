"""
Scene-level Train/Val Split for Lunar Linear Structure Segmentation.

核心原则：
  - 按原始影像（Scene）分组，同一 Scene 的 tile 不会同时出现在 train 和 val
  - 穷举 1-2 个 Scene 组合，找到类别覆盖 + 比例 + 均衡性的最优解
  - 这是遥感语义分割的标准做法，避免空间自相关导致的数据泄漏

算法：
  1. 扫描所有 tile，提取 Scene ID
  2. 读取 mask 获取每个 tile 的主导类别
  3. 以 Scene 为最小单元，穷举所有 1-2 场景组合
  4. 对每个组合评分：类别全覆盖 > 比例合适 > 分布均衡
  5. 输出 top-N 方案供选择

v8 实际数据: 5类别 (0-4), 24个Scene, 1880 tiles
"""

import os
import re
import random
import sys
import numpy as np
import rasterio
from collections import defaultdict
from itertools import combinations

# 强制 UTF-8 输出，避免 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# 配置
# ============================================================================
IMAGE_DIR = r'E:\月球_dataset\dataset\train\dataset_v8\image'
MASK_DIR = r'E:\月球_dataset\dataset\train\dataset_v8\mask'
OUTPUT_DIR = r'E:\月球_dataset\dataset\dataset_analysis'

VAL_RATIO_TARGET = 0.15
VAL_RATIO_MIN = 0.10
VAL_RATIO_MAX = 0.35      # 放宽上限以适应当前数据分布
MIN_SAMPLES_PER_CLASS = 8  # val 中每类最少主导 tile 数
SEED = 42

# v8 实际只有 5 个类别 (0-4)，Catena 是 Scene 名称前缀不是类别
CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
NUM_CLASSES = 5

random.seed(SEED)
np.random.seed(SEED)


def extract_scene(filename):
    """从文件名提取 Scene/Region ID。

    支持两种命名模式：
      - {Region}_5ch_r{row}_c{col}.tif  -> Region
      - train_{Region}_r{row}_c{col}.tif -> Region
    """
    basename = filename.replace('.tif', '').replace('.tiff', '')

    # Pattern A: {Region}_5ch_r{row}_c{col}
    m = re.match(r'^(.+?)_5ch_', basename)
    if m:
        return m.group(1)

    # Pattern B: train_{Region}_r{row}_c{col}
    m = re.match(r'^train_(.+?)_r\d+', basename)
    if m:
        return m.group(1)

    # Fallback
    m = re.match(r'^(?:train_)?(.+?)_r\d+', basename)
    if m:
        return m.group(1)

    return basename


def get_dominant_class(mask):
    """返回 mask 中像素数最多的前景类别（0=背景）。"""
    counts = {c: int(np.sum(mask == c)) for c in range(1, NUM_CLASSES)}
    total_fg = sum(counts.values())
    if total_fg == 0:
        return 0
    return max(counts, key=counts.get)


def get_class_distribution(mask):
    """返回 mask 中各类别的像素占比 [5维数组]."""
    total = mask.size
    dist = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        dist[c] = np.sum(mask == c) / total
    return dist


def jensen_shannon_divergence(p, q):
    """计算两个分布的 Jensen-Shannon 散度（对称版 KL）。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p / (p.sum() + 1e-10), 1e-10, 1.0)
    q = np.clip(q / (q.sum() + 1e-10), 1e-10, 1.0)
    m = (p + q) / 2

    def kl_div(a, b):
        return np.sum(a * np.log(a / b))

    return (kl_div(p, m) + kl_div(q, m)) / 2


def main():
    print("=" * 70)
    print("Scene-Level Train/Val Split for Dataset v8")
    print("=" * 70)

    # ========================================================================
    # 1. 扫描所有 tile，提取 Scene ID
    # ========================================================================
    print("\n[Step 1] Scanning tiles and extracting Scene IDs...")
    tiles = sorted(f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(('.tif', '.tiff')))

    scene_to_tiles = defaultdict(list)
    tile_to_scene = {}
    for t in tiles:
        scene = extract_scene(t)
        scene_to_tiles[scene].append(t)
        tile_to_scene[t] = scene

    scenes = sorted(scene_to_tiles.keys())
    print(f"  Total tiles: {len(tiles)}")
    print(f"  Total scenes: {len(scenes)}")

    # ========================================================================
    # 2. 读取 mask，获取每个 tile 的类别分布
    # ========================================================================
    print("\n[Step 2] Reading masks and computing class distributions...")

    tile_class = {}           # tile -> dominant class
    tile_class_dist = {}      # tile -> pixel-level class distribution
    scene_class_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    scene_class_pixels = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    class_to_tiles = defaultdict(list)

    for i, tile in enumerate(tiles):
        mask_path = os.path.join(MASK_DIR, tile)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        mask[(mask >= NUM_CLASSES) | (mask < 0)] = 0
        dom_class = get_dominant_class(mask)
        tile_class[tile] = dom_class
        tile_class_dist[tile] = get_class_distribution(mask)
        class_to_tiles[dom_class].append(tile)
        scene_class_counts[tile_to_scene[tile]][dom_class] += 1

        for c in range(NUM_CLASSES):
            scene_class_pixels[tile_to_scene[tile]][c] += int(np.sum(mask == c))

        if (i + 1) % 400 == 0:
            print(f"  Processed [{i+1}/{len(tiles)}]...")

    print(f"  Done! {len(tiles)} masks read.")

    # ========================================================================
    # 3. 全局统计
    # ========================================================================
    print("\n[Step 3] Global class distribution (by dominant class):")
    global_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for c in range(NUM_CLASSES):
        global_class_counts[c] = len(class_to_tiles[c])

    print(f"  {'Class':<20} {'Tiles':>8} {'Ratio':>8}")
    print(f"  {'-'*38}")
    for c in range(NUM_CLASSES):
        pct = global_class_counts[c] / len(tiles) * 100
        print(f"  {CLASS_NAMES[c]:<20} {global_class_counts[c]:>8} {pct:>7.2f}%")

    # Foreground-only distribution (excluding background for balance scoring)
    fg_total = sum(global_class_counts[1:])
    global_fg_dist = np.zeros(NUM_CLASSES - 1, dtype=np.float64)
    for c in range(1, NUM_CLASSES):
        global_fg_dist[c - 1] = global_class_counts[c] / fg_total if fg_total > 0 else 0

    # ========================================================================
    # 4. 各 Scene 概况
    # ========================================================================
    print(f"\n[Step 4] Per-scene class distribution (by dominant class):")
    header = f"  {'Scene':<26} {'Tiles':>6} " + " ".join(f"{CLASS_NAMES[c]:>6}" for c in range(NUM_CLASSES))
    print(header)
    print(f"  {'-'*(38 + 7*NUM_CLASSES)}")
    for scene in sorted(scene_to_tiles.keys(), key=lambda s: -len(scene_to_tiles[s])):
        counts = scene_class_counts[scene].astype(int)
        n = len(scene_to_tiles[scene])
        cls_str = " ".join(f"{counts[c]:>6}" for c in range(NUM_CLASSES))
        print(f"  {scene:<26} {n:>6} {cls_str}")

    # ========================================================================
    # 5. 穷举 Scene 组合评分
    # ========================================================================
    print(f"\n[Step 5] Exhaustive search over 1-2 scene combinations...")

    total_tiles = len(tiles)

    def evaluate_combination(val_scene_list):
        """对一组 val scene 计算综合评分。返回 (score, details_dict)

        同时评估 val 和 train 两侧的质量：
        - val 侧：类别覆盖、比例合适、分布均衡
        - train 侧：每类必须有足够样本供模型学习（硬约束）
        """
        val_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        val_tiles_count = 0
        for s in val_scene_list:
            val_counts += scene_class_counts[s]
            val_tiles_count += len(scene_to_tiles[s])

        train_counts = global_class_counts - val_counts
        val_ratio = val_tiles_count / total_tiles

        # 硬约束：比例必须在范围内
        if val_ratio < VAL_RATIO_MIN or val_ratio > VAL_RATIO_MAX:
            return -float('inf'), None

        # 硬约束：val 中所有类别必须出现
        if np.any(val_counts == 0):
            return -float('inf'), None

        # 硬约束：train 中每类前景至少 30 个 tile（否则模型学不会）
        TRAIN_MIN_FG = 30
        if np.any(train_counts[1:] < TRAIN_MIN_FG):
            return -float('inf'), None

        min_class = np.min(val_counts)
        fg_min_class = np.min(val_counts[1:])  # 前景类最小值
        train_fg_min = int(np.min(train_counts[1:]))

        # 1. 最小类别数（短板分）：每类 >= MIN_SAMPLES，越高越好
        if fg_min_class >= MIN_SAMPLES_PER_CLASS:
            min_score = 50.0 + fg_min_class * 2.0
        else:
            min_score = fg_min_class * 5.0

        # 2. 比例分：越接近目标越好（用高斯核）
        ratio_deviation = abs(val_ratio - VAL_RATIO_TARGET)
        ratio_score = 30.0 * np.exp(-0.5 * (ratio_deviation / 0.05) ** 2)

        # 3. 前景分布均衡分：val vs global JS 散度
        val_fg_dist = np.zeros(NUM_CLASSES - 1, dtype=np.float64)
        fg_sum = sum(val_counts[1:])
        if fg_sum > 0:
            for c in range(1, NUM_CLASSES):
                val_fg_dist[c - 1] = val_counts[c] / fg_sum
        js_div = jensen_shannon_divergence(val_fg_dist, global_fg_dist)
        balance_score = 20.0 * (1.0 - min(js_div * 5, 1.0))

        # 4. Train 侧质量分：train 各类别越均衡越好
        train_fg_dist = np.zeros(NUM_CLASSES - 1, dtype=np.float64)
        train_fg_sum = sum(train_counts[1:])
        if train_fg_sum > 0:
            for c in range(1, NUM_CLASSES):
                train_fg_dist[c - 1] = train_counts[c] / train_fg_sum
        js_train = jensen_shannon_divergence(train_fg_dist, global_fg_dist)
        train_quality_score = 15.0 * (1.0 - min(js_train * 5, 1.0))

        total_score = min_score + ratio_score + balance_score + train_quality_score

        details = {
            'val_scenes': val_scene_list,
            'val_tiles': val_tiles_count,
            'val_ratio': val_ratio,
            'val_counts': val_counts,
            'train_counts': train_counts,
            'min_class': int(min_class),
            'fg_min_class': int(fg_min_class),
            'train_fg_min': train_fg_min,
            'js_div': js_div,
            'js_train': js_train,
            'score_breakdown': (min_score, ratio_score, balance_score, train_quality_score),
        }
        return total_score, details

    # 生成所有 1-scene 和 2-scene 组合
    all_combos = []
    # 单 scene
    for s in scenes:
        all_combos.append([s])
    # 双 scene
    for s1, s2 in combinations(scenes, 2):
        all_combos.append([s1, s2])

    print(f"  Evaluating {len(all_combos)} combinations...")

    results = []
    for combo in all_combos:
        score, details = evaluate_combination(combo)
        if score > -float('inf'):
            results.append((score, details))

    results.sort(key=lambda x: -x[0])

    # ========================================================================
    # 6. 展示 Top-N 方案
    # ========================================================================
    TOP_N = min(10, len(results))

    print(f"\n[Step 6] Top-{TOP_N} candidates (out of {len(results)} valid combinations):")
    print(f"  Criteria: all 5 classes present, ratio in [{VAL_RATIO_MIN*100:.0f}%-{VAL_RATIO_MAX*100:.0f}%]")
    print()

    for rank, (score, d) in enumerate(results[:TOP_N]):
        print(f"  --- Rank {rank+1} (score={score:.1f}) ---")
        scene_names = " + ".join(d['val_scenes'])
        print(f"  Val scenes: {scene_names}")
        print(f"  Val tiles: {d['val_tiles']}/{total_tiles} ({d['val_ratio']*100:.1f}%)")
        print(f"  {'Class':<20} {'Train':>6} {'Val':>6} {'Train%':>7} {'Val%':>7}")
        print(f"  {'-'*48}")
        for c in range(NUM_CLASSES):
            tr = int(d['train_counts'][c])
            vl = int(d['val_counts'][c])
            tr_pct = tr / (total_tiles - d['val_tiles']) * 100
            vl_pct = vl / d['val_tiles'] * 100 if d['val_tiles'] > 0 else 0
            flag = " !" if tr < 30 and c > 0 else ""
            print(f"  {CLASS_NAMES[c]:<20} {tr:>6} {vl:>6} {tr_pct:>6.1f}% {vl_pct:>6.1f}%{flag}")
        ms, rs, bs, ts = d['score_breakdown']
        print(f"  Scores: min_class={ms:.1f}, ratio={rs:.1f}, balance={bs:.1f}, train_quality={ts:.1f}")
        print(f"  JS div: val={d['js_div']:.5f}, train={d['js_train']:.5f}")
        print()

    if not results:
        print("  NO valid combination found! Consider relaxing constraints.")
        print("  Checking which constraints fail...")
        # 单 scene 分析
        for s in scenes:
            counts = scene_class_counts[s].astype(int)
            n = len(scene_to_tiles[s])
            ratio = n / total_tiles
            missing = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if counts[c] == 0]
            print(f"  {s}: {n} tiles ({ratio*100:.1f}%), missing: {missing if missing else 'NONE'}")

    # ========================================================================
    # 7. 输出最佳方案
    # ========================================================================
    if results:
        best = results[0][1]
        val_scenes_selected = best['val_scenes']
    else:
        # Fallback: 放宽约束到 3 scenes 或更大比例
        print("\n  Falling back: checking single scenes without ratio constraint...")
        for s in sorted(scenes, key=lambda s: -len(scene_to_tiles[s])):
            counts = scene_class_counts[s].astype(int)
            has_all = np.all(counts > 0)
            n = len(scene_to_tiles[s])
            print(f"  {s}: {n} tiles ({n/total_tiles*100:.1f}%), all_classes={has_all}")
        return  # No valid result

    print(f"\n[Step 7] Applying best split: {' + '.join(val_scenes_selected)}")

    # 生成 tile 列表
    val_tile_list = []
    train_tile_list = []
    for tile in tiles:
        if tile_to_scene[tile] in val_scenes_selected:
            val_tile_list.append(tile)
        else:
            train_tile_list.append(tile)

    # 统计
    val_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    train_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for t in train_tile_list:
        train_class_counts[tile_class[t]] += 1
    for t in val_tile_list:
        val_class_counts[tile_class[t]] += 1

    print(f"\n  --- Split summary ---")
    print(f"  Train: {len(train_tile_list)} tiles ({len(train_tile_list)/total_tiles*100:.1f}%), "
          f"{len(scenes) - len(val_scenes_selected)} scenes")
    print(f"  Val:   {len(val_tile_list)} tiles ({len(val_tile_list)/total_tiles*100:.1f}%), "
          f"{len(val_scenes_selected)} scenes")

    print(f"\n  --- Class distribution comparison ---")
    print(f"  {'Class':<20} {'Train':>8} {'Train%':>8} {'Val':>8} {'Val%':>8} {'Global%':>8}")
    print(f"  {'-'*58}")
    for c in range(NUM_CLASSES):
        tr = train_class_counts[c]
        vl = val_class_counts[c]
        gl = global_class_counts[c]
        print(f"  {CLASS_NAMES[c]:<20} {tr:>8} {tr/len(train_tile_list)*100:>7.2f}% "
              f"{vl:>8} {vl/len(val_tile_list)*100:>7.2f}% "
              f"{gl/len(tiles)*100:>7.2f}%")

    # 像素级分布对比
    print(f"\n  --- Pixel-level class distribution ---")
    val_pixel_dist = np.zeros(NUM_CLASSES, dtype=np.float64)
    train_pixel_dist = np.zeros(NUM_CLASSES, dtype=np.float64)
    global_pixel_dist = np.zeros(NUM_CLASSES, dtype=np.float64)

    for t in tiles:
        dist = tile_class_dist[t]
        global_pixel_dist += dist
        if tile_to_scene[t] in val_scenes_selected:
            val_pixel_dist += dist
        else:
            train_pixel_dist += dist

    global_pixel_dist /= global_pixel_dist.sum()
    val_pixel_dist /= val_pixel_dist.sum()
    train_pixel_dist /= train_pixel_dist.sum()

    print(f"  {'Class':<20} {'Global%':>8} {'Train%':>8} {'Val%':>8}")
    print(f"  {'-'*46}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<20} "
              f"{global_pixel_dist[c]*100:>7.2f}% "
              f"{train_pixel_dist[c]*100:>7.2f}% "
              f"{val_pixel_dist[c]*100:>7.2f}%")

    js_train = jensen_shannon_divergence(train_pixel_dist, global_pixel_dist)
    js_val = jensen_shannon_divergence(val_pixel_dist, global_pixel_dist)
    print(f"\n  Train-Global JS divergence: {js_train:.6f}")
    print(f"  Val-Global   JS divergence: {js_val:.6f}")

    # ========================================================================
    # 8. 写入文件
    # ========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_file = os.path.join(OUTPUT_DIR, 'valid_tiles_train_scene.txt')
    val_file = os.path.join(OUTPUT_DIR, 'valid_tiles_val_scene.txt')
    # 同时保存 scene 级别的划分信息
    info_file = os.path.join(OUTPUT_DIR, 'scene_split_info.txt')

    with open(train_file, 'w', encoding='utf-8') as f:
        for t in sorted(train_tile_list):
            f.write(t + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for t in sorted(val_tile_list):
            f.write(t + '\n')

    # 保存详细划分信息
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Scene-Level Split Info for Dataset v8\n")
        f.write(f"{'='*50}\n")
        f.write(f"Val scenes: {' + '.join(val_scenes_selected)}\n")
        f.write(f"Train tiles: {len(train_tile_list)} ({len(train_tile_list)/total_tiles*100:.1f}%)\n")
        f.write(f"Val tiles:   {len(val_tile_list)} ({len(val_tile_list)/total_tiles*100:.1f}%)\n\n")
        f.write(f"Per-scene breakdown:\n")
        for s in sorted(scene_to_tiles.keys(), key=lambda s: -len(scene_to_tiles[s])):
            role = "VAL" if s in val_scenes_selected else "TRAIN"
            f.write(f"  [{role}] {s}: {len(scene_to_tiles[s])} tiles\n")

    print(f"\n[Step 8] Output files:")
    print(f"  Train list: {train_file} ({len(train_tile_list)} tiles)")
    print(f"  Val list:   {val_file} ({len(val_tile_list)} tiles)")
    print(f"  Split info: {info_file}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
