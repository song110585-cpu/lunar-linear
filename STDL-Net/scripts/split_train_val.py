"""
月球线性构造分割数据集：分层随机划分训练集与验证集 (Stratified Train/Val Split)

功能:
1. 读取 valid_tiles_train.txt 中的有效切片。
2. 逐一读取其 mask 标签，确定其主导前景类 (Wrinkle Ridge, Rille, Fault, Graben)。
   - 如果无前景，归为 Background (0)。
   - 如果有多个前景，选择像素最多的那个前景类别。
3. 按照 90% 训练 / 10% 验证的比例，按类别进行分层抽样 (Stratified Split)，确保稀有类别 (如 Graben/Rille) 分布比例完全一致。
4. 输出:
   - valid_tiles_train_split.txt (90% 训练集)
   - valid_tiles_val.txt (10% 验证集)
5. 打印划分前后的类别分布，便于查验。
"""

import os
import random
import numpy as np
import rasterio
from collections import defaultdict

# ============================================================================
# 配置
# ============================================================================
MASK_DIR = r'E:\月球_dataset\Research area\train\dataset_v6\mask'
FILTER_DIR = r'E:\月球_dataset\Research area\dataset_analysis'
VALID_TRAIN_FILE = os.path.join(FILTER_DIR, 'valid_tiles_train.txt')

# 输出文件路径
OUT_TRAIN_FILE = os.path.join(FILTER_DIR, 'valid_tiles_train_split.txt')
OUT_VAL_FILE = os.path.join(FILTER_DIR, 'valid_tiles_val.txt')

VAL_RATIO = 0.10
SEED = 42

CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
NUM_CLASSES = 5

def main():
    print("=== 开始进行分层随机划分 (Train/Val Split) ===")
    
    # 1. 确保随机种子固定，保证划分可复现
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 2. 读取有效切片列表
    if os.path.exists(VALID_TRAIN_FILE):
        with open(VALID_TRAIN_FILE, 'r', encoding='utf-8') as f:
            tiles = [line.strip() for line in f if line.strip()]
        print(f"[Info] 成功加载有效切片列表 ({VALID_TRAIN_FILE}): 共 {len(tiles)} 张")
    else:
        # 如果不存在过滤后的列表，则默认读取目录下所有 tif
        print(f"[Warning] 未找到有效列表 {VALID_TRAIN_FILE}，将直接扫描 mask 目录")
        tiles = sorted(f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.tif', '.tiff')))
        print(f"扫描到共 {len(tiles)} 张切片")

    # 3. 预扫描 mask 目录，建立文件名集合 (快速查找，避免反复 os.path.exists)
    print("正在扫描 mask 目录...")
    mask_files = set()
    for f in os.listdir(MASK_DIR):
        if f.lower().endswith(('.tif', '.tiff')):
            mask_files.add(f)
    print(f"[Info] mask 目录共 {len(mask_files)} 个文件")

    # 统计每个切片的类别属性
    print("正在分析切片 mask 类别属性，以进行精确分层...")
    tile_to_class = {}
    class_to_tiles = defaultdict(list)
    missing_count = 0

    for i, fname in enumerate(tiles):
        # 生成候选 mask 文件名, 覆盖三种实际命名差异:
        #   Aristarchus_5ch_r000  → mask: train_Aristarchus_r000
        #   Mare Serenitatis_5ch  → mask: train_Mare Serenitatis_r000
        #   Marius Hills_5ch_r000 → mask: Marius Hills_5ch_r000 (同名)
        candidates = [fname]
        # _5ch 在文件名中 => 也试去掉 _5ch, 以及加 train_ / test_ 前缀
        for tag in ("_5ch", "_3ch", "_2ch"):
            if tag in fname:
                no_tag = fname.replace(tag, "")
                candidates.append(no_tag)
                # 抽取纯区域名: "Aristarchus_5ch_r000_c000.tif" -> "train_Aristarchus_r000_c000.tif"
                parts = fname.split(tag, 1)
                region = parts[0]                     # "Aristarchus"
                suffix = parts[1]                     # "_r000_c000.tif"
                for prefix in ("train_", "test_", ""):
                    candidates.append(f"{prefix}{region}{suffix}")
                break  # 只处理第一个匹配的 tag

        mask_name = None
        for cand in candidates:
            if cand in mask_files:
                mask_name = cand
                break

        if mask_name is None:
            missing_count += 1
            if missing_count <= 10:
                print(f"[Warning] 找不到 mask 文件: {fname}")
            continue

        mask_path = os.path.join(MASK_DIR, mask_name)
            
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
            
        # 双保险修正无效类别
        mask[(mask > 4) | (mask < 0)] = 0
        
        # 统计各类别像素数
        counts = {c: np.sum(mask == c) for c in range(1, NUM_CLASSES)}  # 1 到 4 类的像素数
        total_fg = sum(counts.values())
        
        if total_fg == 0:
            # 纯背景图
            dom_class = 0
        else:
            # 找出像素最多的前景类别作为该 tile 的分层标签
            dom_class = max(counts, key=counts.get)
            
        tile_to_class[fname] = dom_class
        class_to_tiles[dom_class].append(fname)
        
        if (i + 1) % 200 == 0 or (i + 1) == len(tiles):
            print(f"  已分析 [{i+1}/{len(tiles)}]...")

    if missing_count > 0:
        print(f"\n[Info] 共 {missing_count} 张切片在 mask 目录中找不到，已跳过")
        print(f"[Info] 实际参与划分: {len(tile_to_class)} 张")

    print("\n=== 原始类别分布统计 ===")
    for c in range(NUM_CLASSES):
        count = len(class_to_tiles[c])
        pct = count / len(tile_to_class) * 100
        print(f"  类别 {c} ({CLASS_NAMES[c]}): 主导切片数 = {count} ({pct:.2f}%)")

    # 4. 按类别分层抽样
    train_split_tiles = []
    val_tiles = []
    
    print(f"\n开始按 {int((1-VAL_RATIO)*100)}/{int(VAL_RATIO*100)} 的比例分层划分...")
    
    for c in range(NUM_CLASSES):
        cls_tiles = class_to_tiles[c]
        # 随机打乱当前类别的切片
        random.shuffle(cls_tiles)
        
        # 计算划分界限
        n_val = max(1, int(len(cls_tiles) * VAL_RATIO)) if len(cls_tiles) > 0 else 0
        # 边缘情况处理
        if len(cls_tiles) == 1:
            # 只有1个切片时，分给训练集
            n_val = 0
            
        cls_val = cls_tiles[:n_val]
        cls_train = cls_tiles[n_val:]
        
        val_tiles.extend(cls_val)
        train_split_tiles.extend(cls_train)
        
        print(f"  类别 {c} ({CLASS_NAMES[c]}): 分配 Train={len(cls_train)}，Val={len(cls_val)}")

    # 再次打乱结果以保证训练/验证读取的多样性
    random.shuffle(train_split_tiles)
    random.shuffle(val_tiles)

    # 5. 写出结果文件
    os.makedirs(os.path.dirname(OUT_TRAIN_FILE), exist_ok=True)
    
    with open(OUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for t in train_split_tiles:
            f.write(t + '\n')
            
    with open(OUT_VAL_FILE, 'w', encoding='utf-8') as f:
        for t in val_tiles:
            f.write(t + '\n')

    print("\n=== 划分完成！ ===")
    print(f"[Save] 训练集列表已保存至: {OUT_TRAIN_FILE} (共 {len(train_split_tiles)} 张)")
    print(f"[Save] 验证集列表已保存至: {OUT_VAL_FILE} (共 {len(val_tiles)} 张)")
    print(f"总计: {len(train_split_tiles) + len(val_tiles)} 张 (训练 {len(train_split_tiles)/len(tiles)*100:.1f}% / 验证 {len(val_tiles)/len(tiles)*100:.1f}%)")

    # 6. 分布验证
    print("\n=== 验证划分均衡性 ===")
    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)
    
    for t in train_split_tiles:
        train_class_counts[tile_to_class[t]] += 1
    for t in val_tiles:
        val_class_counts[tile_to_class[t]] += 1
        
    print(f"{'类别':<15} | {'训练集主导数 (比例)':<20} | {'验证集主导数 (比例)':<20}")
    print("-" * 65)
    for c in range(NUM_CLASSES):
        tr_count = train_class_counts[c]
        val_count = val_class_counts[c]
        tr_pct = tr_count / len(train_split_tiles) * 100 if len(train_split_tiles) > 0 else 0
        val_pct = val_count / len(val_tiles) * 100 if len(val_tiles) > 0 else 0
        
        print(f"{CLASS_NAMES[c]:<15} | {tr_count:<8} ({tr_pct:5.2f}%)       | {val_count:<8} ({val_pct:5.2f}%)")

if __name__ == '__main__':
    main()
