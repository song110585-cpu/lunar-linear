"""
把 v5 的 train + test 全部合并，随机 8:1:1 划分。
输出: datasetv5_random811/train/ val/ test/  各有 image/ mask/
"""
import os, sys, shutil, random
import numpy as np

SRC_BASE = r"E:\月球_dataset\dataset\datasetv5\datasetv5"   # v5 解压后嵌套了一层
DST_BASE = r"E:\月球_dataset\dataset\datasetv5_random811"
SEED = 42

def collect_tiles(split_name):
    """收集某个 split 下的所有 image/mask 对"""
    img_dir = os.path.join(SRC_BASE, split_name, 'image')
    mask_dir = os.path.join(SRC_BASE, split_name, 'mask')
    tiles = []
    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            tiles.append((fname, img_path, mask_path))
    return tiles

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. 收集全部 tile
    all_tiles = []
    for split in ['train', 'test']:
        tiles = collect_tiles(split)
        print(f'  {split}: {len(tiles)} tiles')
        all_tiles.extend(tiles)

    print(f'\n  总计: {len(all_tiles)} tiles')

    # 2. 随机打乱
    random.shuffle(all_tiles)

    # 3. 8:1:1 划分
    n = len(all_tiles)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_tiles = all_tiles[:n_train]
    val_tiles = all_tiles[n_train:n_train + n_val]
    test_tiles = all_tiles[n_train + n_val:]

    print(f'\n  Train: {len(train_tiles)} ({len(train_tiles)/n*100:.1f}%)')
    print(f'  Val:   {len(val_tiles)} ({len(val_tiles)/n*100:.1f}%)')
    print(f'  Test:  {len(test_tiles)} ({len(test_tiles)/n*100:.1f}%)')

    # 4. 复制文件
    for split_name, tiles in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
        img_dst = os.path.join(DST_BASE, split_name, 'image')
        mask_dst = os.path.join(DST_BASE, split_name, 'mask')
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        for fname, img_src, mask_src in tiles:
            shutil.copy2(img_src, os.path.join(img_dst, fname))
            shutil.copy2(mask_src, os.path.join(mask_dst, fname))

        print(f'  {split_name}: {len(tiles)} tiles copied → {img_dst}')

    # 5. 输出 tile 列表（方便后续复查）
    list_dir = os.path.join(DST_BASE, 'split_lists')
    os.makedirs(list_dir, exist_ok=True)
    for split_name, tiles in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
        with open(os.path.join(list_dir, f'{split_name}.txt'), 'w') as f:
            for fname, _, _ in tiles:
                f.write(fname + '\n')

    print(f'\n  Done! 输出目录: {DST_BASE}')
    print(f'  Tile 列表: {list_dir}')

if __name__ == '__main__':
    main()
