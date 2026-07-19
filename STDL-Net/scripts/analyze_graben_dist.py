"""
分析 v8 训练集中 Graben (class=4) 的来源分布
统计：各 Scene 贡献的含 Graben tile 数量 + 像素数
用法：
    python scripts/analyze_graben_dist.py
"""
import os
import re
import sys
import numpy as np

# 强制 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# 配置 (按实际路径修改)
# ============================================================================
MASK_DIR       = r'E:\月球_dataset\dataset\datasetv8\train\mask'
TRAIN_LIST_TXT = r'E:\月球_dataset\dataset\datasetv8\pretrain\valid_tiles_train_scene.txt'
GRABEN_CLASS   = 4
CLASS_NAMES    = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
NUM_CLASSES    = 5

# 识别 Mare Serenitatis 的关键词 (大小写不敏感)
MARE_KEYWORDS  = ['mareserenitatis', 'mare_serenitatis', 'serenitatis']
# ============================================================================


def extract_scene(filename):
    basename = filename.replace('.tif', '').replace('.tiff', '').replace('.png', '')
    m = re.match(r'^(.+?)_5ch_', basename)
    if m:
        return m.group(1)
    m = re.match(r'^train_(.+?)_r\d+', basename)
    if m:
        return m.group(1)
    m = re.match(r'^(?:train_)?(.+?)_r\d+', basename)
    if m:
        return m.group(1)
    return basename


def is_mare_serenitatis(scene_name):
    s = scene_name.lower().replace('-', '').replace(' ', '')
    return any(kw in s for kw in MARE_KEYWORDS)


def read_mask(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1).astype(np.int64)
    else:
        from PIL import Image
        return np.array(Image.open(path)).astype(np.int64)


def find_mask(tile_name, mask_dir):
    stem = re.sub(r'\.(tif|tiff|png)$', '', tile_name, flags=re.IGNORECASE)
    for ext in ['.tif', '.tiff', '.png']:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def main():
    # 读取 train tile 列表
    with open(TRAIN_LIST_TXT, encoding='utf-8') as f:
        train_tiles = [line.strip() for line in f if line.strip()]
    print(f'训练集 tile 总数: {len(train_tiles)}')

    # 按 scene 聚合
    from collections import defaultdict
    scene_total       = defaultdict(int)    # scene -> 总 tile 数
    scene_graben_tiles = defaultdict(int)   # scene -> 含 Graben tile 数
    scene_graben_pixels = defaultdict(int)  # scene -> Graben 像素数
    scene_total_pixels = defaultdict(int)   # scene -> 总像素数

    missing = 0
    for idx, tile in enumerate(train_tiles):
        scene = extract_scene(tile)
        scene_total[scene] += 1

        mask_path = find_mask(tile, MASK_DIR)
        if mask_path is None:
            missing += 1
            continue

        mask = read_mask(mask_path)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        graben_px = int(np.sum(mask == GRABEN_CLASS))
        total_px  = mask.size

        scene_total_pixels[scene] += total_px
        scene_graben_pixels[scene] += graben_px
        if graben_px > 0:
            scene_graben_tiles[scene] += 1

        if (idx + 1) % 200 == 0:
            print(f'  [{idx+1}/{len(train_tiles)}] 处理中...')

    if missing > 0:
        print(f'  警告: {missing} 个 tile 未找到对应 mask，已跳过')

    # ---- 汇总输出 ----
    all_scenes = sorted(scene_total.keys())

    total_graben_tiles  = sum(scene_graben_tiles.values())
    total_graben_pixels = sum(scene_graben_pixels.values())
    total_pixels_all    = sum(scene_total_pixels.values())

    print()
    print('=' * 75)
    print(f'{"Scene":<30} {"总Tile":>7} {"含Graben Tile":>13} {"Graben像素":>12} {"Graben像素%":>11}  来源')
    print('-' * 75)

    mare_graben_tiles  = 0
    other_graben_tiles = 0
    mare_graben_pixels = 0
    other_graben_pixels = 0

    for scene in sorted(all_scenes, key=lambda s: -scene_graben_tiles[s]):
        n_tile   = scene_total[scene]
        g_tile   = scene_graben_tiles[scene]
        g_px     = scene_graben_pixels[scene]
        tot_px   = scene_total_pixels[scene]
        pct_px   = g_px / tot_px * 100 if tot_px > 0 else 0
        is_mare  = is_mare_serenitatis(scene)
        tag      = '<<< Mare Serenitatis' if is_mare else ''

        if is_mare:
            mare_graben_tiles  += g_tile
            mare_graben_pixels += g_px
        else:
            other_graben_tiles  += g_tile
            other_graben_pixels += g_px

        if g_tile > 0:
            print(f'{scene:<30} {n_tile:>7} {g_tile:>13} {g_px:>12,} {pct_px:>10.2f}%  {tag}')

    print('-' * 75)
    print(f'{"[合计]":<30} {len(train_tiles):>7} {total_graben_tiles:>13} {total_graben_pixels:>12,} '
          f'{total_graben_pixels/total_pixels_all*100:>10.2f}%')
    print()
    print(f'来源拆分:')
    print(f'  Mare Serenitatis 上半部:  {mare_graben_tiles:>5} tiles  {mare_graben_pixels:>12,} px '
          f'({mare_graben_pixels/max(total_graben_pixels,1)*100:.1f}% of all Graben px)')
    print(f'  其他 Scene:               {other_graben_tiles:>5} tiles  {other_graben_pixels:>12,} px '
          f'({other_graben_pixels/max(total_graben_pixels,1)*100:.1f}% of all Graben px)')
    print('=' * 75)


if __name__ == '__main__':
    main()
