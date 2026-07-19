"""
重新切分 v8 数据集：Mare Serenitatis 按行分析 Graben 密度，
找出最佳切分行，确保训练集含足够的 Mare 型 Graben。

逻辑：
  - 扫描 v8 train 图像目录下所有 Mare Serenitatis tile
  - 按行号分组，统计各行每类的像素数
  - 找最佳分割行（Graben 丰富的行留给 train，WR 丰富的行给 val）
  - 重新输出 valid_tiles_train_scene.txt / valid_tiles_val_scene.txt

用法：
    python scripts/resplit_mare_graben.py
"""
import os
import re
import sys
import numpy as np
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# 配置
# ============================================================================
MASK_DIR   = r'E:\月球_dataset\dataset\datasetv8\train\mask'
IMAGE_DIR  = r'E:\月球_dataset\dataset\datasetv8\train\image'
OUTPUT_DIR = r'E:\月球_dataset\dataset\datasetv8\pretrain'

CLASS_NAMES   = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
NUM_CLASSES   = 5
GRABEN_CLASS  = 4
MARE_KEYWORDS = ['mareserenitatis', 'mare_serenitatis', 'serenitatis']

# 新切分行：低于此行号 → val（WR 丰富），高于等于此行号 → train（Graben 丰富）
# 先设 None，脚本自动推荐后可手动改
SPLIT_ROW = None   # 例如 3684；None = 自动推荐
# ============================================================================


def extract_scene(filename):
    basename = re.sub(r'\.(tif|tiff|png)$', '', filename, flags=re.IGNORECASE)
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


def extract_row(filename):
    m = re.search(r'_r(\d+)', filename)
    return int(m.group(1)) if m else -1


def is_mare(scene_name):
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
    tiles = sorted(f for f in os.listdir(IMAGE_DIR)
                   if re.search(r'\.(tif|tiff|png)$', f, re.IGNORECASE))
    print(f'v8 train 目录共 {len(tiles)} 个 tile')

    # ── 扫描所有 Mare Serenitatis tile ────────────────────────────────────────
    mare_tiles = [(t, extract_row(t)) for t in tiles if is_mare(extract_scene(t))]
    other_tiles = [t for t in tiles if not is_mare(extract_scene(t))]

    print(f'Mare Serenitatis tile: {len(mare_tiles)}')
    print(f'其他 Scene tile:        {len(other_tiles)}')

    # ── 按行统计各类像素 ────────────────────────────────────────────────────
    print('\n[1/3] 读取 Mare Serenitatis masks，按行统计...')
    row_pixels  = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.int64))  # row -> [px per class]
    row_tiles   = defaultdict(int)
    missing     = 0

    for idx, (tile, row) in enumerate(sorted(mare_tiles, key=lambda x: x[1])):
        mask_path = find_mask(tile, MASK_DIR)
        if mask_path is None:
            missing += 1
            continue
        mask = np.clip(read_mask(mask_path), 0, NUM_CLASSES - 1)
        for c in range(NUM_CLASSES):
            row_pixels[row][c] += int(np.sum(mask == c))
        row_tiles[row] += 1
        if (idx + 1) % 50 == 0:
            print(f'  [{idx+1}/{len(mare_tiles)}] ...')

    if missing:
        print(f'  警告: {missing} 个 tile 缺 mask，已跳过')

    # ── 逐行打印 ────────────────────────────────────────────────────────────
    rows = sorted(row_pixels.keys())
    print()
    print('=' * 90)
    print(f'{"行号":>8} {"Tile数":>6}  ' +
          '  '.join(f'{CLASS_NAMES[c]:>12}px  {CLASS_NAMES[c][:2]}%' for c in range(1, NUM_CLASSES)))
    print('-' * 90)

    row_graben_pct = {}
    for row in rows:
        px   = row_pixels[row]
        tot  = px.sum()
        cols = ''
        for c in range(1, NUM_CLASSES):
            pct = px[c] / tot * 100 if tot > 0 else 0
            cols += f'  {px[c]:>12,}  {pct:>4.1f}%'
        graben_pct = px[GRABEN_CLASS] / tot * 100 if tot > 0 else 0
        row_graben_pct[row] = graben_pct
        print(f'{row:>8} {row_tiles[row]:>6} {cols}')

    print('=' * 90)

    # ── 自动推荐切分行 ───────────────────────────────────────────────────────
    # 策略：找一个行号 R，使得 R 以上（低行号）val 仍有足够5类，
    #       同时 R 以下（高行号）进入 train 包含足量 Graben
    # 简单策略：按 cumulative Graben px 找 50% 分位点行，从该行往下给 train
    all_graben = sum(row_pixels[r][GRABEN_CLASS] for r in rows)
    cum = 0
    recommended_row = rows[len(rows)//2]  # 默认中间
    for row in rows:
        cum += row_pixels[row][GRABEN_CLASS]
        if cum >= all_graben * 0.50:
            recommended_row = row
            break

    print(f'\n自动推荐切分行: {recommended_row}')
    print(f'  行 < {recommended_row}  → Val（低行号，WR 更丰富）')
    print(f'  行 >= {recommended_row} → Train（高行号，Graben 更丰富）')
    print(f'  （SPLIT_ROW = None 时使用此值，可在脚本顶部手动修改 SPLIT_ROW）')

    split_row = SPLIT_ROW if SPLIT_ROW is not None else recommended_row

    # ── 按新切分行重新划分 ───────────────────────────────────────────────────
    print(f'\n[2/3] 使用切分行 {split_row} 重新划分...')

    new_val_tiles   = []
    new_train_tiles = []

    for tile, row in mare_tiles:
        if row < split_row:
            new_val_tiles.append(tile)
        else:
            new_train_tiles.append(tile)

    new_train_tiles += other_tiles

    print(f'  新 Train: {len(new_train_tiles)} tiles  '
          f'（含 {len([t for t,r in mare_tiles if r >= split_row])} Mare 高行 + {len(other_tiles)} 其他）')
    print(f'  新 Val:   {len(new_val_tiles)} tiles  （Mare 低行）')

    # ── 验证新 val 的类别覆盖 ─────────────────────────────────────────────────
    print('\n  新 Val 类别覆盖（像素级）:')
    val_px = np.zeros(NUM_CLASSES, dtype=np.int64)
    for tile, row in mare_tiles:
        if row < split_row:
            mask_path = find_mask(tile, MASK_DIR)
            if mask_path:
                mask = np.clip(read_mask(mask_path), 0, NUM_CLASSES - 1)
                for c in range(NUM_CLASSES):
                    val_px[c] += int(np.sum(mask == c))

    tot_val = val_px.sum()
    for c in range(NUM_CLASSES):
        flag = '  *** 缺失!' if val_px[c] == 0 and c > 0 else ''
        print(f'    {CLASS_NAMES[c]:<18}: {val_px[c]:>12,}  ({val_px[c]/tot_val*100:.2f}%){flag}')

    # ── 写入新 txt ────────────────────────────────────────────────────────────
    print(f'\n[3/3] 写入文件...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_file = os.path.join(OUTPUT_DIR, 'valid_tiles_train_scene.txt')
    val_file   = os.path.join(OUTPUT_DIR, 'valid_tiles_val_scene.txt')

    with open(train_file, 'w', encoding='utf-8') as f:
        for t in sorted(new_train_tiles):
            f.write(t + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for t in sorted(new_val_tiles):
            f.write(t + '\n')

    print(f'  Train list → {train_file}  ({len(new_train_tiles)} tiles)')
    print(f'  Val list   → {val_file}  ({len(new_val_tiles)} tiles)')
    print('\n完成！重新切分后，请重新运行训练。')


if __name__ == '__main__':
    main()
