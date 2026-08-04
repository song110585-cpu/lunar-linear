"""
澄海 WR 面要素 → 二分类训练数据集
SHP (EPSG:3857) + 5ch GeoTIFF (Moon Equirectangular)
→ 栅格化 → 512×512 tile → 8:1:1 split → Train/Val/Test
"""
import os, sys, random
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from tqdm import tqdm

# ===== 配置 =====
SHP_PATH = r"D:\Data-Lunar\shiyan\shp\Mare Serenitatis.shp"
TIF_PATH = r"D:\Data-Lunar\shiyan\data\Mare Serenitatis_5ch.tif"
OUT_DIR = r"D:\Data-Lunar\shiyan\wr_dataset"
TILE_SIZE = 512

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# ===== 1. 加载数据 =====
print("[1] 加载数据...")
gdf = gpd.read_file(SHP_PATH)
print(f"    WR 面要素: {len(gdf)} 条")
print(f"    SHP CRS: {gdf.crs}")

with rasterio.open(TIF_PATH) as src:
    tif_crs = src.crs
    tif_transform = src.transform
    tif_h = src.height
    tif_w = src.width
    tif_profile = src.profile
    print(f"    TIF CRS: {tif_crs}")
    print(f"    TIF 尺寸: {tif_w}×{tif_h} px, {src.count} 通道")

# ===== 2. 对齐 CRS =====
# SHP 在软件中常被标为 EPSG:3857 但坐标实际已是月球投影
# 直接强制设为 TIF 的 CRS 避免投影变换失败
gdf = gdf.set_crs(tif_crs, allow_override=True)
print(f"[2] SHP CRS 已对齐: {tif_crs}")

# ===== 3. 栅格化 WR 为二值掩膜 =====
print("[3] 栅格化 WR 面要素...")
mask = features.rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(tif_h, tif_w),
    transform=tif_transform,
    fill=0,
    dtype=np.uint8
)
n_wr = mask.sum()
print(f"    WR 像素: {n_wr} / {tif_h * tif_w} ({100 * n_wr / (tif_h * tif_w):.2f}%)")

# ===== 4. 切 512×512 tile =====
print("[4] 切 tile...")
os.makedirs(os.path.join(OUT_DIR, 'train', 'image'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'train', 'mask'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'val', 'image'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'val', 'mask'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'test', 'image'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'test', 'mask'), exist_ok=True)

n_row = tif_h // TILE_SIZE
n_col = tif_w // TILE_SIZE
all_tiles = []

# 读所有通道
with rasterio.open(TIF_PATH) as src:
    img_data = src.read()  # (5, H, W)

for r in range(n_row):
    for c in range(n_col):
        y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
        x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
        tile_img = img_data[:, y0:y1, x0:x1]
        tile_mask = mask[y0:y1, x0:x1]

        # 跳过无效 tile（全 NaN 或全 nodata）
        if np.isnan(tile_img).sum() > 0.3 * TILE_SIZE * TILE_SIZE * 5:
            continue

        all_tiles.append((tile_img, tile_mask, f"serenitatis_r{r:04d}_c{c:04d}"))

print(f"    有效 tile: {len(all_tiles)}")

# ===== 5. 8:1:1 随机划分 =====
random.seed(42)
random.shuffle(all_tiles)
n = len(all_tiles)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    'train': all_tiles[:n_train],
    'val': all_tiles[n_train:n_train + n_val],
    'test': all_tiles[n_train + n_val:],
}
print(f"    Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# ===== 6. 写 GeoTIFF =====
profile_img = tif_profile.copy()
profile_img.update(driver='GTiff', width=TILE_SIZE, height=TILE_SIZE, count=5, dtype='float32')
profile_mask = dict(driver='GTiff', width=TILE_SIZE, height=TILE_SIZE, count=1, dtype='uint8',
                    crs=tif_crs, transform=tif_transform)

for split_name, tiles in splits.items():
    img_dir = os.path.join(OUT_DIR, split_name, 'image')
    mask_dir = os.path.join(OUT_DIR, split_name, 'mask')
    wr_count = 0
    for tile_img, tile_mask, basename in tqdm(tiles, desc=split_name):
        # 写 image
        with rasterio.open(os.path.join(img_dir, f'{basename}.tif'), 'w', **profile_img) as dst:
            dst.write(tile_img.astype(np.float32))
        # 写 mask
        with rasterio.open(os.path.join(mask_dir, f'{basename}.tif'), 'w', **profile_mask) as dst:
            dst.write(tile_mask.astype(np.uint8), 1)
        if tile_mask.sum() > 0:
            wr_count += 1
    print(f"    {split_name}: {len(tiles)} tiles, {wr_count} 含 WR")

print(f"\nDone! 输出: {OUT_DIR}")
