"""
Shade 单通道 (59m) → 二分类训练数据集
直接读原始 uint8 Shade TIF, 切 tile 时归一化为 float32 [0,1]
跳过中间 float32 TIF 步骤, 避免 LZW+float32 解压慢的问题

输出:
  E:\月球_dataset\WR\wr_dataset_shade_mixed\    ← 混合随机划分
  E:\月球_dataset\WR\wr_dataset_shade_holdout\  ← 跨区域独立划分
"""
import os, random
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from tqdm import tqdm

os.environ['SHAPE_RESTORE_SHX'] = 'YES'
os.environ['PROJ_IGNORE_CELESTIAL_BODY'] = 'YES'

# ===== 配置 =====
SHP_DIR  = r"E:\月球_dataset\WR\shp"
RAW_DIR  = r"E:\月球_dataset\WR\raw"                  # 原始 uint8 Shade TIF
OUT_MIXED   = r"E:\月球_dataset\WR\wr_dataset_shade_mixed"
OUT_HOLDOUT = r"E:\月球_dataset\WR\wr_dataset_shade_holdout"
TILE_SIZE = 512
IN_CHANNELS = 1

# SHP 文件名 → (shade 子目录, shade TIF 名, 真实源 CRS 或 None)
REGIONS = {
    "Mare Imbrium":          ("Mare Imbrium",          "shade_Mare Imbrium.tif",           None),
    "Mare Serenitatis":      ("Mare Serenitatis",      "Shade-Mare Serenitatis.tif",       "EPSG:3857"),
    "Mare Tranquillitatis":  ("Mare Tranquillitatis",  "shade_Mare Tranquillitatis.tif",   None),
    "Marius Hills":          ("Marius Hills",          "Shade-Marius Hills.tif",           "ESRI:54079"),
    "Oceanus Procellarum":   ("Oceanus Procellarum",   "shade_Oceanus Procellarum-NW.tif", None),
}

TRAIN_REGIONS   = ["Mare Imbrium", "Mare Serenitatis", "Mare Tranquillitatis", "Marius Hills"]
HOLDOUT_REGION  = "Oceanus Procellarum"

random.seed(42)


def process_region(shp_name, subdir, shade_filename, true_crs):
    """处理单个区域: SHP 栅格化 + 窗口切片.
    直接读 uint8 Shade TIF, 切 tile 时归一化到 float32 [0,1].
    """
    shp_path = os.path.join(SHP_DIR, f"{shp_name}.shp")
    tif_path = os.path.join(RAW_DIR, subdir, shade_filename)

    if not os.path.exists(shp_path):
        print(f"  [SKIP] SHP 不存在: {shp_path}")
        return shp_name, []
    if not os.path.exists(tif_path):
        print(f"  [SKIP] TIF 不存在: {tif_path}")
        return shp_name, []

    print(f"  SHP: {shp_path}")
    print(f"  TIF: {tif_path}")

    gdf = gpd.read_file(shp_path)
    print(f"  WR 面要素: {len(gdf)} 条, 文件 CRS: {gdf.crs}")

    if true_crs is not None:
        print(f"  修正 CRS: {gdf.crs} → {true_crs}")
        gdf = gdf.set_crs(true_crs, allow_override=True)

    with rasterio.open(tif_path) as src:
        tif_crs       = src.crs
        tif_transform = src.transform
        tif_h         = src.height
        tif_w         = src.width
        print(f"  TIF: {tif_w}×{tif_h} px, dtype={src.dtypes[0]}, "
              f"compress={src.compression}, tiled={src.is_tiled}, CRS: {tif_crs}")

    if gdf.crs != tif_crs:
        gdf = gdf.to_crs(tif_crs)
        print(f"  已重投影 SHP → TIF CRS")

    mask = features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(tif_h, tif_w),
        transform=tif_transform,
        fill=0,
        dtype=np.uint8,
    )
    n_wr = mask.sum()
    print(f"  WR 像素: {n_wr} / {tif_h * tif_w} ({100 * n_wr / (tif_h * tif_w):.2f}%)")

    # 窗口读取: 每次只解压一个 tile, uint8 → float32 [0,1]
    tiles = []
    n_row, n_col = tif_h // TILE_SIZE, tif_w // TILE_SIZE
    slug = shp_name.replace(' ', '_').replace('-', '_').lower()
    NORM = 255.0

    with rasterio.open(tif_path) as src:
        for r in tqdm(range(n_row), desc=f"  [{shp_name}] 切片", unit="row"):
            for c in range(n_col):
                y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
                x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
                window = rasterio.windows.Window(x0, y0, TILE_SIZE, TILE_SIZE)

                tile_img = src.read(window=window).astype(np.float32) / NORM  # (1,512,512) [0,1]
                tile_mask = mask[y0:y1, x0:x1]

                # NaN 过滤
                if np.isnan(tile_img).sum() > 0.3 * TILE_SIZE * TILE_SIZE * IN_CHANNELS:
                    continue

                # 黑边过滤: <0.004 视为无效, >50% 则丢弃
                if (tile_img < 0.004).sum() > 0.5 * TILE_SIZE * TILE_SIZE * IN_CHANNELS:
                    continue

                tiles.append((tile_img, tile_mask, f"{slug}_r{r:04d}_c{c:04d}"))

    del mask
    print(f"    有效 tile: {len(tiles)}")
    return shp_name, tiles


def create_dirs(out_dir):
    import shutil
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(out_dir, split, 'image')
        msk_dir = os.path.join(out_dir, split, 'mask')
        for d in [img_dir, msk_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)


def write_tiles(splits, out_dir, ref_tif_path):
    """写 tile: 1 通道 float32 image + uint8 mask"""
    with rasterio.open(ref_tif_path) as src:
        ref_crs = src.crs

    # 影像 profile: 1 通道 float32, 无压缩 (512×512 很小)
    profile_img = dict(driver='GTiff', width=TILE_SIZE, height=TILE_SIZE,
                       count=IN_CHANNELS, dtype='float32', crs=ref_crs)
    profile_mask = dict(driver='GTiff', width=TILE_SIZE, height=TILE_SIZE,
                        count=1, dtype='uint8', crs=ref_crs)

    for split_name, tiles in splits.items():
        if len(tiles) == 0:
            print(f"  {split_name}: 0 tiles (skip)")
            continue
        img_dir  = os.path.join(out_dir, split_name, 'image')
        mask_dir = os.path.join(out_dir, split_name, 'mask')
        wr_count = 0
        for tile_img, tile_mask, basename in tqdm(tiles, desc=f"  {split_name}"):
            with rasterio.open(os.path.join(img_dir, f'{basename}.tif'), 'w', **profile_img) as dst:
                dst.write(tile_img.astype(np.float32))
            with rasterio.open(os.path.join(mask_dir, f'{basename}.tif'), 'w', **profile_mask) as dst:
                dst.write(tile_mask.astype(np.uint8), 1)
            if tile_mask.sum() > 0:
                wr_count += 1
        print(f"  {split_name}: {len(tiles)} tiles, {wr_count} 含 WR")


def split_tiles_stratified(tiles):
    """按含 WR / 无 WR 分层, 各 50:50 随机分到 val 和 test"""
    has_wr = [t for t in tiles if t[1].sum() > 0]
    no_wr  = [t for t in tiles if t[1].sum() == 0]

    random.shuffle(has_wr)
    random.shuffle(no_wr)

    n_wr = len(has_wr) // 2
    n_nw = len(no_wr) // 2

    val  = has_wr[:n_wr] + no_wr[:n_nw]
    test = has_wr[n_wr:] + no_wr[n_nw:]
    random.shuffle(val)
    random.shuffle(test)
    return val, test


def main():
    # ===== 1. 处理所有区域 =====
    region_tiles = {}
    for shp_name, (subdir, shade_fname, true_crs) in REGIONS.items():
        print(f"\n{'='*50}")
        print(f"[{shp_name}]")
        print(f"{'='*50}")
        name, tiles = process_region(shp_name, subdir, shade_fname, true_crs)
        region_tiles[name] = tiles

    all_tiles = []
    for tiles in region_tiles.values():
        all_tiles.extend(tiles)
    print(f"\n总有效 tile: {len(all_tiles)}")

    # ref TIF 用于取 CRS (用第一个区域的)
    first_subdir, first_fname, _ = list(REGIONS.values())[0]
    ref_tif = os.path.join(RAW_DIR, first_subdir, first_fname)

    # ===== 2. Mixed: 所有区域混合 8:1:1 =====
    print(f"\n{'='*60}")
    print(">> Mixed Split: 所有区域混合随机 8:1:1")
    print(f"{'='*60}")
    create_dirs(OUT_MIXED)
    random.shuffle(all_tiles)
    n = len(all_tiles)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits_mixed = {
        'train': all_tiles[:n_train],
        'val':   all_tiles[n_train:n_train + n_val],
        'test':  all_tiles[n_train + n_val:],
    }
    print(f"  Train: {len(splits_mixed['train'])}, Val: {len(splits_mixed['val'])}, Test: {len(splits_mixed['test'])}")
    write_tiles(splits_mixed, OUT_MIXED, ref_tif)
    print(f"  -> {OUT_MIXED}")

    # ===== 3. Holdout: 东部4区训练, 风暴洋分层验证/测试 =====
    print(f"\n{'='*60}")
    print(f">> Holdout Split: Train=[{', '.join(TRAIN_REGIONS)}]")
    print(f"   Val/Test=[{HOLDOUT_REGION}] 分层 50:50")
    print(f"{'='*60}")
    create_dirs(OUT_HOLDOUT)

    train_tiles = []
    for name in TRAIN_REGIONS:
        train_tiles.extend(region_tiles.get(name, []))
    random.shuffle(train_tiles)

    proc_tiles = region_tiles.get(HOLDOUT_REGION, [])
    if proc_tiles:
        val_tiles, test_tiles = split_tiles_stratified(proc_tiles)
        val_wr  = sum(1 for t in val_tiles  if t[1].sum() > 0)
        test_wr = sum(1 for t in test_tiles if t[1].sum() > 0)
        print(f"  风暴洋分层切分: Val={len(val_tiles)}(含WR {val_wr}), Test={len(test_tiles)}(含WR {test_wr})")
    else:
        val_tiles, test_tiles = [], []

    splits_holdout = {
        'train': train_tiles,
        'val':   val_tiles,
        'test':  test_tiles,
    }
    print(f"  Train: {len(train_tiles)}, Val: {len(val_tiles)}, Test: {len(test_tiles)}")
    write_tiles(splits_holdout, OUT_HOLDOUT, ref_tif)
    print(f"  -> {OUT_HOLDOUT}")

    # ===== 4. 统计 =====
    print(f"\n{'='*60}")
    print(">> 划分对比")
    print(f"{'='*60}")
    print(f"{'':15s} {'Train':>8s} {'Val':>6s} {'Test':>6s} {'总计':>6s}")
    m = splits_mixed
    h = splits_holdout
    print(f"  {'Mixed':15s} {len(m['train']):>8d} {len(m['val']):>6d} {len(m['test']):>6d} {sum(len(v) for v in m.values()):>6d}")
    print(f"  {'Holdout':15s} {len(h['train']):>8d} {len(h['val']):>6d} {len(h['test']):>6d} {sum(len(v) for v in h.values()):>6d}")

    print(f"\n  各区域 tile 贡献:")
    for name, tiles in region_tiles.items():
        print(f"    {name}: {len(tiles)}")

    print(f"\nDone! Shade 单通道数据集已生成。")


if __name__ == "__main__":
    main()
