"""
将 Shade (uint8, 59m) 归一化为 1-channel float32 GeoTIFF
输出到 D:\Data-Lunar\shiyan\data_shade\
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ===== 配置 =====
RAW_DIR  = r"E:\月球_dataset\WR\raw"
OUT_DIR  = r"E:\月球_dataset\WR\raw\data_shade"

# 区域 → shade 文件名
SHADE_FILES = {
    "Mare Imbrium":          "shade_Mare Imbrium.tif",
    "Mare Serenitatis":      "Shade-Mare Serenitatis.tif",
    "Mare Tranquillitatis":  "shade_Mare Tranquillitatis.tif",
    "Marius Hills":          "Shade-Marius Hills.tif",
    "Oceanus Procellarum":   "shade_Oceanus Procellarum-NW.tif",
}

os.makedirs(OUT_DIR, exist_ok=True)


def min_max_norm(data):
    """Min-Max 归一化到 [0,1]，忽略 NaN"""
    mask = np.isnan(data)
    valid = data[~mask]
    if len(valid) == 0:
        return np.zeros_like(data)
    vmin, vmax = valid.min(), valid.max()
    if vmax == vmin:
        result = np.zeros_like(data)
    else:
        result = (data - vmin) / (vmax - vmin)
    result[mask] = 0.0
    return result.astype(np.float32)


for region_name, shade_file in SHADE_FILES.items():
    shade_path = os.path.join(RAW_DIR, region_name, shade_file)
    out_path   = os.path.join(OUT_DIR, f"{region_name}_1ch.tif")

    if not os.path.exists(shade_path):
        print(f"[SKIP] {region_name}: {shade_file} 不存在")
        continue

    print(f"[{region_name}]")
    print(f"  输入: {shade_path}")

    with rasterio.open(shade_path) as src:
        shade_data = src.read(1).astype(np.float32)
        profile    = src.profile.copy()

    # uint8 0-255 → float32 [0, 1]
    shade_norm = shade_data / 255.0
    del shade_data

    # 写 1 通道 GeoTIFF (不用压缩: float32 + LZW 极慢, 且压缩率低)
    profile_out = profile.copy()
    profile_out.update(dtype='float32', count=1, nodata=None,
                       compress=None, tiled=False)

    with rasterio.open(out_path, 'w', **profile_out) as dst:
        dst.write(shade_norm, 1)

    print(f"  输出: {out_path}")
    print(f"  范围: [{shade_norm.min():.4f}, {shade_norm.max():.4f}], "
          f"尺寸: {profile['width']}×{profile['height']}")
    del shade_norm

print("\nDone!")
