"""
从 Lunar_LRO_LOLAKaguya_DEMmerge (512ppd, 59m) 生成 6 通道 GeoTIFF
通道: [Hillshade, DEM, Slope, TPI, Profile Curvature, Aspect(方差滤波)]

使用各区域已有 Shade 文件确定空间范围, 窗口读取全局 DEM
"""
import os, gc
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter

# ===== 配置 =====
DEM_GLOBAL = r"E:\月球_dataset\Data\map date\Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif"
RAW_DIR    = r"E:\月球_dataset\WR\raw"
OUT_DIR    = r"E:\月球_dataset\WR\data_6ch_59m"

# 各区域的 Shade 文件名 (用于确定空间范围)
SHADE_FILES = {
    "Mare Imbrium":          "shade_Mare Imbrium.tif",
    "Mare Serenitatis":      "Shade-Mare Serenitatis.tif",
    "Mare Tranquillitatis":  "shade_Mare Tranquillitatis.tif",
    "Marius Hills":          "Shade-Marius Hills.tif",
    "Oceanus Procellarum":   "shade_Oceanus Procellarum-NW.tif",
}

os.makedirs(OUT_DIR, exist_ok=True)


def min_max_norm(data):
    """Min-Max 归一化到 [0,1], NaN→0"""
    mask = np.isnan(data)
    valid = data[~mask]
    if len(valid) == 0:
        return np.zeros_like(data)
    vmin, vmax = np.nanpercentile(valid, 1), np.nanpercentile(valid, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-8
    result = (data - vmin) / (vmax - vmin)
    result = np.clip(result, 0, 1)
    result[mask] = 0.0
    return result.astype(np.float32)


def calc_hillshade(dem, res_x, res_y, azimuth=315, altitude=45):
    """从 DEM 计算山体阴影"""
    dy, dx = np.gradient(dem, res_y, res_x)
    az_rad = np.deg2rad(360 - azimuth + 90)
    alt_rad = np.deg2rad(altitude)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(dy, -dx)
    shade = (np.cos(alt_rad) * np.cos(slope_rad)
             + np.sin(alt_rad) * np.sin(slope_rad)
             * np.cos(az_rad - aspect_rad))
    shade = np.clip(shade, 0, 1)
    shade[np.isnan(dem)] = 0.0
    return shade.astype(np.float32)


def calc_slope(dem, res_x, res_y):
    """坡度 (度)"""
    dy, dx = np.gradient(dem, res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[np.isnan(dem)] = np.nan
    return slope


def calc_tpi(dem, size=5):
    """Topographic Position Index: 中心点 - 邻域均值"""
    from scipy.ndimage import uniform_filter
    kernel = np.ones((size, size)) / (size * size)
    kernel[size//2, size//2] = 0
    kernel /= kernel.sum()
    mean_surrounding = uniform_filter(dem.astype(np.float64), size=size)
    # 近似: local mean - 3x3 mean
    local_mean = uniform_filter(dem.astype(np.float64), size=3)
    tpi = local_mean - mean_surrounding
    tpi[np.isnan(dem)] = np.nan
    return tpi


def calc_curvature(dem, res_x, res_y):
    """剖面曲率 (Profile Curvature)"""
    dy, dx = np.gradient(dem, res_y, res_x)
    dyy, _dyx = np.gradient(dy, res_y, res_x)
    dxy, dxx = np.gradient(dx, res_y, res_x)
    p = dx**2 + dy**2
    num = dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy
    denom = p * np.sqrt(p)
    curv = np.full_like(dem, np.nan)
    mask = denom > 1e-8
    curv[mask] = num[mask] / denom[mask]
    curv[np.isnan(dem)] = np.nan
    return curv


def calc_aspect_variance(dem, res_x, res_y):
    """坡向 + 方差滤波 (Lu et al. 2025 方法)"""
    dy, dx = np.gradient(dem, res_y, res_x)
    # 坡向 0-360 度
    aspect = np.degrees(np.arctan2(dy, -dx))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect[np.isnan(dem)] = np.nan

    # 3x3 方差滤波
    valid = ~np.isnan(aspect)
    aspect_filled = np.where(valid, aspect, 0)
    mean = uniform_filter(aspect_filled, size=3)
    sq_mean = uniform_filter(aspect_filled**2, size=3)
    variance = sq_mean - mean**2
    variance[~valid] = np.nan
    return variance


# ===== 主流程 =====
for region_name, shade_file in SHADE_FILES.items():
    shade_path = os.path.join(RAW_DIR, region_name, shade_file)
    out_path   = os.path.join(OUT_DIR, f"{region_name}_6ch.tif")

    if not os.path.exists(shade_path):
        print(f"[SKIP] {region_name}: Shade 不存在: {shade_path}")
        continue

    print(f"\n[{region_name}]")
    print(f"  模板: {shade_path}")

    # 1. 读 Shade 获取空间范围
    with rasterio.open(shade_path) as src_shade:
        bounds    = src_shade.bounds
        shade_crs = src_shade.crs
        ref_w     = src_shade.width
        ref_h     = src_shade.height
        ref_trans = src_shade.transform
        print(f"    范围: {bounds}, 尺寸: {ref_w}×{ref_h}")

    # 2. 从全局 DEM 窗口读取
    print(f"  从全局 DEM 窗口读取...")
    with rasterio.open(DEM_GLOBAL) as src_dem:
        # 对齐 CRS
        if src_dem.crs != shade_crs:
            print(f"    DEM CRS: {src_dem.crs} → 重投影到 {shade_crs}")
            # 窗口读取 + 重投影
            dem_data = np.full((ref_h, ref_w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src_dem, 1),
                destination=dem_data,
                src_crs=src_dem.crs,
                dst_crs=shade_crs,
                dst_transform=ref_trans,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        else:
            # CRS 一致，直接窗口读
            window = from_bounds(*bounds, src_dem.transform)
            win_h = int(window.height)
            win_w = int(window.width)
            dem_data = src_dem.read(1, window=window).astype(np.float32)
            # 如果窗口尺寸不完全匹配 shade，重采样
            if win_h != ref_h or win_w != ref_w:
                dem_aligned = np.full((ref_h, ref_w), np.nan, dtype=np.float32)
                reproject(
                    source=dem_data,
                    destination=dem_aligned,
                    src_transform=src_dem.window_transform(window),
                    src_crs=src_dem.crs,
                    dst_transform=ref_trans,
                    dst_crs=shade_crs,
                    resampling=Resampling.bilinear,
                )
                dem_data = dem_aligned
                del dem_aligned

    res_x, res_y = ref_trans.a, -ref_trans.e  # 注意 e 通常为负
    print(f"    DEM 读取完成, 分辨率: {res_x:.1f}m")

    # 3. 处理 nodata
    dem_nodata_val = -32768  # 常见 DEM nodata
    dem_nodata_mask = (dem_data == dem_nodata_val)
    dem_data[dem_nodata_mask] = np.nan
    valid_ratio = np.sum(~np.isnan(dem_data)) / dem_data.size
    print(f"    有效 DEM 像素: {100*valid_ratio:.1f}%")

    if valid_ratio < 0.01:
        print(f"    [SKIP] 有效 DEM 像素 <1%, 跳过")
        continue

    # 4. 计算各通道
    print(f"  计算 Hillshade...")
    hillshade = calc_hillshade(dem_data, res_x, res_y)

    print(f"  计算 Slope...")
    slope = calc_slope(dem_data, res_x, res_y)

    print(f"  计算 TPI...")
    tpi = calc_tpi(dem_data)

    print(f"  计算 Curvature...")
    curvature = calc_curvature(dem_data, res_x, res_y)

    print(f"  计算 Aspect(方差滤波)...")
    aspect_var = calc_aspect_variance(dem_data, res_x, res_y)

    # 5. 归一化
    print(f"  归一化...")
    hs_norm  = hillshade.astype(np.float32)  # 已在 0-1 范围
    dem_norm = min_max_norm(dem_data)
    sl_norm  = min_max_norm(slope)
    tp_norm  = min_max_norm(tpi)
    cv_norm  = min_max_norm(curvature)
    av_norm  = min_max_norm(aspect_var)

    del dem_data, slope, tpi, curvature, aspect_var; gc.collect()

    # 6. 写 6 通道 GeoTIFF
    print(f"  写 6 通道 GeoTIFF...")
    with rasterio.open(shade_path) as src:
        meta_out = src.meta.copy()
    meta_out.update(count=6, dtype='float32', nodata=None, compress='lzw')

    with rasterio.open(out_path, 'w', **meta_out) as dst:
        for i, (data, name) in enumerate([
            (hs_norm, "Hillshade"), (dem_norm, "DEM"),
            (sl_norm, "Slope"), (tp_norm, "TPI"),
            (cv_norm, "Curvature"), (av_norm, "AspectVar")
        ], 1):
            dst.write(data.astype(np.float32), i)
            print(f"    ch{i} {name}: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
            del data; gc.collect()

    del hs_norm, dem_norm, sl_norm, tp_norm, cv_norm, av_norm; gc.collect()
    print(f"  -> {out_path}")

print("\nDone! 6 通道 59m 数据生成完毕。")
