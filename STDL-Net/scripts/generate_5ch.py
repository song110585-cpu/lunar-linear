r"""
生成五通道影像: WAC + DEM + Slope + TPI + Profile Curvature

用法:
    python generate_5ch.py

读取 D:\Data-Lunar\shiyan\raw\{region}\ 下的 WAC + DEM
输出 D:\Data-Lunar\shiyan\data\{region}_5ch.tif
"""

import os
import gc
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter

# ===== 配置 =====
RAW_ROOT = r"D:\Data-Lunar\shiyan\raw"
OUT_DIR  = r"D:\Data-Lunar\shiyan\data"
os.makedirs(OUT_DIR, exist_ok=True)


# ===== 归一化函数 =====

def min_max_norm(array):
    """Min-Max 归一化到 [0, 1]"""
    mask = ~np.isnan(array)
    if not np.any(mask):
        return np.zeros_like(array)
    vmin, vmax = np.nanmin(array), np.nanmax(array)
    if vmax - vmin < 1e-8:
        return np.zeros_like(array)
    return (array - vmin) / (vmax - vmin)


def percentile_minmax_norm(array, low=1, high=99):
    """百分位鲁棒 Min-Max: 裁剪极端值后映射到 [0, 1]"""
    mask = ~np.isnan(array)
    if not np.any(mask):
        return np.zeros_like(array)
    p_low  = np.nanpercentile(array, low)
    p_high = np.nanpercentile(array, high)
    if p_high - p_low < 1e-8:
        return np.zeros_like(array)
    clipped = np.clip(array, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low)


# ===== 地形因子计算 =====

def nan_uniform_mean(arr, size):
    """NaN 安全的局部均值: 忽略 NaN, 只对有效值求平均"""
    filled = np.nan_to_num(arr, nan=0.0)
    counts = (~np.isnan(arr)).astype(np.float32)
    s = uniform_filter(filled, size=size, mode='reflect')
    c = uniform_filter(counts, size=size, mode='reflect')
    result = s / np.maximum(c, 1.0)
    result[c == 0] = np.nan
    return result


def calculate_terrain_factors(dem, res_x, res_y):
    """基于物理分辨率计算 Slope, TPI, Profile Curvature"""
    print("  计算地形因子...")
    dy, dx = np.gradient(dem, res_y, res_x)

    # Slope (度)
    slope = np.sqrt(dx * dx + dy * dy)
    np.arctan(slope, out=slope)
    slope *= (180.0 / np.pi)

    # TPI (米) — DEM - 11×11 局部均值 (NaN 安全)
    mean_elev = nan_uniform_mean(dem, size=11)
    tpi = dem - mean_elev
    del mean_elev; gc.collect()

    # Profile Curvature: -(dx²·Dxx + 2·dx·dy·Dxy + dy²·Dyy) / (dx² + dy² + ε)
    dx2 = dx * dx
    dy2 = dy * dy

    dyy, _dyx = np.gradient(dy, res_y, res_x)   # _dyx == dxy, 但用另一个名字避免混淆
    del _dyx; gc.collect()
    dxy, dxx = np.gradient(dx, res_y, res_x)
    dxy = dxy.astype(np.float32)

    # 分子: dx²·dxx + 2·dx·dy·dxy + dy²·dyy
    numerator = dx2 * dxx
    del dxx; gc.collect()
    tmp = dx * dy
    tmp *= dxy
    tmp *= 2.0
    numerator += tmp
    del tmp, dxy; gc.collect()
    tmp = dy2 * dyy
    numerator += tmp
    del tmp, dyy, dx, dy; gc.collect()

    # 分母
    p = dx2 + dy2
    del dx2, dy2; gc.collect()
    p += 1e-6

    profile_curvature = -numerator / p
    del numerator, p; gc.collect()

    print("  Slope, TPI, Curvature 完成")
    return slope, tpi, profile_curvature


# ===== 文件查找 =====

def find_wac(region_dir):
    """在区域目录中查找 WAC tif（排除 DEM/shade/Shade/辅助文件）"""
    for f in sorted(os.listdir(region_dir)):
        if not f.lower().endswith('.tif'):
            continue
        base = os.path.splitext(f)[0]
        # 排除 DEM
        if base.lower().startswith(('dem', 'dem_')):
            continue
        # 排除 shade/Shade
        if base.lower().startswith(('shade', 'shade_')):
            continue
        # 排除 train- 前缀的 SHP 相关文件
        if base.lower().startswith('train-'):
            continue
        # 剩下的就是 WAC
        full = os.path.join(region_dir, f)
        # 确认只有一个波段的光学影像
        with rasterio.open(full) as src:
            if src.count == 1 and src.width > 100:
                return f
    return None


def find_dem(region_dir):
    """在区域目录中查找 DEM tif（支持 Dem/DEM/dem 前缀，- 或 _ 分隔符）"""
    for f in sorted(os.listdir(region_dir)):
        if not f.lower().endswith('.tif'):
            continue
        base = os.path.splitext(f)[0]
        # DEM 开头
        if base.lower().startswith(('dem', 'dem-')):
            full = os.path.join(region_dir, f)
            with rasterio.open(full) as src:
                if src.count == 1:
                    return f
    return None


# ===== 主流程 =====

def generate_one_region(region_name, region_dir):
    r"""
    对一个区域生成 5ch tif。
    参数:
        region_name: 区域名, e.g. "Marius Hills"
        region_dir: 该区域的原始数据目录
    输出:
        D:\Data-Lunar\shiyan\data\{region_name}_5ch.tif
    """

    wac_file = find_wac(region_dir)
    dem_file = find_dem(region_dir)

    if wac_file is None:
        print(f"  [ERROR] 找不到 WAC 文件")
        return False
    if dem_file is None:
        print(f"  [ERROR] 找不到 DEM 文件")
        return False

    wac_path = os.path.join(region_dir, wac_file)
    dem_path = os.path.join(region_dir, dem_file)
    out_path = os.path.join(OUT_DIR, f"{region_name}_5ch.tif")

    print(f"  WAC: {wac_file}")
    print(f"  DEM: {dem_file}")
    print(f"  输出: {out_path}")

    # ---- 1. 读 DEM，建立空间基准 ----
    print("  读取 DEM...")
    with rasterio.open(dem_path) as src_dem:
        dem_data = src_dem.read(1).astype(np.float32)
        dem_nodata = src_dem.nodata
        ref_meta   = src_dem.meta.copy()
        ref_trans  = src_dem.transform
        ref_crs    = src_dem.crs
        res_x, res_y = src_dem.res
        ref_h, ref_w = ref_meta['height'], ref_meta['width']
        print(f"    分辨率: {res_x:.1f}m, 尺寸: {ref_w}×{ref_h}")

    # 把 nodata 转 NaN（关键：-32768 不能被百分位归一化当有效值算）
    if dem_nodata is not None:
        dem_data[dem_data == dem_nodata] = np.nan
        print(f"    nodata={dem_nodata} → NaN, 有效像元={np.sum(~np.isnan(dem_data))}/{dem_data.size}")

    # ---- 2. 读 WAC，重采样到 DEM 网格 ----
    print("  读取 WAC 并重采样...")
    with rasterio.open(wac_path) as src_wac:
        wac_aligned = np.full((ref_h, ref_w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src_wac, 1),
            destination=wac_aligned,
            src_transform=src_wac.transform,
            src_crs=src_wac.crs,
            dst_transform=ref_trans,
            dst_crs=ref_crs,
            src_nodata=src_wac.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    print("    WAC 重采样完成")

    # ---- 3. 计算地形因子 ----
    slope, tpi, curvature = calculate_terrain_factors(dem_data, res_x, res_y)

    # ---- 4. 各通道归一化 ----
    print("  归一化各通道...")

    # WAC: Min-Max, NaN→0
    wac_mask = np.isnan(wac_aligned)
    wac_norm = min_max_norm(wac_aligned)
    wac_norm[wac_mask] = 0.0
    del wac_aligned, wac_mask; gc.collect()

    # DEM: 百分位 Min-Max
    dem_norm = percentile_minmax_norm(dem_data)
    del dem_data; gc.collect()

    # Slope: 百分位 Min-Max
    slope_norm = percentile_minmax_norm(slope)
    del slope; gc.collect()

    # TPI: 百分位 Min-Max
    tpi_norm = percentile_minmax_norm(tpi)
    del tpi; gc.collect()

    # Curvature: 百分位 Min-Max
    curv_norm = percentile_minmax_norm(curvature)
    del curvature; gc.collect()

    # ---- 5. 逐通道写入 5-band GeoTIFF ----
    print("  写入 5 通道...")
    meta_out = ref_meta.copy()
    meta_out.update(count=5, dtype='float32', nodata=None)

    with rasterio.open(out_path, 'w', **meta_out) as dst:
        dst.write(wac_norm.astype(np.float32), 1)
        print("    ch1 WAC [OK]")
        del wac_norm; gc.collect()

        dst.write(dem_norm.astype(np.float32), 2)
        print("    ch2 DEM [OK]")
        del dem_norm; gc.collect()

        dst.write(slope_norm.astype(np.float32), 3)
        print("    ch3 Slope [OK]")
        del slope_norm; gc.collect()

        dst.write(tpi_norm.astype(np.float32), 4)
        print("    ch4 TPI [OK]")
        del tpi_norm; gc.collect()

        dst.write(curv_norm.astype(np.float32), 5)
        print("    ch5 Curvature [OK]")
        del curv_norm; gc.collect()

    print(f"  [OK] 完成: {out_path}")
    return True


# ===== 批量处理 =====

def main():
    regions = sorted(os.listdir(RAW_ROOT))
    print(f"待处理区域: {regions}\n")

    for region in regions:
        region_dir = os.path.join(RAW_ROOT, region)
        if not os.path.isdir(region_dir):
            continue

        out_path = os.path.join(OUT_DIR, f"{region}_5ch.tif")
        if os.path.exists(out_path):
            print(f"[{region}] 5ch 已存在，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"[{region}]")
        print(f"{'='*60}")
        try:
            generate_one_region(region, region_dir)
        except Exception as e:
            print(f"  [ERROR] 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n==== 全部完成 ====")


if __name__ == "__main__":
    main()
