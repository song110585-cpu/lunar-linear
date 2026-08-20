"""为 v6 新增研究区生成裁剪后的五通道 GeoTIFF。

通道顺序固定为 WAC / DEM / Slope / TPI / Profile Curvature。
生成雨海两个子区、静海子区、汽海子区、危海完整外部测试区，
并从原始 WAC/DEM 重建阿利斯塔与马略山，修复历史5ch的零填充倾斜外框。

安全约束：
  - 不修改源数据；
  - 输出已存在时默认跳过；
  - 雨海两个 polygon 分开生成，降低内存峰值。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy.ndimage import uniform_filter
from shapely.geometry import box


DEFAULT_RAW_ROOT = Path(r"E:\月球_dataset\Data\研究区\raw")
DEFAULT_BOUNDARY_ROOT = Path(r"E:\月球_dataset\dataset\dataset_v6_source\boundaries")
DEFAULT_REFERENCE_5CH_ROOT = Path(r"E:\月球_dataset\Data\研究区\image")
DEFAULT_OUT_ROOT = Path(r"E:\月球_dataset\dataset\dataset_v6_work\region_5ch")
NODATA_OUT = np.float32(-3.4028235e38)
HALO_PIXELS = 8


REGIONS = {
    "aristarchus": {
        "raw_dir": "Aristarchus",
        "wac": "Aristarchus.tif",
        "dem": "Dem-Aristarchus.tif",
        "reference_5ch": "Aristarchus_5ch.tif",
        "external": False,
        "split_features": False,
    },
    "marius_hills": {
        "raw_dir": "Marius Hills",
        "wac": "Marius Hills.tif",
        "dem": "Dem-Marius Hills.tif",
        "reference_5ch": "Marius Hills_5ch.tif",
        "external": False,
        "split_features": False,
    },
    "mare_imbrium": {
        "raw_dir": "Mare Imbrium",
        "wac": "Mare Imbrium.tif",
        "dem": "DEM_Mare Imbrium.tif",
        "extent": ("selected_subarea", "mare_imbrium_subarea_01.shp"),
        "external": False,
        "split_features": True,
    },
    "mare_tranquillitatis": {
        "raw_dir": "Mare Tranquillitatis",
        "wac": "Mare Tranquillitatis.tif",
        "dem": "DEM_Mare Tranquillitatis.tif",
        "extent": ("selected_subarea", "mare_tranquillitatis_subarea_01.shp"),
        "external": False,
        "split_features": False,
    },
    "mare_vaporum": {
        "raw_dir": "Mare vaporum",
        "wac": "Mare vaporum.tif",
        "dem": "DEM_Mare vaporum.tif",
        "extent": ("selected_subarea", "mare_vaporum_subarea_01.shp"),
        "external": False,
        "split_features": False,
    },
    "mare_crisium": {
        "raw_dir": "Mare Crisium",
        # 该目录内文件历史误命名为 Tranquillitatis，但空间范围确为 Crisium。
        "wac": "Mare Tranquillitatis.tif",
        "dem": "DEM_Mare Tranquillitatis.tif",
        "extent": ("region_boundary", "mare_crisium_boundary.shp"),
        "external": True,
        "split_features": False,
    },
}


def set_runtime_environment() -> None:
    os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "YES")
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")


def moon_geographic_crs(boundary_root: Path) -> CRS:
    reference = boundary_root / "region_boundary" / "mare_imbrium_boundary.shp"
    return CRS.from_user_input(gpd.read_file(reference, rows=1).crs)


def read_lunar_vector(path: Path, moon_geog: CRS) -> gpd.GeoDataFrame:
    """读取月球矢量；修复被导出为 Earth Web Mercator 的边界。"""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"空矢量: {path}")
    crs = CRS.from_user_input(gdf.crs)
    if crs.to_epsg() == 3857:
        # 坐标数值来自 Web Mercator；先还原 lon/lat，再改为月球地理 CRS。
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.set_crs(moon_geog, allow_override=True)
    return gdf


def clamp_window(window: Window, width: int, height: int, halo: int) -> Window:
    col0 = max(0, int(np.floor(window.col_off)) - halo)
    row0 = max(0, int(np.floor(window.row_off)) - halo)
    col1 = min(width, int(np.ceil(window.col_off + window.width)) + halo)
    row1 = min(height, int(np.ceil(window.row_off + window.height)) + halo)
    return Window(col0, row0, col1 - col0, row1 - row0)


def normalize(array: np.ndarray, valid: np.ndarray, percentile: bool) -> np.ndarray:
    vals = array[valid & np.isfinite(array)]
    if vals.size == 0:
        return np.full(array.shape, NODATA_OUT, dtype=np.float32)
    if percentile:
        low, high = np.percentile(vals, [1.0, 99.0])
    else:
        low, high = float(vals.min()), float(vals.max())
    if float(high - low) < 1e-8:
        result = np.zeros(array.shape, dtype=np.float32)
    else:
        result = np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)
    result[~valid | ~np.isfinite(result)] = NODATA_OUT
    return result


def nan_local_mean(array: np.ndarray, valid: np.ndarray, size: int = 11) -> np.ndarray:
    filled = np.where(valid, array, 0.0).astype(np.float32, copy=False)
    fraction = uniform_filter(valid.astype(np.float32), size=size, mode="reflect")
    mean = uniform_filter(filled, size=size, mode="reflect")
    np.divide(mean, np.maximum(fraction, 1e-6), out=mean)
    mean[fraction <= 0] = np.nan
    return mean


def output_profile(dem_src: rasterio.DatasetReader, transform, width: int, height: int) -> dict:
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 5,
        "dtype": "float32",
        "crs": dem_src.crs,
        "transform": transform,
        "nodata": float(NODATA_OUT),
        "compress": "deflate",
        "predictor": 3,
        "zlevel": 4,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    }


def generate_asset(
    region_id: str,
    asset_id: str,
    geometry,
    config: dict,
    raw_root: Path,
    out_root: Path,
) -> dict:
    out_path = out_root / f"{asset_id}_5ch.tif"
    if out_path.exists():
        print(f"[SKIP] 已存在: {out_path}")
        with rasterio.open(out_path) as src:
            return {
                "region_id": region_id,
                "asset_id": asset_id,
                "path": str(out_path),
                "width": src.width,
                "height": src.height,
                "external": bool(config["external"]),
                "status": "existing",
            }

    region_dir = raw_root / config["raw_dir"]
    dem_path = region_dir / config["dem"]
    wac_path = region_dir / config["wac"]
    if not dem_path.is_file() or not wac_path.is_file():
        raise FileNotFoundError(f"缺少 WAC/DEM: {wac_path} / {dem_path}")

    print(f"\n[{asset_id}] DEM={dem_path.name}, WAC={wac_path.name}")
    with rasterio.open(dem_path) as dem_src:
        # geometry 已由调用者投影到 DEM CRS。
        bounds = geometry.bounds
        raw_window = from_bounds(*bounds, transform=dem_src.transform)
        window = clamp_window(raw_window, dem_src.width, dem_src.height, HALO_PIXELS)
        transform = window_transform(window, dem_src.transform)
        height, width = int(window.height), int(window.width)
        dem_ma = dem_src.read(1, window=window, masked=True).astype(np.float32)
        dem = dem_ma.filled(np.nan).astype(np.float32, copy=False)
        valid = ~np.ma.getmaskarray(dem_ma) & np.isfinite(dem)
        del dem_ma

        inside = geometry_mask(
            [geometry.__geo_interface__],
            out_shape=(height, width),
            transform=transform,
            invert=True,
            all_touched=False,
        )
        valid &= inside
        if not np.any(valid):
            raise ValueError(f"{asset_id} 边界与 DEM 无有效交集")

        profile = output_profile(dem_src, transform, width, height)

    print(f"  crop={width}x{height}, valid={valid.mean():.3f}")
    out_root.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        # WAC -> DEM crop grid
        wac = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(wac_path) as wac_src:
            reproject(
                source=rasterio.band(wac_src, 1),
                destination=wac,
                src_transform=wac_src.transform,
                src_crs=wac_src.crs,
                src_nodata=wac_src.nodata,
                dst_transform=transform,
                dst_crs=profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        valid_all = valid & np.isfinite(wac)
        dst.write(normalize(wac, valid_all, percentile=False), 1)
        del wac
        gc.collect()

        dst.write(normalize(dem, valid_all, percentile=True), 2)

        # TPI before gradients, keeping peak memory lower.
        local_mean = nan_local_mean(dem, valid_all, size=11)
        tpi = dem - local_mean
        del local_mean
        dst.write(normalize(tpi, valid_all, percentile=True), 4)
        del tpi
        gc.collect()

        res_x = abs(float(transform.a))
        res_y = abs(float(transform.e))
        dy, dx = np.gradient(dem, res_y, res_x)
        dx = dx.astype(np.float32, copy=False)
        dy = dy.astype(np.float32, copy=False)

        slope = np.hypot(dx, dy).astype(np.float32, copy=False)
        np.arctan(slope, out=slope)
        slope *= np.float32(180.0 / np.pi)
        dst.write(normalize(slope, valid_all, percentile=True), 3)
        del slope
        gc.collect()

        dyy, _ = np.gradient(dy, res_y, res_x)
        dxy, dxx = np.gradient(dx, res_y, res_x)
        del _
        numerator = dx * dx
        numerator *= dxx
        tmp = dx * dy
        tmp *= dxy
        tmp *= np.float32(2.0)
        numerator += tmp
        del tmp, dxy, dxx
        tmp = dy * dy
        denominator = tmp.copy()
        tmp *= dyy
        numerator += tmp
        del tmp, dyy
        denominator += dx * dx
        denominator += np.float32(1e-6)
        numerator *= np.float32(-1.0)
        np.divide(numerator, denominator, out=numerator)
        dst.write(normalize(numerator, valid_all, percentile=True), 5)

    del dem, valid, valid_all, inside, dx, dy, numerator, denominator
    gc.collect()
    print(f"  [OK] {out_path}")
    return {
        "region_id": region_id,
        "asset_id": asset_id,
        "path": str(out_path),
        "width": width,
        "height": height,
        "external": bool(config["external"]),
        "status": "generated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--boundary-root", type=Path, default=DEFAULT_BOUNDARY_ROOT)
    parser.add_argument("--reference-5ch-root", type=Path, default=DEFAULT_REFERENCE_5CH_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--regions", nargs="*", choices=sorted(REGIONS), default=sorted(REGIONS))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    set_runtime_environment()

    moon_geog = moon_geographic_crs(args.boundary_root)
    manifest = []
    for region_id in args.regions:
        config = REGIONS[region_id]
        raw_dir = args.raw_root / config["raw_dir"]
        dem_path = raw_dir / config["dem"]
        if "reference_5ch" in config:
            extent_path = args.reference_5ch_root / config["reference_5ch"]
            if not extent_path.is_file():
                raise FileNotFoundError(f"缺少历史范围参考影像: {extent_path}")
            with rasterio.open(extent_path) as reference_src, rasterio.open(dem_path) as dem_src:
                reference_geom = gpd.GeoSeries(
                    [box(*reference_src.bounds)], crs=reference_src.crs
                ).to_crs(dem_src.crs).iloc[0]
            geometries = [reference_geom]
        else:
            folder, filename = config["extent"]
            extent_path = args.boundary_root / folder / filename
            gdf = read_lunar_vector(extent_path, moon_geog)
            with rasterio.open(dem_path) as dem_src:
                gdf = gdf.to_crs(dem_src.crs)
            if config["split_features"]:
                geometries = list(gdf.geometry)
            else:
                geometries = [gdf.geometry.union_all()]

        print(f"[{region_id}] extent_features={len(geometries)}, source={extent_path}")
        if args.audit_only:
            continue
        for index, geometry in enumerate(geometries, start=1):
            asset_id = f"{region_id}_{index:02d}" if len(geometries) > 1 else region_id
            manifest.append(
                generate_asset(region_id, asset_id, geometry, config, args.raw_root, args.out_root)
            )

    if not args.audit_only:
        manifest_path = args.out_root.parent / "generated_5ch_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
