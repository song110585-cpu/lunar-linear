"""构建 v6 空间无重叠 8:1:1 数据集。

前置：先运行 prepare_v6_5ch.py，生成新增月海以及修复后的阿利斯塔/马略山五通道。

流程：候选 512 tile -> 1024 像素空间组 -> 类别阳性 tile/研究区平衡搜索
-> Train/Val/Test 写盘。三个 split 之间不共享原始像素。
危海不进入数据集，只保留完整 5ch 供最终外部测试。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio import features
from rasterio.errors import RasterioIOError
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box
from tqdm import tqdm


SOURCE_ROOT = Path(r"E:\月球_dataset\dataset\dataset_v6_source")
OLD_5CH_ROOT = Path(r"E:\月球_dataset\Data\研究区\image")
WORK_ROOT = Path(r"E:\月球_dataset\dataset\dataset_v6_work")
DEFAULT_OUTPUT = Path(r"E:\月球_dataset\dataset\dataset_v6_spatial811_g1024_fixed")

TILE_SIZE = 512
GROUP_SIZE = 1024
MIN_VALID_FRACTION = 0.95
MIN_EXTENT_FRACTION = 0.95
MIN_WAC_NONZERO_FRACTION = 0.95
CLASS_NAMES = ["background", "wrinkle_ridge", "rille", "fault", "graben"]


@dataclass(frozen=True)
class Asset:
    asset_id: str
    region_id: str
    image: Path
    annotation: Path
    extent: Path | None = None
    extent_from_annotation: bool = False


def standard_assets(source_root: Path, old_5ch_root: Path, work_root: Path) -> list[Asset]:
    ann = source_root / "annotations"
    bnd = source_root / "boundaries"
    new = work_root / "region_5ch"
    assets = [
        Asset("aristarchus", "aristarchus", new / "aristarchus_5ch.tif", ann / "Aristarchus.shp"),
        Asset(
            "mare_serenitatis", "mare_serenitatis",
            old_5ch_root / "Mare Serenitatis_5ch.tif", ann / "Mare Serenitatis.shp",
            bnd / "region_boundary" / "mare_serenitatis_boundary.shp",
        ),
        Asset("marius_hills", "marius_hills", new / "marius_hills_5ch.tif", ann / "Marius Hills.shp"),
        Asset(
            "orientale_basin", "orientale_basin",
            old_5ch_root / "Orientale Basin_5ch.tif", ann / "Orientale Basin.shp",
            bnd / "selected_subarea" / "orientale_basin_subarea_01.shp",
        ),
        Asset("mare_imbrium_01", "mare_imbrium", new / "mare_imbrium_01_5ch.tif", ann / "Mare Imbrium.shp"),
        Asset("mare_imbrium_02", "mare_imbrium", new / "mare_imbrium_02_5ch.tif", ann / "Mare Imbrium.shp"),
        Asset(
            "mare_tranquillitatis", "mare_tranquillitatis",
            new / "mare_tranquillitatis_5ch.tif", ann / "Mare Tranquillitatis.shp",
        ),
        Asset("mare_vaporum", "mare_vaporum", new / "mare_vaporum_5ch.tif", ann / "Mare vaporum.shp"),
    ]
    for index in range(1, 20):
        name = f"Catena-{index}"
        assets.append(
            Asset(
                f"catena_{index:02d}", "catena",
                old_5ch_root / f"{name}_5ch.tif", ann / f"{name}.shp",
                extent_from_annotation=True,
            )
        )
    return assets


def configure_environment() -> None:
    os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "YES")
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")


def moon_geographic_crs(source_root: Path) -> CRS:
    ref = source_root / "boundaries" / "region_boundary" / "mare_imbrium_boundary.shp"
    return CRS.from_user_input(gpd.read_file(ref, rows=1).crs)


def read_lunar_vector(path: Path, moon_geog: CRS) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"空 SHP: {path}")
    crs = CRS.from_user_input(gdf.crs)
    if crs.to_epsg() == 3857:
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.set_crs(moon_geog, allow_override=True)
    return gdf


def valid_annotations(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "class" not in gdf.columns:
        raise KeyError("标注 SHP 缺少 class 字段")
    cls = np.array([int(float(v)) if str(v).lower() not in {"nan", "none", ""} else -1 for v in gdf["class"]])
    keep = (cls >= 1) & (cls <= 4) & gdf.geometry.notna().to_numpy()
    result = gdf.loc[keep, [gdf.geometry.name]].copy()
    result["class_id"] = cls[keep].astype(np.uint8)
    return result


def candidate_range(src: rasterio.DatasetReader, extent_geom) -> tuple[range, range]:
    if extent_geom is None:
        col0 = row0 = 0
        col1, row1 = src.width, src.height
    else:
        w = from_bounds(*extent_geom.bounds, transform=src.transform)
        col0 = max(0, int(np.floor(w.col_off)))
        row0 = max(0, int(np.floor(w.row_off)))
        col1 = min(src.width, int(np.ceil(w.col_off + w.width)))
        row1 = min(src.height, int(np.ceil(w.row_off + w.height)))
    first_col = ((col0 + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    first_row = ((row0 + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    return range(first_row, row1 - TILE_SIZE + 1, TILE_SIZE), range(first_col, col1 - TILE_SIZE + 1, TILE_SIZE)


def rasterize_tile(annotations: gpd.GeoDataFrame, transform) -> np.ndarray:
    if annotations.empty:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    west, south, east, north = rasterio.transform.array_bounds(TILE_SIZE, TILE_SIZE, transform)
    indexes = list(annotations.sindex.intersection((west, south, east, north)))
    if not indexes:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    subset = annotations.iloc[indexes].sort_values("class_id")
    shapes = [(geom, int(cls)) for geom, cls in zip(subset.geometry, subset["class_id"]) if geom is not None]
    return features.rasterize(
        shapes,
        out_shape=(TILE_SIZE, TILE_SIZE),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )


def primary_coverage_union(assets: list[Asset], source_root: Path, moon_geog: CRS):
    """构建主研究区覆盖范围，用于从 Catena 困难背景中剔除地理重复。"""
    geometries = []
    for asset in assets:
        if asset.region_id == "catena":
            continue
        with rasterio.open(asset.image) as src:
            if asset.extent is not None:
                geom = read_lunar_vector(asset.extent, moon_geog).to_crs(moon_geog).geometry.union_all()
            else:
                geom = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).to_crs(moon_geog).iloc[0]
            geometries.append(geom)
    return gpd.GeoSeries(geometries, crs=moon_geog).union_all()


def enumerate_candidates(assets: list[Asset], source_root: Path) -> list[dict]:
    moon_geog = moon_geographic_crs(source_root)
    primary_union = primary_coverage_union(assets, source_root, moon_geog)
    candidates: list[dict] = []
    for asset in assets:
        for required in [asset.image, asset.annotation]:
            if not required.is_file():
                raise FileNotFoundError(f"缺少文件: {required}")
        with rasterio.open(asset.image) as src:
            annotation_raw = read_lunar_vector(asset.annotation, moon_geog).to_crs(src.crs)
            annotations = valid_annotations(annotation_raw)
            if asset.extent_from_annotation:
                extent_geom = annotation_raw.geometry.union_all()
                # Catena 只作为困难背景；与主研究区重叠的部分必须去掉，
                # 防止同一月面像素以不同 asset 身份跨 split 重复出现。
                primary_in_asset_crs = gpd.GeoSeries([primary_union], crs=moon_geog).to_crs(src.crs).iloc[0]
                extent_geom = extent_geom.difference(primary_in_asset_crs)
            elif asset.extent is not None:
                if not asset.extent.is_file():
                    raise FileNotFoundError(f"缺少边界: {asset.extent}")
                extent_geom = read_lunar_vector(asset.extent, moon_geog).to_crs(src.crs).geometry.union_all()
            else:
                extent_geom = None

            if extent_geom is not None and extent_geom.is_empty:
                print(f"[{asset.asset_id}] 与主研究区完全重叠，跳过")
                continue

            rows, cols = candidate_range(src, extent_geom)
            total_windows = len(rows) * len(cols)
            print(f"[{asset.asset_id}] scan={total_windows}, image={src.width}x{src.height}")
            skip_asset = False
            for row in tqdm(rows, desc=asset.asset_id, unit="row"):
                for col in cols:
                    window = Window(col, row, TILE_SIZE, TILE_SIZE)
                    try:
                        image_window = src.read(window=window, masked=True)
                    except RasterioIOError as exc:
                        if asset.region_id != "catena":
                            raise RuntimeError(
                                f"正式研究区影像读取失败，不能静默跳过: "
                                f"asset={asset.asset_id}, row={row}, col={col}, file={asset.image}"
                            ) from exc
                        print(
                            f"\n[warning] {asset.asset_id} 背景影像损坏，跳过整个文件；"
                            f"此前已扫描出的该文件候选也会移除。"
                        )
                        candidates = [
                            tile for tile in candidates if tile["asset_id"] != asset.asset_id
                        ]
                        skip_asset = True
                        break
                    image_mask = np.ma.getmaskarray(image_window)
                    image_filled = image_window.filled(np.nan)
                    valid_pixels = (~np.any(image_mask, axis=0)) & np.all(
                        np.isfinite(image_filled), axis=0
                    )
                    valid_fraction = float(valid_pixels.mean())
                    wac_nonzero_fraction = float(
                        np.mean(valid_pixels & (np.abs(image_filled[0]) > 1e-8))
                    )
                    del image_window, image_mask, image_filled, valid_pixels
                    if valid_fraction < MIN_VALID_FRACTION:
                        continue
                    if wac_nonzero_fraction < MIN_WAC_NONZERO_FRACTION:
                        continue
                    transform = window_transform(window, src.transform)
                    if extent_geom is not None:
                        extent_mask = features.geometry_mask(
                            [extent_geom.__geo_interface__],
                            out_shape=(TILE_SIZE, TILE_SIZE),
                            transform=transform,
                            invert=True,
                            all_touched=False,
                        )
                        extent_fraction = float(extent_mask.mean())
                        if extent_fraction < MIN_EXTENT_FRACTION:
                            continue
                    else:
                        extent_fraction = valid_fraction
                    mask = rasterize_tile(annotations, transform)
                    counts = np.bincount(mask.ravel(), minlength=5).astype(np.int64)
                    candidates.append({
                        "asset_id": asset.asset_id,
                        "region_id": asset.region_id,
                        "image": str(asset.image),
                        "row": int(row),
                        "col": int(col),
                        "group_id": f"{asset.asset_id}_g{row // GROUP_SIZE:04d}_{col // GROUP_SIZE:04d}",
                        "valid_fraction": valid_fraction,
                        "wac_nonzero_fraction": wac_nonzero_fraction,
                        "extent_fraction": extent_fraction,
                        "class_pixels": counts.tolist(),
                    })
                if skip_asset:
                    break
    if not candidates:
        raise RuntimeError("没有生成任何候选 tile")
    return candidates


def aggregate_groups(candidates: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for index, tile in enumerate(candidates):
        gid = tile["group_id"]
        if gid not in grouped:
            grouped[gid] = {
                "group_id": gid,
                "region_id": tile["region_id"],
                "tile_indexes": [],
                "class_pixels": np.zeros(5, dtype=np.int64),
                "class_tiles": np.zeros(5, dtype=np.int64),
                "positive_tiles": 0,
            }
        grouped[gid]["tile_indexes"].append(index)
        pixels = np.asarray(tile["class_pixels"], dtype=np.int64)
        grouped[gid]["class_pixels"] += pixels
        grouped[gid]["class_tiles"] += (pixels > 0).astype(np.int64)
        grouped[gid]["positive_tiles"] += int(np.any(pixels[1:] > 0))
    return list(grouped.values())


def split_groups(groups: list[dict], seed: int, trials: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    targets = np.array([0.8, 0.1, 0.1], dtype=np.float64)
    tile_counts = np.array([len(g["tile_indexes"]) for g in groups], dtype=np.float64)
    class_pixels = np.stack([g["class_pixels"] for g in groups]).astype(np.float64)
    class_tiles = np.stack([g["class_tiles"] for g in groups]).astype(np.float64)
    positive_tiles = np.array([g["positive_tiles"] for g in groups], dtype=np.float64)
    regions = sorted({g["region_id"] for g in groups})
    region_matrix = np.zeros((len(groups), len(regions)), dtype=np.float64)
    region_positive_matrix = np.zeros((len(groups), len(regions)), dtype=np.float64)
    region_to_col = {name: i for i, name in enumerate(regions)}
    for i, group in enumerate(groups):
        col = region_to_col[group["region_id"]]
        region_matrix[i, col] = tile_counts[i]
        region_positive_matrix[i, col] = positive_tiles[i]
    region_group_counts = np.count_nonzero(region_matrix, axis=0)
    region_positive_group_counts = np.count_nonzero(region_positive_matrix, axis=0)

    best_score = float("inf")
    best_assign = None
    for _ in range(trials):
        assign = rng.choice(3, size=len(groups), p=targets)
        # 避免某个 split 为空。
        if len(set(assign.tolist())) < 3:
            continue
        split_tiles = np.array([tile_counts[assign == s].sum() for s in range(3)])
        tile_ratio = split_tiles / max(split_tiles.sum(), 1.0)
        score = 30.0 * float(np.square((tile_ratio - targets) / targets).sum())
        if abs(tile_ratio[0] - targets[0]) > 0.025 or np.any(abs(tile_ratio[1:] - targets[1:]) > 0.02):
            score += 1000.0

        # 主要平衡“含某类的 tile 数”，避免细线像素面积小导致 Val/Test 样本过少。
        split_class_tiles = np.stack([class_tiles[assign == s].sum(axis=0) for s in range(3)])
        for class_id in range(1, 5):
            total = split_class_tiles[:, class_id].sum()
            if total <= 0 or np.any(split_class_tiles[:, class_id] == 0):
                score += 1000.0
            else:
                ratio = split_class_tiles[:, class_id] / total
                score += 4.0 * float(np.square((ratio - targets) / targets).sum())
                min_eval = max(1, min(10, int(total * 0.08)))
                for split_id in (1, 2):
                    deficit = max(0.0, min_eval - split_class_tiles[split_id, class_id])
                    score += 100.0 * deficit / min_eval

        # 类别像素量只作为次级目标，不再支配划分。
        split_class_pixels = np.stack([class_pixels[assign == s].sum(axis=0) for s in range(3)])
        for class_id in range(1, 5):
            total = split_class_pixels[:, class_id].sum()
            if total > 0:
                ratio = split_class_pixels[:, class_id] / total
                score += 0.05 * float(np.square((ratio - targets) / targets).sum())

        split_region = np.stack([region_matrix[assign == s].sum(axis=0) for s in range(3)])
        split_region_positive = np.stack([
            region_positive_matrix[assign == s].sum(axis=0) for s in range(3)
        ])
        for col in range(len(regions)):
            total = split_region[:, col].sum()
            if region_group_counts[col] >= 3 and np.any(split_region[:, col] == 0):
                score += 1000.0
            if total > 0:
                ratio = split_region[:, col] / total
                score += 1.0 * float(np.square((ratio - targets) / targets).sum())
            if region_positive_group_counts[col] >= 3:
                for split_id in (1, 2):
                    if split_region_positive[split_id, col] == 0:
                        score += 1000.0

        if score < best_score:
            best_score, best_assign = score, assign.copy()
    if best_assign is None:
        raise RuntimeError("空间分组搜索失败")
    print(f"best split score={best_score:.6f}")
    problems = []
    final_tiles = np.array([tile_counts[best_assign == s].sum() for s in range(3)])
    final_ratio = final_tiles / final_tiles.sum()
    if abs(final_ratio[0] - targets[0]) > 0.025 or np.any(abs(final_ratio[1:] - targets[1:]) > 0.02):
        problems.append(f"总 tile 比例偏差过大: {final_ratio.tolist()}")
    final_class_tiles = np.stack([
        class_tiles[best_assign == s].sum(axis=0) for s in range(3)
    ])
    for class_id in range(1, 5):
        total = final_class_tiles[:, class_id].sum()
        min_eval = max(1, min(10, int(total * 0.08)))
        for split_id, split_name in ((1, "val"), (2, "test")):
            if final_class_tiles[split_id, class_id] < min_eval:
                problems.append(
                    f"{split_name} 的 {CLASS_NAMES[class_id]} tile 仅"
                    f"{int(final_class_tiles[split_id, class_id])}，最低要求 {min_eval}"
                )
    final_region_positive = np.stack([
        region_positive_matrix[best_assign == s].sum(axis=0) for s in range(3)
    ])
    for col, region in enumerate(regions):
        if region_positive_group_counts[col] >= 3:
            for split_id, split_name in ((1, "val"), (2, "test")):
                if final_region_positive[split_id, col] == 0:
                    problems.append(f"{split_name} 的 {region} 没有正样本 tile")
    if problems:
        raise RuntimeError("空间分组结果未通过硬约束:\n- " + "\n- ".join(problems))
    print(f"tile ratios train/val/test={final_ratio.tolist()}")
    names = ["train", "val", "test"]
    return {group["group_id"]: names[int(best_assign[i])] for i, group in enumerate(groups)}


def output_is_safe(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空，为避免覆盖已停止: {output}")


def write_dataset(
    candidates: list[dict], assignment: dict[str, str], output: Path,
    source_root: Path, assets: list[Asset],
) -> None:
    output_is_safe(output)
    for split in ["train", "val", "test"]:
        (output / split / "image").mkdir(parents=True, exist_ok=True)
        (output / split / "mask").mkdir(parents=True, exist_ok=True)

    moon_geog = moon_geographic_crs(source_root)
    annotation_cache = {}
    source_cache = {}
    asset_by_id = {asset.asset_id: asset for asset in assets}
    rows = []
    channel_sum = np.zeros(5, dtype=np.float64)
    channel_sumsq = np.zeros(5, dtype=np.float64)
    channel_count = np.zeros(5, dtype=np.int64)
    try:
        for tile in tqdm(candidates, desc="write tiles", unit="tile"):
            split = assignment[tile["group_id"]]
            image_path = Path(tile["image"])
            asset_id = tile["asset_id"]
            if str(image_path) not in source_cache:
                source_cache[str(image_path)] = rasterio.open(image_path)
            src = source_cache[str(image_path)]
            if asset_id not in annotation_cache:
                # 找回对应标注路径。
                asset = asset_by_id[asset_id]
                annotation_cache[asset_id] = valid_annotations(read_lunar_vector(asset.annotation, moon_geog).to_crs(src.crs))
            annotations = annotation_cache[asset_id]
            window = Window(tile["col"], tile["row"], TILE_SIZE, TILE_SIZE)
            transform = window_transform(window, src.transform)
            image = src.read(window=window).astype(np.float32)
            mask = rasterize_tile(annotations, transform)
            name = f"{asset_id}_r{tile['row']:05d}_c{tile['col']:05d}.tif"

            if split == "train":
                for channel in range(5):
                    values = image[channel]
                    valid_values = values[np.isfinite(values) & (values > -1e10)].astype(np.float64)
                    channel_sum[channel] += valid_values.sum()
                    channel_sumsq[channel] += np.square(valid_values).sum()
                    channel_count[channel] += valid_values.size

            image_profile = src.profile.copy()
            image_profile.update(
                width=TILE_SIZE, height=TILE_SIZE, count=5, transform=transform,
                dtype="float32", compress="deflate", predictor=3, zlevel=4,
                tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
            )
            with rasterio.open(output / split / "image" / name, "w", **image_profile) as dst:
                dst.write(image)
            mask_profile = {
                "driver": "GTiff", "width": TILE_SIZE, "height": TILE_SIZE,
                "count": 1, "dtype": "uint8", "crs": src.crs, "transform": transform,
                "nodata": 255, "compress": "deflate", "tiled": True,
                "blockxsize": 512, "blockysize": 512,
            }
            with rasterio.open(output / split / "mask" / name, "w", **mask_profile) as dst:
                dst.write(mask, 1)
            row = dict(tile)
            row["split"] = split
            for class_id, count in enumerate(tile["class_pixels"]):
                row[f"pixels_{CLASS_NAMES[class_id]}"] = count
            row.pop("class_pixels")
            rows.append(row)
    finally:
        for src in source_cache.values():
            src.close()

    fields = list(rows[0].keys())
    with open(output / "tile_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mean = channel_sum / np.maximum(channel_count, 1)
    variance = channel_sumsq / np.maximum(channel_count, 1) - np.square(mean)
    stats = {
        "channel_order": ["WAC", "DEM", "Slope", "TPI", "Profile Curvature"],
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
        "valid_pixel_count": channel_count.tolist(),
    }
    (output / "normalization_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarize(candidates: list[dict], assignment: dict[str, str]) -> dict:
    summary = {}
    for split in ["train", "val", "test"]:
        selected = [t for t in candidates if assignment[t["group_id"]] == split]
        pixels = np.sum([t["class_pixels"] for t in selected], axis=0).astype(np.int64)
        class_tiles = np.sum(
            [np.asarray(t["class_pixels"]) > 0 for t in selected], axis=0
        ).astype(np.int64)
        positive_tiles = sum(any(np.asarray(t["class_pixels"])[1:] > 0) for t in selected)
        regions = defaultdict(int)
        region_positive = defaultdict(int)
        for tile in selected:
            regions[tile["region_id"]] += 1
            if any(np.asarray(tile["class_pixels"])[1:] > 0):
                region_positive[tile["region_id"]] += 1
        summary[split] = {
            "tiles": len(selected),
            "positive_tiles": int(positive_tiles),
            "background_only_tiles": int(len(selected) - positive_tiles),
            "class_tiles": {CLASS_NAMES[i]: int(class_tiles[i]) for i in range(5)},
            "class_pixels": {CLASS_NAMES[i]: int(pixels[i]) for i in range(5)},
            "regions": dict(sorted(regions.items())),
            "region_positive_tiles": dict(sorted(region_positive.items())),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--old-5ch-root", type=Path, default=OLD_5CH_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--plan-only", action="store_true", help="只生成候选清单和划分报告，不写 tile")
    args = parser.parse_args()
    configure_environment()

    assets = standard_assets(args.source_root, args.old_5ch_root, args.work_root)
    candidates = enumerate_candidates(assets, args.source_root)
    groups = aggregate_groups(candidates)
    assignment = split_groups(groups, args.seed, args.trials)
    summary = summarize(candidates, assignment)
    args.work_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.work_root / f"spatial_split_plan_{args.output.name}.json"
    plan_path.write_text(
        json.dumps({"seed": args.seed, "tile_size": TILE_SIZE, "group_size": GROUP_SIZE,
                    "summary": summary, "assignment": assignment}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"plan: {plan_path}")
    if not args.plan_only:
        write_dataset(candidates, assignment, args.output, args.source_root, assets)
        (args.output / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"dataset: {args.output}")


if __name__ == "__main__":
    main()
