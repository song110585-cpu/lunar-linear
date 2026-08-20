"""全量校验 v6 数据集并生成分层可视化抽检图集。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap


DEFAULT_DATASET = Path(r"E:\月球_dataset\dataset\dataset_v6_spatial811_g1024_fixed")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "data_audit_v6_g1024_fixed"
CLASS_NAMES = ["background", "wrinkle_ridge", "rille", "fault", "graben"]
CLASS_COLORS = ["#00000000", "#ff3030cc", "#32d7ffcc", "#ffe14acc", "#d94cffcc"]


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["row"] = int(row["row"])
        row["col"] = int(row["col"])
        row["class_pixels"] = np.array([
            int(row[f"pixels_{name}"]) for name in CLASS_NAMES
        ], dtype=np.int64)
        row["foreground_pixels"] = int(row["class_pixels"][1:].sum())
    return rows


def validate_all(dataset: Path, rows: list[dict]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    label_values: Counter[int] = Counter()
    actual_pixels = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    file_counts = {}
    for split in ("train", "val", "test"):
        manifest_names = {f"{r['asset_id']}_r{r['row']:05d}_c{r['col']:05d}.tif" for r in rows if r["split"] == split}
        image_names = {p.name for p in (dataset / split / "image").glob("*.tif")}
        mask_names = {p.name for p in (dataset / split / "mask").glob("*.tif")}
        file_counts[split] = {
            "manifest": len(manifest_names), "images": len(image_names), "masks": len(mask_names)
        }
        if manifest_names != image_names or manifest_names != mask_names:
            errors.append(f"{split}: manifest/image/mask 文件集合不一致")

        for name in sorted(manifest_names):
            image_path = dataset / split / "image" / name
            mask_path = dataset / split / "mask" / name
            try:
                with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
                    if (image_src.count, image_src.height, image_src.width) != (5, 512, 512):
                        errors.append(f"{split}/{name}: image shape={image_src.count,image_src.height,image_src.width}")
                    if (mask_src.count, mask_src.height, mask_src.width) != (1, 512, 512):
                        errors.append(f"{split}/{name}: mask shape={mask_src.count,mask_src.height,mask_src.width}")
                    if image_src.crs != mask_src.crs or not image_src.transform.almost_equals(mask_src.transform):
                        errors.append(f"{split}/{name}: image/mask 地理参考不一致")
                    mask = mask_src.read(1)
                    image = image_src.read(masked=True)
                    image_filled = image.filled(np.nan)
                    invalid_pixels = np.any(
                        np.ma.getmaskarray(image) | ~np.isfinite(image_filled), axis=0
                    )
                    invalid_fraction = float(invalid_pixels.mean())
                    positive_overlap = int(np.count_nonzero(invalid_pixels & (mask > 0)))
                    if invalid_fraction > 0.05:
                        errors.append(
                            f"{split}/{name}: 无效像素={invalid_fraction:.4%}, "
                            f"覆盖正标签={positive_overlap}"
                        )
                    elif invalid_fraction > 0:
                        warnings.append(
                            f"{split}/{name}: 允许的无效边缘={invalid_fraction:.4%}, "
                            f"覆盖标签={positive_overlap}（加载时清为背景）"
                        )
            except Exception as exc:
                errors.append(f"{split}/{name}: 读取失败: {exc}")
                continue
            unique, counts = np.unique(mask, return_counts=True)
            label_values.update({int(k): int(v) for k, v in zip(unique, counts)})
            invalid = unique[~np.isin(unique, np.arange(5))]
            if invalid.size:
                errors.append(f"{split}/{name}: 非法标签={invalid.tolist()}")
            actual_pixels[split] += np.bincount(mask.ravel(), minlength=5)[:5]

    manifest_pixels = {
        split: np.sum([r["class_pixels"] for r in rows if r["split"] == split], axis=0)
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        if not np.array_equal(actual_pixels[split], manifest_pixels[split]):
            errors.append(f"{split}: mask实际像素统计与manifest不一致")

    duplicate_windows = sum(
        count > 1 for count in Counter((r["asset_id"], r["row"], r["col"]) for r in rows).values()
    )
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    crossing_groups = sum(len(splits) > 1 for splits in group_splits.values())
    if duplicate_windows:
        errors.append(f"重复窗口={duplicate_windows}")
    if crossing_groups:
        errors.append(f"跨split空间组={crossing_groups}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "rows": len(rows),
        "file_counts": file_counts,
        "label_values": dict(sorted(label_values.items())),
        "duplicate_windows": duplicate_windows,
        "groups_crossing_splits": crossing_groups,
        "actual_class_pixels": {
            split: {CLASS_NAMES[i]: int(values[i]) for i in range(5)}
            for split, values in actual_pixels.items()
        },
    }


def choose_region_samples(rows: list[dict]) -> list[dict]:
    chosen = []
    regions = sorted({r["region_id"] for r in rows})
    for region in regions:
        for split in ("train", "val", "test"):
            pool = [r for r in rows if r["region_id"] == region and r["split"] == split]
            chosen.append(max(pool, key=lambda r: r["foreground_pixels"]))
    return chosen


def choose_class_samples(rows: list[dict], class_id: int, per_split: int = 3) -> list[dict]:
    chosen = []
    for split in ("train", "val", "test"):
        pool = [r for r in rows if r["split"] == split and r["class_pixels"][class_id] > 0]
        pool.sort(key=lambda r: r["class_pixels"][class_id], reverse=True)
        if not pool:
            continue
        indexes = np.linspace(0, len(pool) - 1, min(per_split, len(pool)), dtype=int)
        chosen.extend(pool[index] for index in indexes)
    return chosen


def draw_panel(ax, dataset: Path, row: dict, title: str) -> None:
    name = f"{row['asset_id']}_r{row['row']:05d}_c{row['col']:05d}.tif"
    with rasterio.open(dataset / row["split"] / "image" / name) as src:
        wac = src.read(1, masked=True).filled(np.nan)
    with rasterio.open(dataset / row["split"] / "mask" / name) as src:
        mask = src.read(1)
    valid = wac[np.isfinite(wac)]
    lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
    ax.imshow(wac, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-6))
    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(overlay, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_region_montage(dataset: Path, rows: list[dict], output: Path) -> None:
    regions = sorted({r["region_id"] for r in rows})
    chosen = {(r["region_id"], r["split"]): r for r in choose_region_samples(rows)}
    fig, axes = plt.subplots(len(regions), 3, figsize=(11, 3.4 * len(regions)))
    for y, region in enumerate(regions):
        for x, split in enumerate(("train", "val", "test")):
            row = chosen[(region, split)]
            draw_panel(axes[y, x], dataset, row, f"{region} | {split} | fg={row['foreground_pixels']}")
    fig.suptitle("v6: highest-foreground tile by region and split", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "by_region_split.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_class_montages(dataset: Path, rows: list[dict], output: Path) -> None:
    for class_id in range(1, 5):
        chosen = choose_class_samples(rows, class_id)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for ax in axes.ravel():
            ax.axis("off")
        positions = {"train": 0, "val": 0, "test": 0}
        split_row = {"train": 0, "val": 1, "test": 2}
        for row in chosen:
            y = split_row[row["split"]]
            x = positions[row["split"]]
            positions[row["split"]] += 1
            draw_panel(
                axes[y, x], dataset, row,
                f"{row['split']} | {row['region_id']} | pixels={row['class_pixels'][class_id]}",
            )
        fig.suptitle(f"class {class_id}: {CLASS_NAMES[class_id]} (red/cyan/yellow/magenta mask overlay)")
        fig.tight_layout()
        fig.savefig(output / f"class_{class_id}_{CLASS_NAMES[class_id]}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"审计输出目录已存在且非空: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.dataset / "tile_manifest.csv")
    report = validate_all(args.dataset, rows)
    save_region_montage(args.dataset, rows, args.output)
    save_class_montages(args.dataset, rows, args.output)
    report_path = args.output / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
