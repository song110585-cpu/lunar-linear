"""检测旧5通道影像中被写成普通0值、未被NoData掩膜识别的WAC空白区。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


DATASET = Path(r"E:\月球_dataset\dataset\dataset_v6_spatial811_g1024_fixed")
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "data_audit_v6_g1024_fixed" / "wac_coverage_report.json"


def main() -> None:
    with (DATASET / "tile_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    affected = []
    stats = defaultdict(lambda: {"tiles": 0, "wac_zero_over_5pct": 0, "wac_all_zero": 0})
    for row in tqdm(rows, desc="WAC coverage", unit="tile"):
        split, region = row["split"], row["region_id"]
        name = f"{row['asset_id']}_r{int(row['row']):05d}_c{int(row['col']):05d}.tif"
        with rasterio.open(DATASET / split / "image" / name) as src:
            wac = src.read(1)
        zero_fraction = float(np.mean(wac == 0))
        key = f"{split}/{region}"
        stats[key]["tiles"] += 1
        stats[key]["wac_zero_over_5pct"] += int(zero_fraction > 0.05)
        stats[key]["wac_all_zero"] += int(zero_fraction == 1.0)
        if zero_fraction > 0.05:
            affected.append({
                "split": split, "region": region, "filename": name,
                "wac_zero_fraction": zero_fraction,
            })
    payload = {
        "threshold": 0.05,
        "affected_tiles": len(affected),
        "all_zero_tiles": sum(item["wac_zero_fraction"] == 1.0 for item in affected),
        "by_split_region": dict(sorted(stats.items())),
        "worst_tiles": sorted(affected, key=lambda item: item["wac_zero_fraction"], reverse=True)[:50],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
