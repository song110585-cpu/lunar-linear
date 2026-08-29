"""Validate and launch one controlled DeepLab five-channel modality ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


METADATA_FILES = (
    "dataset_protocol.json",
    "dataset_summary.json",
    "normalization_stats.json",
    "tile_manifest.csv",
)
ABLATION_MODES = {"full", "wac_only", "terrain_only", "wac_dem"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_images(data_root: Path, split: str) -> int:
    image_dir = data_root / split / "image"
    return len([*image_dir.glob("*.tif"), *image_dir.glob("*.tiff")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config["module"] != "deeplab" or config["encoder"] != "resnet50":
        raise ValueError("input ablation must use the DeepLabV3+-ResNet50 baseline")
    if config["channel_mode"] not in ABLATION_MODES:
        raise ValueError(f"unexpected channel_mode: {config['channel_mode']}")
    if config["seed"] != 42 or config["epochs"] != 80:
        raise ValueError("controlled input ablation requires seed42 and 80 epochs")
    if config["batch_size"] != 4 or config["accum_steps"] != 1:
        raise ValueError("controlled input ablation requires physical batch4")
    if config["learning_rate"] != 5e-5 or config["lovasz_weight"] != 0.0:
        raise ValueError("optimizer and loss must match the baseline")
    if config["selection_metric"] != "val_mIoU_fg":
        raise ValueError("selection metric must be val_mIoU_fg")
    if config["automatic_test_evaluation"] is not False:
        raise ValueError("Test evaluation must remain disabled")
    for path in (project_dir, data_root):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = {}
    for name in METADATA_FILES:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"dataset metadata not found: {path}")
        metadata[name] = sha256_file(path)
    if metadata != config["expected_metadata_sha256"]:
        raise RuntimeError(f"dataset metadata mismatch: {json.dumps(metadata, indent=2)}")
    tile_counts = {
        split: count_images(data_root, split) for split in ("train", "val", "test")
    }
    if tile_counts != config["expected_tiles"]:
        raise RuntimeError(
            f"dataset tile counts mismatch: actual={tile_counts}, "
            f"expected={config['expected_tiles']}"
        )

    result_dir = output_root / f"result_{config['run_name']}"
    if result_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {result_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(project_dir / "scripts" / "train_module_experiment.py"),
        "--module", config["module"],
        "--data-dir", str(data_root),
        "--output-dir", str(output_root),
        "--run-name", config["run_name"],
        "--seed", str(config["seed"]),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--accum-steps", str(config["accum_steps"]),
        "--num-workers", str(config["num_workers"]),
        "--learning-rate", str(config["learning_rate"]),
        "--boundary-weight", str(config["boundary_weight"]),
        "--foreground-weight", str(config["foreground_weight"]),
        "--lovasz-weight", str(config["lovasz_weight"]),
        "--channel-mode", config["channel_mode"],
        "--encoder-weights", config["encoder_weights"],
    ]
    print(json.dumps({"config": config, "dataset_tiles": tile_counts}, ensure_ascii=False, indent=2))
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)

    result = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    if result["test_evaluated"] is not False:
        raise RuntimeError("unexpected Test evaluation")
    archive_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
    print(f"Training complete: {result_dir}")
    print(f"Archive: {archive_path}")
    print("Test was not evaluated.")


if __name__ == "__main__":
    main()
