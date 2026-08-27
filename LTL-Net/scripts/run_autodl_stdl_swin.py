"""Preflight and launch one controlled STDL SwinV2 experiment on AutoDL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REQUIRED_METADATA = (
    "dataset_protocol.json",
    "dataset_summary.json",
    "normalization_stats.json",
    "tile_manifest.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_count(data_root: Path, split: str) -> int:
    image_dir = data_root / split / "image"
    return len([*image_dir.glob("*.tif"), *image_dir.glob("*.tiff")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrain-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    pretrain_dir = Path(args.pretrain_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    expected_modules = {"stdl_swinv2_small": "small", "stdl_swinv2_base": "base"}
    if config["module"] not in expected_modules:
        raise ValueError(f"not an STDL Swin experiment: {config['module']}")
    expected_variant = expected_modules[config["module"]]
    assert config["encoder"] == f"swinv2_{expected_variant}"
    assert config["freeze_stages"] == 0
    assert config["batch_size"] == 4 and config["accum_steps"] == 1
    assert config["automatic_test_evaluation"] is False
    assert config["selection_metric"] == "val_mIoU_fg"

    if not project_dir.is_dir():
        raise FileNotFoundError(f"project directory not found: {project_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_root}")
    pretrain_path = pretrain_dir / config["pretrain_filename"]
    if not pretrain_path.is_file():
        raise FileNotFoundError(
            f"pretrained weight not found: {pretrain_path}\n"
            "Upload the exact file to PRETRAIN_DIR before running this notebook."
        )
    actual_pretrain_hash = sha256_file(pretrain_path)
    if actual_pretrain_hash != config["expected_pretrain_sha256"]:
        raise RuntimeError(
            "pretrained weight SHA-256 mismatch: "
            f"actual={actual_pretrain_hash}, expected={config['expected_pretrain_sha256']}"
        )

    actual_metadata = {}
    for name in REQUIRED_METADATA:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"dataset metadata not found: {path}")
        actual_metadata[name] = sha256_file(path)
    if actual_metadata != config["expected_metadata_sha256"]:
        raise RuntimeError(
            f"dataset metadata mismatch: {json.dumps(actual_metadata, indent=2)}"
        )
    actual_tiles = {
        split: image_count(data_root, split) for split in ("train", "val", "test")
    }
    if actual_tiles != config["expected_tiles"]:
        raise RuntimeError(
            f"dataset tile counts mismatch: actual={actual_tiles}, "
            f"expected={config['expected_tiles']}"
        )

    result_dir = output_root / f"result_{config['run_name']}"
    if result_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {result_dir}")

    os.environ["SWIN_PRETRAIN_DIR"] = str(pretrain_dir)
    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(project_dir / "scripts"))
    import torch
    from models.module_models import build_module_model
    from train_module_experiment import ExperimentLoss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the batch4 512x512 smoke test")
    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    model = build_module_model(
        config["module"], encoder_weights=config["encoder_weights"]
    ).to(device)
    identity = model.experiment_identity
    if identity["variant"] != expected_variant or identity["freeze_stages"] != 0:
        raise RuntimeError(f"model identity mismatch: {identity}")
    report = identity["pretrained_load_report"]
    if report["source_sha256"] != config["expected_pretrain_sha256"]:
        raise RuntimeError(f"runtime pretrained report mismatch: {report}")
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    if frozen_parameters:
        raise RuntimeError(f"full tuning required, but {frozen_parameters} parameters are frozen")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    smoke_x = torch.randn(config["batch_size"], 5, 512, 512, device=device)
    smoke_y = torch.randint(
        0, 5, (config["batch_size"], 512, 512), device=device
    )
    criterion = ExperimentLoss(config["module"], 0.0, 0.0).to(device)
    with torch.amp.autocast("cuda"):
        smoke_logits = model(smoke_x)
        smoke_loss, _ = criterion(smoke_logits, smoke_y, None)
    if smoke_logits.shape != (config["batch_size"], 5, 512, 512):
        raise RuntimeError(f"unexpected smoke output shape: {smoke_logits.shape}")
    if not torch.isfinite(smoke_loss):
        raise FloatingPointError(f"non-finite smoke loss: {smoke_loss}")
    smoke_loss.backward()
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(
        json.dumps(
            {
                "preflight": "passed",
                "variant": expected_variant,
                "parameter_count": parameter_count,
                "frozen_parameter_count": frozen_parameters,
                "matched_pretrained_tensors": report["matched_tensor_count"],
                "pretrained_sha256": report["source_sha256"],
                "batch_size": config["batch_size"],
                "input_shape": list(smoke_x.shape),
                "output_shape": list(smoke_logits.shape),
                "peak_memory_gib": peak_memory_gib,
                "dataset_tiles": actual_tiles,
                "test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del model, criterion, smoke_x, smoke_y, smoke_logits, smoke_loss
    gc.collect()
    torch.cuda.empty_cache()

    command = [
        sys.executable,
        str(project_dir / "scripts" / "train_module_experiment.py"),
        "--module",
        config["module"],
        "--data-dir",
        str(data_root),
        "--output-dir",
        str(output_root),
        "--run-name",
        config["run_name"],
        "--seed",
        str(config["seed"]),
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--accum-steps",
        str(config["accum_steps"]),
        "--num-workers",
        str(config["num_workers"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--boundary-weight",
        str(config["boundary_weight"]),
        "--foreground-weight",
        str(config["foreground_weight"]),
        "--encoder-weights",
        config["encoder_weights"],
        "--expected-pretrain-sha256",
        config["expected_pretrain_sha256"],
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)

    archive_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
    print(f"Training complete: {result_dir}")
    print(f"Archive: {archive_path}")
    print("Test was not evaluated.")


if __name__ == "__main__":
    main()
