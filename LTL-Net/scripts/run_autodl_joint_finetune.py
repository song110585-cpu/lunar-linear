"""Validate and launch controlled joint fine-tuning from a complete checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


METADATA_FILES = (
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


def count_images(data_root: Path, split: str) -> int:
    image_dir = data_root / split / "image"
    return len([*image_dir.glob("*.tif"), *image_dir.glob("*.tiff")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    init_checkpoint = Path(args.init_checkpoint).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config["module"] != "gated_rezero_cmcr":
        raise ValueError(f"unexpected module: {config['module']}")
    if config["seed"] != 1337:
        raise ValueError("first controlled screen must use seed1337")
    if not config["joint_finetune"] or config["freeze_base"]:
        raise ValueError("joint fine-tuning must be enabled without freeze_base")
    if not config["freeze_encoder"] or not config["freeze_batch_norm"]:
        raise ValueError("encoder and BatchNorm must remain frozen")
    if config["batch_size"] != 4 or config["accum_steps"] != 1:
        raise ValueError("controlled experiment requires physical batch4")
    if config["boundary_weight"] != 0.0:
        raise ValueError("boundary auxiliary loss must remain disabled")
    if config["channel_mode"] != "full":
        raise ValueError("loss screening must use the full five-channel input")
    if config["selection_metric"] != "val_mIoU_fg":
        raise ValueError("selection metric must be val_mIoU_fg")
    if config["automatic_test_evaluation"] is not False:
        raise ValueError("Test evaluation must remain disabled")
    for path in (project_dir, data_root, init_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    actual_checkpoint_hash = sha256_file(init_checkpoint)
    if actual_checkpoint_hash != config["expected_init_checkpoint_sha256"]:
        raise RuntimeError(
            "initial checkpoint SHA-256 mismatch: "
            f"actual={actual_checkpoint_hash}, "
            f"expected={config['expected_init_checkpoint_sha256']}"
        )

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

    for path in (project_dir, project_dir / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from models.module_models import build_module_model
    from train_module_experiment import (
        ExperimentLoss,
        load_joint_finetune_checkpoint,
        set_joint_finetune_train_mode,
        training_outputs,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the smoke test")
    device = torch.device("cuda")
    model = build_module_model(config["module"], encoder_weights=None).to(device)
    load_report = load_joint_finetune_checkpoint(
        model, str(init_checkpoint), device
    )
    set_joint_finetune_train_mode(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable or any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("invalid joint fine-tune trainable scope")
    if trainable_count != config["expected_trainable_parameter_count"]:
        raise RuntimeError(
            "unexpected joint fine-tune parameter count: "
            f"actual={trainable_count}, "
            f"expected={config['expected_trainable_parameter_count']}"
        )

    criterion = ExperimentLoss(
        config["module"],
        boundary_weight=config["boundary_weight"],
        foreground_weight=config["foreground_weight"],
        lovasz_weight=config["lovasz_weight"],
    ).to(device)
    images = torch.randn(2, 5, 128, 128, device=device)
    labels = torch.randint(0, 5, (2, 128, 128), device=device)
    with torch.amp.autocast("cuda"):
        logits, boundary_logits, foreground_logits = training_outputs(
            model, images, config["module"]
        )
        loss, parts = criterion(logits, labels, boundary_logits, foreground_logits)
    if logits.shape != (2, 5, 128, 128) or not torch.isfinite(loss):
        raise RuntimeError(f"invalid smoke result: shape={logits.shape}, loss={loss}")
    loss.backward()
    if any(parameter.grad is not None for parameter in model.encoder.parameters()):
        raise RuntimeError("frozen encoder received gradients")
    print(
        json.dumps(
            {
                "checkpoint_sha256": actual_checkpoint_hash,
                "checkpoint_load_report": load_report,
                "lovasz_weight": config["lovasz_weight"],
                "trainable_parameter_count": trainable_count,
                "smoke_loss": float(loss.detach()),
                "smoke_parts": parts,
                "dataset_tiles": tile_counts,
                "test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del model, criterion, images, labels, logits, loss
    torch.cuda.empty_cache()

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
        "--init-checkpoint", str(init_checkpoint),
        "--joint-finetune",
        "--early-stopping-patience", str(config["early_stopping_patience"]),
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)

    metrics_path = result_dir / "metrics.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    initial = result.get("initial_validation")
    if initial is None:
        raise RuntimeError("joint fine-tune did not record epoch-zero validation")
    expected_initial = config["expected_initial_validation_miou_fg"]
    tolerance = config["initial_validation_tolerance"]
    if abs(initial["miou_fg"] - expected_initial) > tolerance:
        raise RuntimeError(
            "epoch-zero validation mismatch: "
            f"actual={initial['miou_fg']}, expected={expected_initial}, "
            f"tolerance={tolerance}"
        )
    archive_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
    print(f"Training complete: {result_dir}")
    print(f"Archive: {archive_path}")
    print("Test was not evaluated.")


if __name__ == "__main__":
    main()
