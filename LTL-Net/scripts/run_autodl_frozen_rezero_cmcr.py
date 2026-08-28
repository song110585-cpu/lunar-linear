"""Validate and launch a frozen Gated-ReZero -> CMCR AutoDL experiment."""
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


def build_frozen_experiment_models(config: dict, model_builder):
    """Build the declared parent and correction model without hard-coded variants."""
    parent = model_builder(config["parent_model"], encoder_weights=None)
    enhanced = model_builder(config["module"], encoder_weights=None)
    return parent, enhanced


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

    if config["module"] not in {"gated_rezero_cmcr", "gated_rezero_ms_cmcr"}:
        raise ValueError(f"unexpected module: {config['module']}")
    if config["parent_model"] != "gated_rezero":
        raise ValueError(f"unexpected parent model: {config['parent_model']}")
    if config["seed"] not in {42, 1337} or config["parent_seed"] != config["seed"]:
        raise ValueError("CMCR seed must match the paired Gated-ReZero parent seed")
    if not config["freeze_base"]:
        raise ValueError("the Gated-ReZero parent must be frozen")
    if config["batch_size"] != 4 or config["accum_steps"] != 1:
        raise ValueError("the controlled experiment requires physical batch4")
    if config["boundary_weight"] != 0.0:
        raise ValueError("boundary loss must remain disabled")
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
            "Gated-ReZero checkpoint SHA-256 mismatch: "
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

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    if str(project_dir / "scripts") not in sys.path:
        sys.path.insert(0, str(project_dir / "scripts"))
    from models.module_models import build_module_model
    from train_module_experiment import (
        ExperimentLoss,
        load_frozen_cmcr_base,
        training_outputs,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the batch4 smoke test")
    device = torch.device("cuda")
    parent, enhanced = build_frozen_experiment_models(config, build_module_model)
    parent = parent.to(device).eval()
    parent.load_state_dict(
        torch.load(init_checkpoint, map_location=device, weights_only=True), strict=True
    )
    enhanced = enhanced.to(device)
    load_report = load_frozen_cmcr_base(enhanced, str(init_checkpoint), device)
    enhanced.eval()
    enhanced.cmcr.train()

    torch.manual_seed(config["seed"])
    identity_input = torch.randn(1, 5, 128, 128, device=device)
    with torch.no_grad():
        parent_outputs = parent.forward_with_aux(identity_input)
        enhanced_outputs = enhanced.forward_with_aux(identity_input)
    if not torch.equal(parent_outputs["logits"], enhanced_outputs["logits"]):
        raise RuntimeError("zero-init CMCR is not an exact parent identity")
    if not torch.equal(
        parent_outputs["boundary_logits"], enhanced_outputs["boundary_logits"]
    ):
        raise RuntimeError("parent boundary output changed during CMCR construction")
    del parent, parent_outputs, enhanced_outputs, identity_input
    torch.cuda.empty_cache()

    trainable = [parameter for parameter in enhanced.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    expected_trainable_count = config.get("expected_trainable_parameter_count", 42133)
    if trainable_count != expected_trainable_count:
        raise RuntimeError(f"unexpected CMCR trainable parameter count: {trainable_count}")
    if torch.count_nonzero(enhanced.cmcr.residual_head.weight):
        raise RuntimeError("CMCR residual head must start at exact zero")

    criterion = ExperimentLoss(
        "gated_rezero_cmcr", boundary_weight=config["boundary_weight"]
    ).to(device)
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])
    images = torch.randn(4, 5, 512, 512, device=device)
    labels = torch.randint(0, 5, (4, 512, 512), device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda"):
        logits, boundary_logits, foreground_logits = training_outputs(
            enhanced, images, config["module"]
        )
        loss, parts = criterion(
            logits, labels, boundary_logits, foreground_logits
        )
    if logits.shape != (4, 5, 512, 512) or not torch.isfinite(loss):
        raise RuntimeError(f"invalid smoke result: shape={logits.shape}, loss={loss}")
    loss.backward()
    cmcr_gradient = enhanced.cmcr.residual_head.weight.grad
    if cmcr_gradient is None or not torch.isfinite(cmcr_gradient).all():
        raise RuntimeError("CMCR residual head did not receive a finite gradient")
    optimizer.step()
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(
        json.dumps(
            {
                "module": config["module"],
                "parent_seed": config["parent_seed"],
                "checkpoint_sha256": actual_checkpoint_hash,
                "load_report": load_report,
                "trainable_parameter_count": trainable_count,
                "smoke_loss": float(loss.detach()),
                "smoke_parts": parts,
                "peak_memory_gib": peak_memory_gib,
                "dataset_tiles": tile_counts,
                "test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del enhanced, criterion, optimizer, images, labels, logits, loss
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
        "--encoder-weights",
        config["encoder_weights"],
        "--init-checkpoint",
        str(init_checkpoint),
        "--freeze-base",
        "--early-stopping-patience",
        str(config["early_stopping_patience"]),
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)

    metrics_path = result_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    initial = metrics.get("initial_validation")
    if initial is None:
        raise RuntimeError("frozen run did not record epoch-zero validation")
    expected_initial = config["expected_initial_validation_miou_fg"]
    tolerance = config["initial_validation_tolerance"]
    if abs(initial["miou_fg"] - expected_initial) > tolerance:
        raise RuntimeError(
            "epoch-zero parent validation mismatch: "
            f"actual={initial['miou_fg']}, expected={expected_initial}, "
            f"tolerance={tolerance}"
        )

    archive_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
    print(f"Training complete: {result_dir}")
    print(f"Archive: {archive_path}")
    print("Test was not evaluated.")


if __name__ == "__main__":
    main()
