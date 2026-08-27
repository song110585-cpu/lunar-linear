"""Preflight and launch one controlled Gated-ReZero experiment on AutoDL."""

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


EXPECTED_HF_CACHE = {
    "config.json": "01bf2cf24eb29b405c28c159f46ceda92c098ab85868be88fb100967db47166e",
    "model.safetensors": "df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c",
}
SNAPSHOT_ID = "00cb74e366966d59cd9a35af57e618af9f88efe9"
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


def count_images(data_root: Path, split: str) -> int:
    image_dir = data_root / split / "image"
    return len([*image_dir.glob("*.tif"), *image_dir.glob("*.tiff")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-source", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    cache_source = Path(args.hf_cache_source).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["module"] == "gated_rezero"
    assert config["seed"] in {42, 1337}
    assert config["residual_scale_init"] == 0.0
    assert config["batch_size"] == 4 and config["accum_steps"] == 1
    assert config["boundary_weight"] == 0.0
    assert config["selection_metric"] == "val_mIoU_fg"
    assert config["automatic_test_evaluation"] is False
    if not project_dir.is_dir():
        raise FileNotFoundError(f"project directory not found: {project_dir}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_root}")

    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--smp-hub--resnet50.imagenet"
        / "snapshots"
        / SNAPSHOT_ID
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in EXPECTED_HF_CACHE.items():
        source = cache_source / name
        target = snapshot / name
        if not source.is_file():
            raise FileNotFoundError(f"offline ResNet50 cache file not found: {source}")
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"offline cache SHA-256 mismatch for {name}: {actual_hash}"
            )
        if not target.is_file() or sha256_file(target) != expected_hash:
            shutil.copy2(source, target)
    os.environ["HF_HUB_OFFLINE"] = "1"

    metadata = {}
    for name in REQUIRED_METADATA:
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

    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(project_dir / "scripts"))
    import torch
    from models.module_models import build_module_model
    from train_module_experiment import ExperimentLoss, model_outputs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the batch4 smoke test")
    torch.manual_seed(config["seed"])
    model = build_module_model("gated_rezero", encoder_weights="imagenet").cuda().train()
    identity = model.experiment_identity
    if identity["residual_scale_init"] != 0.0:
        raise RuntimeError(f"wrong model identity: {identity}")
    alpha = model.boundary_refinement.residual_scale
    if alpha is None or alpha.detach().item() != 0.0:
        raise RuntimeError(f"residual scale must start at zero: {alpha}")
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    if frozen_parameters:
        raise RuntimeError(f"full tuning required, but {frozen_parameters} parameters are frozen")

    criterion = ExperimentLoss("gated_rezero", boundary_weight=0.0).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    images = torch.randn(4, 5, 512, 512, device="cuda")
    labels = torch.randint(0, 5, (4, 512, 512), device="cuda")
    with torch.amp.autocast("cuda"):
        logits, boundary_logits = model_outputs(model, images, True)
        loss, parts = criterion(logits, labels, boundary_logits)
    if logits.shape != (4, 5, 512, 512) or not torch.isfinite(loss):
        raise RuntimeError(f"invalid smoke result: shape={logits.shape}, loss={loss}")
    loss.backward()
    optimizer.step()
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(
        json.dumps(
            {
                "preflight": "passed",
                "model_identity": identity,
                "seed": config["seed"],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "frozen_parameter_count": frozen_parameters,
                "initial_alpha": 0.0,
                "alpha_after_one_step": float(alpha.detach()),
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
    del model, criterion, optimizer, images, labels, logits, boundary_logits, loss
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
        "--encoder-weights",
        config["encoder_weights"],
    ]
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)
    archive_path = shutil.make_archive(str(result_dir), "zip", root_dir=result_dir)
    print(f"Training complete: {result_dir}")
    print(f"Archive: {archive_path}")
    print("Test was not evaluated.")


if __name__ == "__main__":
    main()
