"""Train one controlled ResNet50 module experiment on the fixed overlap40 split.

The script selects checkpoints only by validation foreground mIoU and does not
load or evaluate the test split. Test evaluation is a separate, explicit step
after architecture selection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "utils", PROJECT_ROOT / "datasets"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import metrics  # noqa: E402
from experiment_artifacts import save_training_history  # noqa: E402
from MyDataset import MyDataset  # noqa: E402
from models.module_models import build_module_model  # noqa: E402


CLASS_WEIGHTS = [0.15, 1.0, 2.73, 1.98, 2.12]
DATA_METADATA_FILES = (
    "dataset_protocol.json",
    "dataset_summary.json",
    "normalization_stats.json",
    "tile_manifest.csv",
)
GATED_MODULE_NAMES = {"gated_boundary", "gated_cmcr"}
FEC_MODULE_NAMES = {"deeplab_fec", "gated_fec"}
BOUNDARY_MODULE_NAMES = GATED_MODULE_NAMES | {"gated_fec"}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def fingerprint_dataset_metadata(data_root: Path) -> dict[str, str]:
    """Hash split-defining metadata so runs on separate accounts are comparable."""
    fingerprints: dict[str, str] = {}
    for name in DATA_METADATA_FILES:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset metadata required for fingerprint: {path}")
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def masks_to_boundaries(labels: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Build a dilated binary foreground boundary target from multiclass masks."""
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    vertical &= (labels[:, 1:, :] > 0) | (labels[:, :-1, :] > 0)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    horizontal &= (labels[:, :, 1:] > 0) | (labels[:, :, :-1] > 0)
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary = F.max_pool2d(boundary[:, None].float(), kernel_size=5, stride=1, padding=2)
    return F.interpolate(boundary, size=output_size, mode="nearest")


class ExperimentLoss(nn.Module):
    def __init__(
        self,
        module_name: str,
        boundary_weight: float = 0.2,
        foreground_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.boundary_weight = boundary_weight
        self.foreground_weight = foreground_weight
        self.register_buffer("class_weights", torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))
        self.semantic = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_logits: torch.Tensor | None,
        foreground_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        semantic_loss = self.semantic(logits, labels)
        boundary_loss = logits.new_zeros(())
        if self.module_name in BOUNDARY_MODULE_NAMES:
            if boundary_logits is None:
                raise RuntimeError(f"{self.module_name} model did not return boundary logits")
            targets = masks_to_boundaries(labels, boundary_logits.shape[-2:])
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits, targets, pos_weight=logits.new_tensor(4.0)
            )
        foreground_loss = logits.new_zeros(())
        if self.module_name in FEC_MODULE_NAMES:
            if foreground_logits is None:
                raise RuntimeError(f"{self.module_name} did not return foreground logits")
            foreground_target = F.interpolate(
                (labels > 0)[:, None].float(),
                size=foreground_logits.shape[-2:],
                mode="nearest",
            )
            foreground_loss = F.binary_cross_entropy_with_logits(
                foreground_logits,
                foreground_target,
                pos_weight=logits.new_tensor(4.0),
            )
        total = (
            semantic_loss
            + self.boundary_weight * boundary_loss
            + self.foreground_weight * foreground_loss
        )
        return total, {
            "semantic_loss": float(semantic_loss.detach()),
            "boundary_loss": float(boundary_loss.detach()),
            "foreground_loss": float(foreground_loss.detach()),
        }


def model_outputs(model: nn.Module, images: torch.Tensor, with_aux: bool):
    if with_aux:
        outputs = model.forward_with_aux(images)
        return outputs["logits"], outputs["boundary_logits"]
    return model(images), None


def training_outputs(model: nn.Module, images: torch.Tensor, module_name: str):
    if module_name in BOUNDARY_MODULE_NAMES | FEC_MODULE_NAMES:
        outputs = model.forward_with_aux(images)
        return (
            outputs["logits"],
            outputs.get("boundary_logits"),
            outputs.get("foreground_logits"),
        )
    return model(images), None, None


def summarize_hist(hist: torch.Tensor) -> dict:
    result = dict(metrics.metrics_from_hist(hist))
    result["miou_fg"] = float(np.mean(result["iou_per_class"][1:]))
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: ExperimentLoss,
    device: torch.device,
    with_aux: bool,
    max_steps: int,
) -> dict:
    model.eval()
    hist = torch.zeros(5, 5, dtype=torch.float64)
    losses: list[float] = []
    semantic_losses: list[float] = []
    boundary_losses: list[float] = []
    foreground_losses: list[float] = []
    for step, (images, labels, _) in enumerate(tqdm(loader, desc="Validation", unit="batch")):
        if max_steps and step >= max_steps:
            break
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, boundary_logits, foreground_logits = training_outputs(
                model, images, criterion.module_name
            )
            loss, parts = criterion(
                logits, labels, boundary_logits, foreground_logits
            )
        losses.append(float(loss))
        semantic_losses.append(parts["semantic_loss"])
        boundary_losses.append(parts["boundary_loss"])
        foreground_losses.append(parts["foreground_loss"])
        hist += metrics.multiclass_confusion(logits.argmax(1), labels, 5).double()
    result = summarize_hist(hist)
    result.update(
        loss=float(np.mean(losses)),
        semantic_loss=float(np.mean(semantic_losses)),
        boundary_loss=float(np.mean(boundary_losses)),
        foreground_loss=float(np.mean(foreground_losses)),
    )
    return result


def load_frozen_cmcr_base(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> dict[str, list[str]]:
    """Load a trained Gated parent and leave only the zero-init CMCR trainable."""
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(checkpoint_path, map_location=device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected or any(not key.startswith("cmcr.") for key in missing):
        raise RuntimeError(
            "base checkpoint is not a compatible Gated model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not missing:
        raise RuntimeError("expected a Gated checkpoint without CMCR parameters")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("cmcr."))
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve() / f"result_{args.run_name}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    data_root = Path(args.data_dir).resolve()
    train_data = MyDataset(str(data_root / "train" / "image"), str(data_root / "train" / "mask"))
    val_data = MyDataset(str(data_root / "val" / "image"), str(data_root / "val" / "mask"))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_module_model(args.module, encoder_weights=encoder_weights).to(device)
    load_report = None
    init_checkpoint_sha256 = None
    if args.freeze_base:
        load_report = load_frozen_cmcr_base(model, args.init_checkpoint, device)
        init_checkpoint_sha256 = hashlib.sha256(
            Path(args.init_checkpoint).read_bytes()
        ).hexdigest()
    with_aux = args.module in GATED_MODULE_NAMES
    criterion = ExperimentLoss(
        args.module, args.boundary_weight, args.foreground_weight
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = output_dir / "best_model.pth"
    history: list[dict] = []
    best: dict | None = None

    config = vars(args).copy()
    config.update(
        data_dir=str(data_root),
        dataset_metadata_sha256=fingerprint_dataset_metadata(data_root),
        output_dir=str(output_dir),
        device=str(device),
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in trainable_parameters),
        init_checkpoint_sha256=init_checkpoint_sha256,
        checkpoint_load_report=load_report,
        class_weights=CLASS_WEIGHTS,
        selection_metric="val_mIoU_fg",
        automatic_test_evaluation=False,
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))

    epochs_without_improvement = 0
    if args.freeze_base:
        baseline_validation = evaluate(
            model, val_loader, criterion, device, with_aux, args.max_steps
        )
        best = {"epoch": 0, **baseline_validation}
        torch.save(model.state_dict(), checkpoint)
        print(
            json.dumps(
                {"frozen_base_epoch_zero": baseline_validation},
                ensure_ascii=False,
            )
        )

    for epoch in range(1, args.epochs + 1):
        if args.freeze_base:
            model.eval()
            model.cmcr.train()
        else:
            model.train()
        optimizer.zero_grad(set_to_none=True)
        train_hist = torch.zeros(5, 5, dtype=torch.float64)
        train_losses: list[float] = []
        train_semantic_losses: list[float] = []
        train_boundary_losses: list[float] = []
        train_foreground_losses: list[float] = []
        for step, (images, labels, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        ):
            if args.max_steps and step >= args.max_steps:
                break
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, boundary_logits, foreground_logits = training_outputs(
                    model, images, args.module
                )
                full_loss, parts = criterion(
                    logits, labels, boundary_logits, foreground_logits
                )
                if not torch.isfinite(full_loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch={epoch}, step={step}"
                    )
                loss = full_loss / args.accum_steps
            scaler.scale(loss).backward()
            reached_requested_limit = bool(
                args.max_steps and (step + 1) >= args.max_steps
            )
            if (
                (step + 1) % args.accum_steps == 0
                or (step + 1) == len(train_loader)
                or reached_requested_limit
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(full_loss.detach()))
            train_semantic_losses.append(parts["semantic_loss"])
            train_boundary_losses.append(parts["boundary_loss"])
            train_foreground_losses.append(parts["foreground_loss"])
            train_hist += metrics.multiclass_confusion(logits.argmax(1), labels, 5).double()

        train_metrics = summarize_hist(train_hist)
        val_metrics = evaluate(
            model, val_loader, criterion, device, with_aux, args.max_steps
        )
        is_best = best is None or val_metrics["miou_fg"] > best["miou_fg"]
        if is_best:
            best = {"epoch": epoch, **val_metrics}
            torch.save(model.state_dict(), checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(train_losses)),
            "train_semantic_loss": float(np.mean(train_semantic_losses)),
            "train_boundary_loss": float(np.mean(train_boundary_losses)),
            "train_foreground_loss": float(np.mean(train_foreground_losses)),
            "train_accuracy": train_metrics["accuracy"],
            "train_miou": train_metrics["miou"],
            "train_miou_fg": train_metrics["miou_fg"],
            "train_mf1": train_metrics["mf1"],
            "val_loss": val_metrics["loss"],
            "val_semantic_loss": val_metrics["semantic_loss"],
            "val_boundary_loss": val_metrics["boundary_loss"],
            "val_foreground_loss": val_metrics["foreground_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_miou": val_metrics["miou"],
            "val_miou_fg": val_metrics["miou_fg"],
            "val_mf1": val_metrics["mf1"],
            "val_iou_per_class": val_metrics["iou_per_class"],
            "is_best": int(is_best),
        }
        history.append(row)
        save_training_history(history, str(output_dir))
        print(json.dumps(row, ensure_ascii=False))
        scheduler.step()
        if args.early_stopping_patience and (
            epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epoch} epochs; "
                f"best epoch={best['epoch']}, best val_mIoU_fg={best['miou_fg']:.6f}"
            )
            break

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    result = {
        "model": args.module,
        "encoder": "resnet50",
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "git_commit": commit,
        "selection_metric": "val_mIoU_fg",
        "best_validation": best,
        "checkpoint": str(checkpoint),
        "test_evaluated": False,
        "config_file": "config.json",
        "history_file": "history.csv",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        required=True,
        choices=(
            "deeplab",
            "deeplab_cmcr",
            "deeplab_fec",
            "dsconv",
            "gated_boundary",
            "gated_cmcr",
            "gated_fec",
        ),
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--foreground-weight", type=float, default=0.1)
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.accum_steps <= 0:
        parser.error("batch-size and accum-steps must be positive")
    if args.boundary_weight < 0:
        parser.error("boundary-weight must be non-negative")
    if args.foreground_weight < 0:
        parser.error("foreground-weight must be non-negative")
    if args.early_stopping_patience < 0:
        parser.error("early-stopping-patience must be non-negative")
    if args.freeze_base and args.module != "gated_cmcr":
        parser.error("freeze-base currently requires --module gated_cmcr")
    if args.freeze_base and not args.init_checkpoint:
        parser.error("freeze-base requires --init-checkpoint")
    if args.freeze_base and not Path(args.init_checkpoint).is_file():
        parser.error(f"init checkpoint not found: {args.init_checkpoint}")
    return args


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
