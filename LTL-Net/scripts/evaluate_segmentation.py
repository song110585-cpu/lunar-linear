"""用最佳权重统一生成测试指标、混淆矩阵和定性对比图。"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
for _sub in ("utils", "models", "datasets"):
    sys.path.insert(0, str(_root / _sub))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics
from MyDataset import CHANNEL_MODE_MASKS, MyDataset
from models.dlinknet import DLinkNet
from models.pidnet_multiclass import PIDNetSmall
from models.ltl_net import LTLNet
from models.module_models import build_module_model
from train_module_experiment import masks_to_boundaries


CLASS_NAMES = ["Background", "WR", "Rille", "Fault", "Graben"]
MODULE_MODEL_NAMES = {
    "deeplab",
    "deeplab_cmcr",
    "deeplab_fec",
    "dsconv",
    "gated_boundary",
    "gated_cmcr",
    "gated_fec",
    "gated_rezero",
    "gated_rezero_cmcr",
    "gated_rezero_ms_cmcr",
    "stdl_swinv2_small",
    "stdl_swinv2_base",
}
GATED_MODEL_NAMES = {
    "gated_boundary",
    "gated_cmcr",
    "gated_fec",
    "gated_rezero",
    "gated_rezero_cmcr",
    "gated_rezero_ms_cmcr",
}
LABEL_CMAP = ListedColormap(["#111111", "#f4d03f", "#2e86de", "#e74c3c", "#af7ac5"])
ERROR_CMAP = ListedColormap(["#111111", "#e74c3c", "#3498db", "#f1c40f"])


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 不是 state_dict: {type(checkpoint).__name__}")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def build_model(args):
    normalized = args.model.lower().replace("-", "_")
    if normalized in MODULE_MODEL_NAMES:
        return build_module_model(normalized, encoder_weights=None)
    if args.model.lower() in ("ltl", "ltlnet", "ltl-net"):
        return LTLNet(
            encoder_name=args.encoder,
            encoder_weights=None,
            in_channels=5,
            classes=5,
            highres_detail_channels=args.detail_channels,
        )
    if args.model.lower() in ("dlinknet", "d-linknet"):
        return DLinkNet(
            encoder_name=args.encoder,
            encoder_weights=None,
            in_channels=5,
            classes=5,
        )
    if normalized in ("pidnet", "pidnet_s", "pidnets"):
        return PIDNetSmall(in_channels=5, classes=5)
    if not hasattr(smp, args.model):
        raise ValueError(f"segmentation_models_pytorch 中不存在模型: {args.model}")
    return getattr(smp, args.model)(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=5,
        classes=5,
    )


def save_confusion_plot(hist, output_path, normalization=None):
    values = hist.astype(np.float64)
    if normalization == "row":
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    elif normalization == "column":
        values = values / np.maximum(values.sum(axis=0, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180)
    image = ax.imshow(
        values, cmap="Blues", vmin=0, vmax=1 if normalization else None
    )
    ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    titles = {
        None: "Confusion matrix (pixels)",
        "row": "Row-normalized confusion matrix",
        "column": "Column-normalized confusion matrix",
    }
    ax.set_title(titles[normalization])
    threshold = values.max() * 0.55 if values.size else 0
    for row in range(5):
        for col in range(5):
            text_value = (
                f"{values[row, col]:.3f}"
                if normalization
                else f"{int(values[row, col]):,}"
            )
            ax.text(col, row, text_value, ha="center", va="center",
                    fontsize=7.5, color="white" if values[row, col] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_class_plot(result, output_path):
    x = np.arange(5)
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
    for offset, (key, label) in enumerate((
        ("iou_per_class", "IoU"),
        ("precision_per_class", "Precision"),
        ("recall_per_class", "Recall"),
        ("f1_per_class", "F1"),
    )):
        ax.bar(x + (offset - 1.5) * width, result[key], width, label=label)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def tile_foreground_iou(hist):
    present = (hist[1:, :].sum(axis=1) + hist[:, 1:].sum(axis=0)) > 0
    if not np.any(present):
        return None
    tp = np.diag(hist)[1:]
    fp = hist[:, 1:].sum(axis=0) - tp
    fn = hist[1:, :].sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    return float(np.mean(iou[present]))


def choose_representative(rows, count):
    """只依据真值前景比例取均匀分位点，保证不同模型选择相同样本。"""
    positives = sorted((row for row in rows if row["foreground_ratio"] > 0),
                       key=lambda row: (row["foreground_ratio"], row["name"]))
    if len(positives) <= count:
        return [row["name"] for row in positives]
    indices = np.linspace(0, len(positives) - 1, count, dtype=int)
    return [positives[index]["name"] for index in indices]


def make_error_map(target, pred):
    error = np.zeros_like(target, dtype=np.uint8)
    error[(target == 0) & (pred > 0)] = 1       # false positive
    error[(target > 0) & (pred == 0)] = 2       # false negative
    error[(target > 0) & (pred > 0) & (target != pred)] = 3
    return error


def foreground_binary_confusion(hist):
    """Collapse a multiclass matrix into [[TN, FP], [FN, TP]]."""
    hist = np.asarray(hist, dtype=np.int64)
    return np.asarray(
        [
            [hist[0, 0], hist[0, 1:].sum()],
            [hist[1:, 0].sum(), hist[1:, 1:].sum()],
        ],
        dtype=np.int64,
    )


def binary_boundaries(mask):
    """One-pixel boundaries of a binary foreground mask."""
    mask = mask.bool()
    boundary = torch.zeros_like(mask)
    vertical = mask[:, 1:, :] != mask[:, :-1, :]
    horizontal = mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    return boundary


def update_boundary_counts(counts, predictions, targets, tolerances):
    pred_boundary = binary_boundaries(predictions > 0)[:, None].float()
    target_boundary = binary_boundaries(targets > 0)[:, None].float()
    for tolerance in tolerances:
        kernel = 2 * tolerance + 1
        target_dilated = F.max_pool2d(
            target_boundary, kernel_size=kernel, stride=1, padding=tolerance
        ).bool()
        pred_dilated = F.max_pool2d(
            pred_boundary, kernel_size=kernel, stride=1, padding=tolerance
        ).bool()
        entry = counts.setdefault(
            tolerance,
            {"matched_pred": 0, "pred": 0, "matched_target": 0, "target": 0},
        )
        pred_bool = pred_boundary.bool()
        target_bool = target_boundary.bool()
        entry["matched_pred"] += int((pred_bool & target_dilated).sum())
        entry["pred"] += int(pred_bool.sum())
        entry["matched_target"] += int((target_bool & pred_dilated).sum())
        entry["target"] += int(target_bool.sum())


def summarize_boundary_counts(counts):
    result = {}
    for tolerance, entry in counts.items():
        precision = entry["matched_pred"] / max(entry["pred"], 1)
        recall = entry["matched_target"] / max(entry["target"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        result[str(tolerance)] = {
            **entry,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def forward_outputs(model, images, model_name):
    normalized = model_name.lower().replace("-", "_")
    if normalized in GATED_MODEL_NAMES:
        outputs = model.forward_with_aux(images)
        return outputs["logits"], outputs["boundary_logits"]
    return model(images), None


def save_qualitative(model, model_name, split, dataset, names, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_to_index = {Path(name).stem: index for index, name in enumerate(dataset.filenames)}
    model.eval()
    for name in names:
        if name not in stem_to_index:
            print(f"[warning] 定性样本不在 {split} 集: {name}")
            continue
        image, target, _ = dataset[stem_to_index[name]]
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda"
        ):
            logits, _ = forward_outputs(model, image.unsqueeze(0).to(device), model_name)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
        target_np = target.numpy()
        wac = image[0].numpy() * float(dataset.std[0]) + float(dataset.mean[0])
        finite = np.isfinite(wac)
        low, high = np.percentile(wac[finite], [2, 98]) if finite.any() else (0.0, 1.0)
        if high <= low:
            high = low + 1.0

        fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1), dpi=180)
        axes[0].imshow(wac, cmap="gray", vmin=low, vmax=high)
        axes[0].set_title("WAC")
        axes[1].imshow(target_np, cmap=LABEL_CMAP, vmin=0, vmax=4, interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred, cmap=LABEL_CMAP, vmin=0, vmax=4, interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[3].imshow(make_error_map(target_np, pred), cmap=ERROR_CMAP,
                       vmin=0, vmax=3, interpolation="nearest")
        axes[3].set_title("Error: FP red / FN blue / wrong yellow")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LTLNet",
                        help="LTLNet、DLinkNet，或smp模型名，如DeepLabV3Plus/Unet/Linknet")
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument("--detail-channels", type=int, default=16)
    parser.add_argument("--data-dir", required=True, help="含 split/image 与 split/mask 的数据集根目录")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--channel-mode",
        choices=tuple(CHANNEL_MODE_MASKS),
        default="full",
        help="归一化后将被消融通道置零；full保持原输入",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="启用 CUDA autocast；默认使用 FP32，避免部分消费级 GPU 推理产生非有限 logits",
    )
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--sample-names-file", default=None,
                        help="可选：固定样本 stem 列表；跨模型对比时使用同一文件")
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
    dataset = MyDataset(
        str(data_root / args.split / "image"),
        str(data_root / args.split / "mask"),
        channel_mode=args.channel_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([0.15, 1.0, 2.73, 1.98, 2.12], device=device)
    )

    total_hist = np.zeros((5, 5), dtype=np.int64)
    boundary_counts = {}
    auxiliary_boundary_hist = np.zeros((2, 2), dtype=np.int64)
    auxiliary_boundary_pixels = 0
    gate_stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0, "low": 0, "high": 0}
    cmcr_gate_stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0, "low": 0, "high": 0}
    cmcr_context_gate_stats = {
        "sum": 0.0,
        "sum_sq": 0.0,
        "count": 0,
        "low": 0,
        "high": 0,
    }
    cmcr_scale_stats = {"sum": [0.0, 0.0], "sum_sq": [0.0, 0.0], "count": 0}
    hook_handles = []

    def register_gate_hook(module, stats):
        def capture_gate(_module, _inputs, output):
            values = output.detach().float()
            stats["sum"] += float(values.sum())
            stats["sum_sq"] += float((values * values).sum())
            stats["count"] += values.numel()
            stats["low"] += int((values < 0.05).sum())
            stats["high"] += int((values > 0.95).sum())

        hook_handles.append(module.register_forward_hook(capture_gate))

    def register_scale_hook(module, stats):
        def capture_scale(_module, _inputs, output):
            values = output.detach().float()
            channel_sum = values.sum(dim=(0, 2, 3)).cpu().tolist()
            channel_sum_sq = (values * values).sum(dim=(0, 2, 3)).cpu().tolist()
            for index in range(2):
                stats["sum"][index] += channel_sum[index]
                stats["sum_sq"][index] += channel_sum_sq[index]
            stats["count"] += values.shape[0] * values.shape[2] * values.shape[3]

        hook_handles.append(module.register_forward_hook(capture_scale))

    normalized_model = args.model.lower().replace("-", "_")
    if normalized_model in GATED_MODEL_NAMES:
        register_gate_hook(model.boundary_refinement.gate, gate_stats)
    if normalized_model in {"deeplab_cmcr", "gated_cmcr", "gated_rezero_cmcr"}:
        register_gate_hook(model.cmcr.consistency_gate, cmcr_gate_stats)
    if normalized_model == "gated_rezero_ms_cmcr":
        register_gate_hook(model.cmcr.local_consistency_gate, cmcr_gate_stats)
        register_gate_hook(
            model.cmcr.context_consistency_gate, cmcr_context_gate_stats
        )
        register_scale_hook(model.cmcr.scale_gate, cmcr_scale_stats)
    losses = []
    rows = []
    model.eval()
    with torch.no_grad():
        for step, (images, targets, names) in enumerate(
            tqdm(loader, desc=f"Evaluate {args.split}", unit="batch")
        ):
            if args.max_steps and step >= args.max_steps:
                break
            images = images.to(device)
            targets_device = targets.to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits, boundary_logits = forward_outputs(model, images, args.model)
                batch_loss = loss_fn(logits, targets_device)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"{args.model} 在 {args.split} step={step} 产生 NaN/Inf logits；"
                    "请关闭 --amp 或减小 --batch-size，禁止继续汇总 argmax 指标"
                )
            losses.append(float(batch_loss.item()))
            preds = logits.argmax(dim=1).cpu()
            update_boundary_counts(boundary_counts, preds, targets, (1, 2, 4, 8))
            if boundary_logits is not None:
                boundary_targets = masks_to_boundaries(
                    targets_device, boundary_logits.shape[-2:]
                ).bool()
                boundary_predictions = boundary_logits.sigmoid() >= 0.5
                target_flat = boundary_targets.view(-1)
                pred_flat = boundary_predictions.view(-1)
                auxiliary_boundary_hist[0, 0] += int((~target_flat & ~pred_flat).sum())
                auxiliary_boundary_hist[0, 1] += int((~target_flat & pred_flat).sum())
                auxiliary_boundary_hist[1, 0] += int((target_flat & ~pred_flat).sum())
                auxiliary_boundary_hist[1, 1] += int((target_flat & pred_flat).sum())
                auxiliary_boundary_pixels += target_flat.numel()
            for index, name in enumerate(names):
                hist = metrics.multiclass_confusion(preds[index], targets[index], 5).numpy()
                total_hist += hist
                target_np = targets[index].numpy()
                rows.append({
                    "name": name,
                    "foreground_ratio": float(np.mean(target_np > 0)),
                    "predicted_foreground_ratio": float(np.mean(preds[index].numpy() > 0)),
                    "present_classes": ",".join(str(v) for v in np.unique(target_np[target_np > 0])),
                    "tile_miou_fg_present": tile_foreground_iou(hist),
                    "foreground_fp_pixels": int(hist[0, 1:].sum()),
                    "foreground_fn_pixels": int(hist[1:, 0].sum()),
                    "wrong_foreground_pixels": int(
                        hist[1:, 1:].sum() - np.diag(hist)[1:].sum()
                    ),
                })

    for hook_handle in hook_handles:
        hook_handle.remove()

    result = metrics.metrics_from_hist(torch.from_numpy(total_hist))
    result["loss"] = float(np.mean(losses))
    result["miou_fg"] = float(np.mean(result["iou_per_class"][1:]))
    binary_hist = foreground_binary_confusion(total_hist)
    boundary_metrics = summarize_boundary_counts(boundary_counts)
    payload = {
        "model": args.model,
        "encoder": args.encoder,
        "detail_channels": args.detail_channels if args.model.lower() in ("ltl", "ltlnet", "ltl-net") else None,
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "split": args.split,
        "channel_mode": args.channel_mode,
        "class_names": CLASS_NAMES,
        "confusion_matrix": total_hist.tolist(),
        "foreground_binary_confusion": binary_hist.tolist(),
        "foreground_boundary_metrics": boundary_metrics,
        "metrics": result,
    }
    payload[args.split] = result
    if auxiliary_boundary_pixels:
        tn, fp, fn, tp = auxiliary_boundary_hist.ravel().tolist()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        payload["gated_boundary_head"] = {
            "confusion_matrix": auxiliary_boundary_hist.tolist(),
            "target_positive_ratio": (tp + fn) / auxiliary_boundary_pixels,
            "predicted_positive_ratio": (tp + fp) / auxiliary_boundary_pixels,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    if gate_stats["count"]:
        mean = gate_stats["sum"] / gate_stats["count"]
        variance = max(gate_stats["sum_sq"] / gate_stats["count"] - mean * mean, 0.0)
        payload["gate_activation"] = {
            "mean": mean,
            "std": variance ** 0.5,
            "fraction_below_0_05": gate_stats["low"] / gate_stats["count"],
            "fraction_above_0_95": gate_stats["high"] / gate_stats["count"],
        }
    if cmcr_gate_stats["count"]:
        mean = cmcr_gate_stats["sum"] / cmcr_gate_stats["count"]
        variance = max(
            cmcr_gate_stats["sum_sq"] / cmcr_gate_stats["count"] - mean * mean,
            0.0,
        )
        payload["cmcr_gate_activation"] = {
            "mean": mean,
            "std": variance ** 0.5,
            "fraction_below_0_05": cmcr_gate_stats["low"] / cmcr_gate_stats["count"],
            "fraction_above_0_95": cmcr_gate_stats["high"] / cmcr_gate_stats["count"],
        }
    if cmcr_context_gate_stats["count"]:
        mean = cmcr_context_gate_stats["sum"] / cmcr_context_gate_stats["count"]
        variance = max(
            cmcr_context_gate_stats["sum_sq"]
            / cmcr_context_gate_stats["count"]
            - mean * mean,
            0.0,
        )
        payload["cmcr_context_gate_activation"] = {
            "mean": mean,
            "std": variance ** 0.5,
            "fraction_below_0_05": cmcr_context_gate_stats["low"]
            / cmcr_context_gate_stats["count"],
            "fraction_above_0_95": cmcr_context_gate_stats["high"]
            / cmcr_context_gate_stats["count"],
        }
    if cmcr_scale_stats["count"]:
        means = [value / cmcr_scale_stats["count"] for value in cmcr_scale_stats["sum"]]
        variances = [
            max(
                cmcr_scale_stats["sum_sq"][index] / cmcr_scale_stats["count"]
                - means[index] * means[index],
                0.0,
            )
            for index in range(2)
        ]
        payload["cmcr_scale_weights"] = {
            "local_1_4_mean": means[0],
            "context_1_8_mean": means[1],
            "local_1_4_std": variances[0] ** 0.5,
            "context_1_8_std": variances[1] ** 0.5,
        }
    residual_scale = getattr(
        getattr(model, "boundary_refinement", None), "residual_scale", None
    )
    if residual_scale is not None:
        raw_scale = float(residual_scale.detach().cpu())
        payload["gated_rezero_scale"] = {
            "raw_alpha": raw_scale,
            "effective_tanh_alpha": float(np.tanh(raw_scale)),
        }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(output_dir / "per_tile_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_confusion_plot(total_hist, output_dir / "confusion_matrix_counts.png")
    save_confusion_plot(
        total_hist, output_dir / "confusion_matrix_row_normalized.png", normalization="row"
    )
    save_confusion_plot(
        total_hist,
        output_dir / "confusion_matrix_column_normalized.png",
        normalization="column",
    )
    save_per_class_plot(result, output_dir / "per_class_metrics.png")

    if args.sample_names_file:
        representative = [line.strip() for line in Path(args.sample_names_file).read_text(encoding="utf-8").splitlines()
                          if line.strip()]
    else:
        representative = choose_representative(rows, args.samples_per_group)
    (output_dir / "representative_names.txt").write_text("\n".join(representative) + "\n", encoding="utf-8")

    ranked = [row for row in rows if row["tile_miou_fg_present"] is not None]
    ranked.sort(key=lambda row: (row["tile_miou_fg_present"], row["name"]))
    hard = [row["name"] for row in ranked[:args.samples_per_group]]
    best = [row["name"] for row in ranked[-args.samples_per_group:]]
    if not args.skip_qualitative:
        save_qualitative(
            model, args.model, args.split, dataset, representative,
            output_dir / "qualitative" / "representative", device,
        )
        save_qualitative(
            model, args.model, args.split, dataset, best,
            output_dir / "qualitative" / "best", device,
        )
        save_qualitative(
            model, args.model, args.split, dataset, hard,
            output_dir / "qualitative" / "hard_cases", device,
        )

    print(f"完成: {output_dir}")
    print(
        f"{args.split} mIoU={result['miou']:.4f}, "
        f"foreground mIoU={result['miou_fg']:.4f}, mF1={result['mf1']:.4f}"
    )


if __name__ == "__main__":
    main()
