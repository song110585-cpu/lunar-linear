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
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics
from MyDataset import MyDataset
from models.dlinknet import DLinkNet
from models.ltl_net import LTLNet


CLASS_NAMES = ["Background", "WR", "Rille", "Fault", "Graben"]
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
    if not hasattr(smp, args.model):
        raise ValueError(f"segmentation_models_pytorch 中不存在模型: {args.model}")
    return getattr(smp, args.model)(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=5,
        classes=5,
    )


def save_confusion_plot(hist, output_path, normalized):
    values = hist.astype(np.float64)
    if normalized:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Row-normalized confusion matrix" if normalized else "Confusion matrix (pixels)")
    threshold = values.max() * 0.55 if values.size else 0
    for row in range(5):
        for col in range(5):
            text_value = f"{values[row, col]:.3f}" if normalized else f"{int(values[row, col]):,}"
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


def save_qualitative(model, dataset, names, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_to_index = {Path(name).stem: index for index, name in enumerate(dataset.filenames)}
    model.eval()
    for name in names:
        if name not in stem_to_index:
            print(f"[warning] 定性样本不在 test 集: {name}")
            continue
        image, target, _ = dataset[stem_to_index[name]]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
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
    parser.add_argument("--data-dir", required=True, help="含 test/image 与 test/mask 的数据集根目录")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
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
    dataset = MyDataset(str(data_root / "test" / "image"), str(data_root / "test" / "mask"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([0.15, 1.0, 2.73, 1.98, 2.12], device=device)
    )

    total_hist = np.zeros((5, 5), dtype=np.int64)
    losses = []
    rows = []
    model.eval()
    with torch.no_grad():
        for images, targets, names in tqdm(loader, desc="Evaluate test", unit="batch"):
            images = images.to(device)
            targets_device = targets.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                batch_loss = loss_fn(logits, targets_device)
            losses.append(float(batch_loss.item()))
            preds = logits.argmax(dim=1).cpu()
            for index, name in enumerate(names):
                hist = metrics.multiclass_confusion(preds[index], targets[index], 5).numpy()
                total_hist += hist
                target_np = targets[index].numpy()
                rows.append({
                    "name": name,
                    "foreground_ratio": float(np.mean(target_np > 0)),
                    "present_classes": ",".join(str(v) for v in np.unique(target_np[target_np > 0])),
                    "tile_miou_fg_present": tile_foreground_iou(hist),
                })

    result = metrics.metrics_from_hist(torch.from_numpy(total_hist))
    result["loss"] = float(np.mean(losses))
    result["miou_fg"] = float(np.mean(result["iou_per_class"][1:]))
    payload = {
        "model": args.model,
        "encoder": args.encoder,
        "detail_channels": args.detail_channels if args.model.lower() in ("ltl", "ltlnet", "ltl-net") else None,
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "class_names": CLASS_NAMES,
        "confusion_matrix": total_hist.tolist(),
        "test": result,
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(output_dir / "per_tile_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_confusion_plot(total_hist, output_dir / "confusion_matrix_counts.png", normalized=False)
    save_confusion_plot(total_hist, output_dir / "confusion_matrix_normalized.png", normalized=True)
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
    save_qualitative(model, dataset, representative, output_dir / "qualitative" / "representative", device)
    save_qualitative(model, dataset, best, output_dir / "qualitative" / "best", device)
    save_qualitative(model, dataset, hard, output_dir / "qualitative" / "hard_cases", device)

    print(f"完成: {output_dir}")
    print(f"test mIoU={result['miou']:.4f}, foreground mIoU={result['miou_fg']:.4f}, mF1={result['mf1']:.4f}")


if __name__ == "__main__":
    main()
