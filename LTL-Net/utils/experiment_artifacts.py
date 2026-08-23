"""训练过程表格与曲线的统一保存工具。"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_miou",
    "train_miou_fg",
    "train_mf1",
    "val_loss",
    "val_accuracy",
    "val_miou",
    "val_miou_fg",
    "val_mf1",
    "is_best",
]


def _plot_lines(history, series, ylabel, output_path):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for key, label in series:
        ax.plot(epochs, [row[key] for row in history], label=label, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_training_history(history, record_path):
    """每轮覆盖写入，训练中断时也能保留已完成 epoch 的记录。"""
    os.makedirs(record_path, exist_ok=True)
    csv_path = os.path.join(record_path, "history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    json_path = os.path.join(record_path, "history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if history:
        _plot_lines(
            history,
            [("train_loss", "Train loss"), ("val_loss", "Validation loss")],
            "Loss",
            os.path.join(record_path, "loss_curve.png"),
        )
        _plot_lines(
            history,
            [("train_miou", "Train mIoU"), ("val_miou", "Validation mIoU")],
            "mIoU (all classes)",
            os.path.join(record_path, "miou_curve.png"),
        )
        _plot_lines(
            history,
            [("train_mf1", "Train mF1"), ("val_mf1", "Validation mF1")],
            "mF1 (all classes)",
            os.path.join(record_path, "mf1_curve.png"),
        )
    return csv_path
