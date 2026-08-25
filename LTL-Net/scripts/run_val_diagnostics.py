"""Batch-run reproducible validation diagnostics for overlap40 checkpoints.

Edit ``DEFAULT_DATA_DIR`` below, or pass ``--data-dir`` on the command line.
The script delegates model evaluation to ``evaluate_segmentation.py`` and then
creates one comparison CSV/JSON across all successful experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(r"E:\月球_dataset\dataset\dataset_v6_random811_overlap40")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "v6_overlap40"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "val_diagnostics"
CLASS_NAMES = ("background", "wr", "rille", "fault", "graben")


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    checkpoint: Path


EXPERIMENTS = (
    Experiment(
        "deeplab_batch2_control",
        "deeplab",
        Path("deeplab_resnet50_batch2_control/best_model.pth"),
    ),
    Experiment("dsconv", "dsconv", Path("M2/DSConv/best_model.pth")),
    Experiment(
        "gated_with_boundary_loss",
        "gated_boundary",
        Path("M3/gated/best_model.pth"),
    ),
    Experiment(
        "gated_without_boundary_loss",
        "gated_boundary",
        Path("gated_boundary_resnet50_no_boundary/best_model.pth"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量分析 overlap40 模型的 Val 混淆矩阵、边界指标和错误图"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[experiment.name for experiment in EXPERIMENTS],
        help="只运行指定实验；默认运行全部四个",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="显式启用评价 AMP；默认 FP32。若出现 NaN/Inf 请勿使用",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="跳过推理前的 Val TIFF 完整读取检查（不推荐）",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="即使 evaluation_metrics.json 已存在也重新运行",
    )
    parser.add_argument("--list", action="store_true", help="列出实验和 checkpoint 后退出")
    return parser.parse_args()


def selected_experiments(names: list[str] | None) -> list[Experiment]:
    if not names:
        return list(EXPERIMENTS)
    selected = set(names)
    return [experiment for experiment in EXPERIMENTS if experiment.name in selected]


def validate_data_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    required = (data_dir / "val" / "image", data_dir / "val" / "mask")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Val 数据目录不存在，请修改 DEFAULT_DATA_DIR 或传入 --data-dir：\n"
            + "\n".join(missing)
        )
    return data_dir


def check_tiff_integrity(data_dir: Path, split: str = "val") -> list[tuple[Path, str]]:
    """Read every TIFF once and return all unreadable files with error messages."""
    import rasterio

    files: list[Path] = []
    for kind in ("image", "mask"):
        directory = data_dir / split / kind
        files.extend(sorted((*directory.glob("*.tif"), *directory.glob("*.tiff"))))
    failures: list[tuple[Path, str]] = []
    print(f"检查 {split} TIFF 完整性，共 {len(files)} 个文件...", flush=True)
    for index, path in enumerate(files, start=1):
        try:
            with rasterio.open(path) as source:
                source.read()
        except Exception as error:  # rasterio wraps several GDAL exception types
            failures.append((path, f"{type(error).__name__}: {error}"))
        if index % 50 == 0 or index == len(files):
            print(f"  已检查 {index}/{len(files)}", flush=True)
    return failures


def build_command(
    experiment: Experiment,
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    sample_names_file: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_segmentation.py"),
        "--model",
        experiment.model,
        "--encoder",
        "resnet50",
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--split",
        "val",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.max_steps:
        command.extend(("--max-steps", str(args.max_steps)))
    if args.skip_qualitative:
        command.append("--skip-qualitative")
    if args.amp:
        command.append("--amp")
    if sample_names_file is not None and sample_names_file.is_file():
        command.extend(("--sample-names-file", str(sample_names_file)))
    return command


def flatten_evaluation(name: str, payload: dict) -> dict:
    metrics = payload["metrics"]
    row = {
        "experiment": name,
        "model": payload["model"],
        "miou": metrics["miou"],
        "miou_fg": metrics["miou_fg"],
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
    }
    for metric_name in ("iou", "precision", "recall", "f1"):
        values = metrics[f"{metric_name}_per_class"]
        for class_name, value in zip(CLASS_NAMES, values):
            row[f"{metric_name}_{class_name}"] = value
    binary = payload["foreground_binary_confusion"]
    row.update(
        {
            "foreground_tn": binary[0][0],
            "foreground_fp": binary[0][1],
            "foreground_fn": binary[1][0],
            "foreground_tp": binary[1][1],
        }
    )
    for tolerance, values in payload.get("foreground_boundary_metrics", {}).items():
        row[f"boundary_f1_t{tolerance}"] = values["f1"]
        row[f"boundary_precision_t{tolerance}"] = values["precision"]
        row[f"boundary_recall_t{tolerance}"] = values["recall"]
    if "gated_boundary_head" in payload:
        row["boundary_head_f1"] = payload["gated_boundary_head"]["f1"]
    if "gate_activation" in payload:
        row["gate_mean"] = payload["gate_activation"]["mean"]
        row["gate_std"] = payload["gate_activation"]["std"]
    return row


def write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_root / "comparison_summary.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    experiments = selected_experiments(args.only)
    results_root = args.results_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.list:
        for experiment in experiments:
            print(f"{experiment.name}: {results_root / experiment.checkpoint}")
        return 0

    data_dir = validate_data_dir(args.data_dir)
    if not args.skip_data_check:
        corrupt_files = check_tiff_integrity(data_dir)
        if corrupt_files:
            details = "\n".join(f"{path}\n  {error}" for path, error in corrupt_files)
            raise RuntimeError(
                f"发现 {len(corrupt_files)} 个无法完整读取的 Val TIFF；请从原始数据副本替换后再运行：\n"
                + details
            )
        print("Val TIFF 完整性检查通过")
    missing_checkpoints = [
        results_root / experiment.checkpoint
        for experiment in experiments
        if not (results_root / experiment.checkpoint).is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "缺少 checkpoint：\n" + "\n".join(str(path) for path in missing_checkpoints)
        )

    rows: list[dict] = []
    failures: list[str] = []
    shared_names_file: Path | None = None
    for experiment in experiments:
        checkpoint = results_root / experiment.checkpoint
        output_dir = output_root / experiment.name
        metrics_file = output_dir / "evaluation_metrics.json"
        if args.rerun or not metrics_file.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(
                experiment,
                checkpoint,
                data_dir,
                output_dir,
                args,
                shared_names_file,
            )
            print(f"\n=== {experiment.name} ===", flush=True)
            print(subprocess.list2cmdline(command), flush=True)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if completed.returncode != 0:
                failures.append(experiment.name)
                continue
        else:
            print(f"复用已有结果: {metrics_file}")

        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(flatten_evaluation(experiment.name, payload))
        candidate_names = output_dir / "representative_names.txt"
        if shared_names_file is None and candidate_names.is_file():
            shared_names_file = candidate_names
        write_summary(rows, output_root)

    print(f"\n汇总: {output_root / 'comparison_summary.csv'}")
    if failures:
        print("失败实验: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
