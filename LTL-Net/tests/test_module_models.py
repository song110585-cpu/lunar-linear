import sys
from pathlib import Path
import csv
import json
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.dynamic_snake import DynamicSnakeConv2d
from models.module_models import DeepLabResNet50, DSConvResNet50, GatedBoundaryResNet50
from train_module_experiment import (
    DATA_METADATA_FILES,
    ExperimentLoss,
    fingerprint_dataset_metadata,
    masks_to_boundaries,
)
from evaluate_segmentation import (
    foreground_binary_confusion,
    summarize_boundary_counts,
    update_boundary_counts,
)
from experiment_artifacts import HISTORY_FIELDS, save_training_history


def test_dynamic_snake_keeps_shape_and_trains_offsets():
    module = DynamicSnakeConv2d(8, 12, kernel_size=5, morph=0)
    x = torch.randn(2, 8, 16, 20, requires_grad=True)
    output = module(x)
    assert output.shape == (2, 12, 16, 20)
    output.mean().backward()
    assert module.offset[0].weight.grad is not None
    assert torch.isfinite(module.offset[0].weight.grad).all()


def test_dsconv_resnet50_output_contract():
    model = DSConvResNet50(encoder_weights=None).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 5, 64, 64))
    assert output.shape == (1, 5, 64, 64)


def test_deeplab_control_output_contract():
    model = DeepLabResNet50(encoder_weights=None).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 5, 64, 64))
    assert output.shape == (1, 5, 64, 64)


def test_gated_boundary_output_contract():
    model = GatedBoundaryResNet50(encoder_weights=None).eval()
    with torch.no_grad():
        output = model.forward_with_aux(torch.randn(1, 5, 64, 64))
    assert output["logits"].shape == (1, 5, 64, 64)
    assert output["boundary_logits"].shape == (1, 1, 32, 32)


def test_boundary_target_marks_foreground_transitions_only():
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 4:12, 7:9] = 1
    target = masks_to_boundaries(labels, (8, 8))
    assert target.shape == (1, 1, 8, 8)
    assert 0 < target.sum() < target.numel()


def test_zero_boundary_weight_keeps_structure_but_removes_auxiliary_loss():
    criterion = ExperimentLoss("gated_boundary", boundary_weight=0.0)
    logits = torch.randn(1, 5, 16, 16, requires_grad=True)
    boundary_logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 4:12, 7:9] = 1
    total, parts = criterion(logits, labels, boundary_logits)
    semantic = criterion.semantic(logits, labels)
    assert torch.allclose(total, semantic)
    assert parts["boundary_loss"] > 0


def test_foreground_binary_confusion_collapses_multiclass_errors():
    hist = np.asarray(
        [
            [10, 2, 3],
            [4, 5, 1],
            [6, 2, 7],
        ]
    )
    assert foreground_binary_confusion(hist).tolist() == [[10, 5], [10, 15]]


def test_boundary_tolerance_accepts_one_pixel_shift():
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    prediction = torch.zeros_like(target)
    target[:, 4:12, 7:9] = 1
    prediction[:, 4:12, 8:10] = 1
    counts = {}
    update_boundary_counts(counts, prediction, target, (1,))
    result = summarize_boundary_counts(counts)["1"]
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_dataset_metadata_fingerprint_is_complete():
    with tempfile.TemporaryDirectory() as output_dir:
        root = Path(output_dir)
        for index, name in enumerate(DATA_METADATA_FILES):
            (root / name).write_text(f"fixture-{index}", encoding="utf-8")
        fingerprints = fingerprint_dataset_metadata(root)
    assert set(fingerprints) == set(DATA_METADATA_FILES)
    assert all(len(value) == 64 for value in fingerprints.values())


def test_control_configs_are_val_only_and_single_variable():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_deeplab_batch2_control.json",
        "v6_overlap40_gated_no_boundary_control.json",
    )
    configs = [json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names]
    assert [config["module"] for config in configs] == ["deeplab", "gated_boundary"]
    assert all(config["automatic_test_evaluation"] is False for config in configs)
    assert all(config["selection_metric"] == "val_mIoU_fg" for config in configs)
    assert all(config["batch_size"] == 2 and config["accum_steps"] == 2 for config in configs)
    assert all(config["boundary_weight"] == 0.0 for config in configs)
    assert configs[0]["expected_metadata_sha256"] == configs[1]["expected_metadata_sha256"]


def test_history_writer_preserves_extra_metrics():
    row = {field: 0.0 for field in HISTORY_FIELDS}
    row["epoch"] = 1
    row["val_iou_per_class"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    with tempfile.TemporaryDirectory() as output_dir:
        csv_path = save_training_history([row], output_dir)
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            saved = next(csv.DictReader(handle))
        assert "val_iou_per_class" in saved
        assert saved["val_iou_per_class"] == "[0.1, 0.2, 0.3, 0.4, 0.5]"
