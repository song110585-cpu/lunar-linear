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
from models.module_models import (
    DeepLabCMCRResNet50,
    DeepLabFECResNet50,
    DeepLabResNet50,
    DSConvResNet50,
    GatedBoundaryResNet50,
    GatedCMCRResNet50,
    GatedFECResNet50,
    build_module_model,
)
from train_module_experiment import (
    DATA_METADATA_FILES,
    ExperimentLoss,
    load_frozen_cmcr_base,
    fingerprint_dataset_metadata,
    masks_to_boundaries,
)
from evaluate_segmentation import (
    foreground_binary_confusion,
    summarize_boundary_counts,
    update_boundary_counts,
)
from experiment_artifacts import HISTORY_FIELDS, save_training_history
from run_val_diagnostics import (
    check_tiff_integrity,
    discover_checkpoint,
    discover_model_type,
    flatten_evaluation,
    selected_experiments,
)


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


def test_deeplab_cmcr_starts_as_exact_deeplab_and_trains_residual_head():
    torch.manual_seed(2026)
    base = DeepLabResNet50(encoder_weights=None).eval()
    torch.manual_seed(2026)
    enhanced = DeepLabCMCRResNet50(encoder_weights=None).eval()
    inputs = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        assert torch.equal(base(inputs), enhanced(inputs))
    loss = enhanced(inputs)[..., 8:56, 8:56].mean()
    loss.backward()
    gradient = enhanced.cmcr.residual_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_gated_cmcr_starts_as_exact_gated_model_and_trains_residual_head():
    torch.manual_seed(2026)
    base = GatedBoundaryResNet50(encoder_weights=None).eval()
    torch.manual_seed(2026)
    enhanced = GatedCMCRResNet50(encoder_weights=None).eval()
    inputs = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        base_outputs = base.forward_with_aux(inputs)
        enhanced_outputs = enhanced.forward_with_aux(inputs)
    assert torch.equal(base_outputs["logits"], enhanced_outputs["logits"])
    assert torch.equal(
        base_outputs["boundary_logits"], enhanced_outputs["boundary_logits"]
    )
    assert torch.count_nonzero(enhanced.cmcr.residual_head.weight) == 0
    loss = enhanced(inputs)[..., 8:56, 8:56].mean()
    loss.backward()
    gradient = enhanced.cmcr.residual_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_frozen_cmcr_loader_restores_parent_and_freezes_everything_else():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "gated.pth"
        torch.manual_seed(2026)
        parent = GatedBoundaryResNet50(encoder_weights=None).eval()
        torch.save(parent.state_dict(), checkpoint)
        torch.manual_seed(7)
        enhanced = GatedCMCRResNet50(encoder_weights=None).eval()
        report = load_frozen_cmcr_base(enhanced, str(checkpoint), torch.device("cpu"))
        inputs = torch.randn(1, 5, 64, 64)
        with torch.no_grad():
            assert torch.equal(parent(inputs), enhanced(inputs))
        assert report["missing_keys"]
        assert all(key.startswith("cmcr.") for key in report["missing_keys"])
        assert all(
            parameter.requires_grad == name.startswith("cmcr.")
            for name, parameter in enhanced.named_parameters()
        )


def test_fec_models_start_as_exact_parents_and_receive_auxiliary_gradients():
    pairs = (
        (DeepLabResNet50, DeepLabFECResNet50),
        (GatedBoundaryResNet50, GatedFECResNet50),
    )
    inputs = torch.randn(1, 5, 64, 64)
    labels = torch.randint(0, 5, (1, 64, 64))
    for parent_type, enhanced_type in pairs:
        torch.manual_seed(2026)
        parent = parent_type(encoder_weights=None).eval()
        torch.manual_seed(2026)
        enhanced = enhanced_type(encoder_weights=None).eval()
        with torch.no_grad():
            assert torch.equal(parent(inputs), enhanced(inputs))
        outputs = enhanced.forward_with_aux(inputs)
        criterion = ExperimentLoss(
            "gated_fec" if parent_type is GatedBoundaryResNet50 else "deeplab_fec",
            boundary_weight=0.0,
            foreground_weight=0.1,
        )
        loss, parts = criterion(
            outputs["logits"],
            labels,
            outputs.get("boundary_logits"),
            outputs["foreground_logits"],
        )
        loss.backward()
        assert parts["foreground_loss"] > 0
        assert torch.isfinite(enhanced.fec.calibration_strength.grad)
        assert enhanced.fec.evidence_head[-1].weight.grad is not None
        assert torch.isfinite(enhanced.fec.evidence_head[-1].weight.grad).all()


def test_val_diagnostic_experiment_filter_keeps_declared_order():
    selected = selected_experiments(["gated_without_boundary_loss", "dsconv"])
    assert [experiment.name for experiment in selected] == [
        "dsconv",
        "gated_without_boundary_loss",
    ]


def test_flatten_evaluation_preserves_comparison_fields():
    payload = {
        "model": "deeplab",
        "metrics": {
            "miou": 0.7,
            "miou_fg": 0.6,
            "accuracy": 0.9,
            "loss": 0.2,
            "iou_per_class": [0.9, 0.5, 0.6, 0.7, 0.6],
            "precision_per_class": [0.9] * 5,
            "recall_per_class": [0.8] * 5,
            "f1_per_class": [0.85] * 5,
        },
        "foreground_binary_confusion": [[10, 2], [3, 8]],
        "foreground_boundary_metrics": {
            "2": {"precision": 0.7, "recall": 0.8, "f1": 0.746},
        },
    }
    row = flatten_evaluation("control", payload)
    assert row["experiment"] == "control"
    assert row["miou_fg"] == 0.6
    assert row["iou_fault"] == 0.7
    assert row["foreground_fp"] == 2
    assert row["boundary_f1_t2"] == 0.746


def test_tiff_integrity_check_reports_corrupt_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for kind in ("image", "mask"):
            (root / "val" / kind).mkdir(parents=True)
        corrupt = root / "val" / "image" / "broken.tif"
        corrupt.write_bytes(b"not a tiff")
        failures = check_tiff_integrity(root)
    assert len(failures) == 1
    assert failures[0][0] == corrupt


def test_single_model_folder_discovers_checkpoint_and_model_type():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        checkpoint = folder / "best_model.pth"
        checkpoint.write_bytes(b"checkpoint placeholder")
        (folder / "config.json").write_text(
            json.dumps({"module": "gated_boundary"}), encoding="utf-8"
        )
        assert discover_checkpoint(folder) == checkpoint
        assert discover_model_type(folder) == "gated_boundary"


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


def test_standalone_and_combined_cmcr_configs_share_batch4_protocol():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_deeplab_cmcr_batch4_seed42.json",
        "v6_overlap40_gated_cmcr_batch4_seed42.json",
    )
    configs = [json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names]
    assert [config["module"] for config in configs] == ["deeplab_cmcr", "gated_cmcr"]
    assert [config["seed"] for config in configs] == [42, 42]
    assert all(config["automatic_test_evaluation"] is False for config in configs)
    assert all(config["selection_metric"] == "val_mIoU_fg" for config in configs)
    assert all(config["batch_size"] == 4 and config["accum_steps"] == 1 for config in configs)
    assert all(config["boundary_weight"] == 0.0 for config in configs)
    assert configs[0]["expected_metadata_sha256"] == configs[1]["expected_metadata_sha256"]


def test_frozen_cmcr_config_has_epoch_zero_guard_protocol():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_frozen_gated_cmcr_batch4_seed42.json",
        "v6_overlap40_frozen_gated_cmcr_batch4_seed1337.json",
        "v6_overlap40_frozen_gated_cmcr_batch4_seed3407.json",
    )
    configs = [json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names]
    assert [config["seed"] for config in configs] == [42, 1337, 3407]
    for config in configs:
        assert config["module"] == "gated_cmcr"
        assert config["freeze_base"] is True
        assert config["epochs"] == 40
        assert config["early_stopping_patience"] == 8
        assert config["batch_size"] == 4 and config["accum_steps"] == 1
        assert config["boundary_weight"] == 0.0
        assert len(config["expected_init_checkpoint_sha256"]) == 64
        assert config["automatic_test_evaluation"] is False
    controlled_fields = {
        "experiment",
        "hypothesis",
        "unique_variable",
        "seed",
        "run_name",
    }
    for key in configs[0]:
        if key not in controlled_fields:
            assert all(config[key] == configs[0][key] for config in configs[1:]), key


def test_full_pipeline_seed1337_frozen_config_binds_new_parent():
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "v6_overlap40_frozen_gatedA1337_cmcr_batch4_seed1337.json"
        ).read_text(encoding="utf-8")
    )
    assert config["module"] == "gated_cmcr"
    assert config["seed"] == 1337
    assert config["freeze_base"] is True
    assert config["batch_size"] == 4 and config["accum_steps"] == 1
    assert config["early_stopping_patience"] == 8
    assert config["expected_init_checkpoint_sha256"] == (
        "0d0bca4e7358e959efe0d09fab88c43b1a9f651ef5861f3c4a385ab7168a43e3"
    )
    assert "gatedA1337" in config["run_name"]
    assert config["automatic_test_evaluation"] is False


def test_deeplab_and_gated_seed1337_are_structure_only_controls():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_deeplab_batch4_seed1337.json",
        "v6_overlap40_gated_no_boundary_batch4_seed1337.json",
    )
    configs = [json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names]
    assert set(configs[0]) == set(configs[1])
    assert [config["module"] for config in configs] == ["deeplab", "gated_boundary"]
    controlled_fields = {
        "experiment",
        "hypothesis",
        "unique_variable",
        "module",
        "run_name",
    }
    for key in configs[0]:
        if key not in controlled_fields:
            assert configs[0][key] == configs[1][key], key
    assert all(config["seed"] == 1337 for config in configs)
    assert all(config["batch_size"] == 4 and config["accum_steps"] == 1 for config in configs)
    assert all(config["automatic_test_evaluation"] is False for config in configs)


def test_stdl_swin_configs_differ_only_by_capacity_identity():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_stdl_swinv2_small_full_seed42.json",
        "v6_overlap40_stdl_swinv2_base_full_seed42.json",
    )
    small, base = [
        json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names
    ]
    allowed_differences = {
        "experiment",
        "hypothesis",
        "comparison_scope",
        "module",
        "encoder",
        "pretrain_filename",
        "expected_pretrain_sha256",
        "run_name",
    }
    assert set(small) == set(base)
    for key in small:
        if key not in allowed_differences:
            assert small[key] == base[key], key
    assert small["module"] == "stdl_swinv2_small"
    assert base["module"] == "stdl_swinv2_base"
    assert small["freeze_stages"] == base["freeze_stages"] == 0
    assert small["automatic_test_evaluation"] is False
    assert base["automatic_test_evaluation"] is False


def test_stdl_swin_small_identity_without_pretraining():
    model = build_module_model("stdl_swinv2_small", encoder_weights=None)
    identity = model.experiment_identity
    assert identity["variant"] == "small"
    assert identity["backbone"] == "swinv2_small"
    assert identity["freeze_stages"] == 0
    assert identity["in_channels"] == 5
    assert identity["pretrained"] is False
    assert model.backbone.patch_embed.proj.in_channels == 5
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_fec_configs_share_the_controlled_batch4_protocol():
    config_dir = PROJECT_ROOT / "configs"
    names = (
        "v6_overlap40_deeplab_fec_batch4_seed42.json",
        "v6_overlap40_gated_fec_batch4_seed42.json",
    )
    configs = [json.loads((config_dir / name).read_text(encoding="utf-8")) for name in names]
    assert [config["module"] for config in configs] == ["deeplab_fec", "gated_fec"]
    assert all(config["seed"] == 42 for config in configs)
    assert all(config["batch_size"] == 4 and config["accum_steps"] == 1 for config in configs)
    assert all(config["boundary_weight"] == 0.0 for config in configs)
    assert all(config["foreground_weight"] == 0.1 for config in configs)
    assert all(config["selection_metric"] == "val_mIoU_fg" for config in configs)
    assert all(config["automatic_test_evaluation"] is False for config in configs)
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
