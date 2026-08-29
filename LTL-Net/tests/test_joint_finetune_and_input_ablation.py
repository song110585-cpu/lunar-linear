import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "datasets"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from MyDataset import apply_channel_mode, channel_mask
from train_module_experiment import (
    CLASS_WEIGHTS,
    ExperimentLoss,
    load_joint_finetune_checkpoint,
    set_joint_finetune_train_mode,
)


def test_channel_modes_zero_only_declared_normalized_channels():
    image = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3) + 1
    expected = {
        "full": (1, 1, 1, 1, 1),
        "wac_only": (1, 0, 0, 0, 0),
        "terrain_only": (0, 1, 1, 1, 1),
        "wac_dem": (1, 1, 0, 0, 0),
    }
    for mode, mask in expected.items():
        result = apply_channel_mode(image, mode)
        assert np.array_equal(channel_mask(mode), np.asarray(mask, dtype=np.float32))
        for index, active in enumerate(mask):
            if active:
                assert np.array_equal(result[index], image[index])
            else:
                assert np.count_nonzero(result[index]) == 0


def test_channel_mode_rejects_unknown_mode_and_wrong_shape():
    with pytest.raises(ValueError, match="channel_mode"):
        channel_mask("invented")
    with pytest.raises(ValueError, match="\(5,H,W\)"):
        apply_channel_mode(np.zeros((4, 8, 8), dtype=np.float32), "full")


def test_zero_lovasz_weight_is_exact_weighted_cross_entropy_control():
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    labels = torch.randint(0, 5, (2, 8, 8))
    criterion = ExperimentLoss("deeplab", boundary_weight=0.0, lovasz_weight=0.0)
    total, parts = criterion(logits, labels, None)
    expected = nn.functional.cross_entropy(
        logits, labels, weight=torch.tensor(CLASS_WEIGHTS)
    )
    assert torch.equal(total, expected)
    assert parts["lovasz_loss"] == 0.0


def test_lovasz_term_is_finite_and_contributes_gradients():
    torch.manual_seed(12)
    logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    labels = torch.randint(0, 5, (2, 8, 8))
    control = ExperimentLoss("deeplab", boundary_weight=0.0, lovasz_weight=0.0)
    enhanced = ExperimentLoss("deeplab", boundary_weight=0.0, lovasz_weight=0.2)
    control_total, _ = control(logits, labels, None)
    total, parts = enhanced(logits, labels, None)
    assert parts["lovasz_loss"] > 0
    assert torch.allclose(
        total.detach(),
        control_total.detach() + 0.2 * total.new_tensor(parts["lovasz_loss"]),
        atol=1e-6,
    )
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


class TinyJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(5, 4, 1), nn.BatchNorm2d(4))
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.segmentation_head = nn.Conv2d(4, 5, 1)
        self.boundary_refinement = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.cmcr = nn.Conv2d(5, 5, 1)


def test_joint_finetune_strictly_loads_and_freezes_encoder_and_batchnorm():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "complete.pth"
        source = TinyJointModel()
        torch.save(source.state_dict(), checkpoint)
        target = TinyJointModel()
        report = load_joint_finetune_checkpoint(
            target, str(checkpoint), torch.device("cpu")
        )
        assert report["frozen_encoder"] is True
        assert all(not parameter.requires_grad for parameter in target.encoder.parameters())
        assert any(parameter.requires_grad for parameter in target.decoder.parameters())
        assert any(parameter.requires_grad for parameter in target.segmentation_head.parameters())
        assert any(parameter.requires_grad for parameter in target.boundary_refinement.parameters())
        assert any(parameter.requires_grad for parameter in target.cmcr.parameters())
        assert all(
            not parameter.requires_grad
            for module in target.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for parameter in module.parameters(recurse=False)
        )
        set_joint_finetune_train_mode(target)
        assert target.training is True
        assert target.encoder.training is False
        assert all(
            module.training is False
            for module in target.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )


def test_joint_finetune_configs_differ_only_by_lovasz_hypothesis_and_identity():
    config_dir = PROJECT_ROOT / "configs"
    control = json.loads(
        (config_dir / "v6_overlap40_joint_finetune_rezero_cmcr_ce_seed1337.json").read_text(
            encoding="utf-8"
        )
    )
    lovasz = json.loads(
        (config_dir / "v6_overlap40_joint_finetune_rezero_cmcr_lovasz02_seed1337.json").read_text(
            encoding="utf-8"
        )
    )
    allowed = {"experiment", "hypothesis", "unique_variable", "lovasz_weight", "run_name"}
    assert set(control) == set(lovasz)
    for key in control:
        if key not in allowed:
            assert control[key] == lovasz[key], key
    assert control["lovasz_weight"] == 0.0
    assert lovasz["lovasz_weight"] == 0.2
    assert control["expected_init_checkpoint_sha256"] == lovasz[
        "expected_init_checkpoint_sha256"
    ]
    assert control["automatic_test_evaluation"] is False


def test_input_ablation_configs_keep_protocol_fixed_and_only_change_channel_mode():
    config_dir = PROJECT_ROOT / "configs"
    modes = ("full", "wac_only", "terrain_only", "wac_dem")
    configs = {
        mode: json.loads(
            (config_dir / f"v6_overlap40_deeplab_input_{mode}_batch4_seed42.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in modes
    }
    allowed = {"experiment", "hypothesis", "unique_variable", "channel_mode", "run_name"}
    control = configs["full"]
    for mode, config in configs.items():
        assert set(config) == set(control)
        for key in control:
            if key not in allowed:
                assert config[key] == control[key], (mode, key)
        assert config["channel_mode"] == mode
        assert config["module"] == "deeplab"
        assert config["automatic_test_evaluation"] is False
