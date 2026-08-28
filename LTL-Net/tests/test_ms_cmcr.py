import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.module_models import (
    GatedReZeroMSCMCRResNet50,
    GatedReZeroResNet50,
    build_module_model,
)
from train_module_experiment import load_frozen_cmcr_base
from run_autodl_frozen_rezero_cmcr import build_frozen_experiment_models


def test_ms_cmcr_starts_as_exact_rezero_parent():
    torch.manual_seed(2026)
    parent = GatedReZeroResNet50(encoder_weights=None).eval()
    torch.manual_seed(2026)
    enhanced = GatedReZeroMSCMCRResNet50(encoder_weights=None).eval()
    inputs = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        parent_outputs = parent.forward_with_aux(inputs)
        enhanced_outputs = enhanced.forward_with_aux(inputs)
    assert torch.equal(parent_outputs["logits"], enhanced_outputs["logits"])
    assert torch.equal(
        parent_outputs["boundary_logits"], enhanced_outputs["boundary_logits"]
    )
    assert torch.count_nonzero(enhanced.cmcr.residual_head.weight) == 0


def test_ms_cmcr_scale_weights_are_normalized_and_head_trains():
    model = GatedReZeroMSCMCRResNet50(encoder_weights=None).train()
    captured = []
    hook = model.cmcr.scale_gate.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    inputs = torch.randn(2, 5, 64, 64)
    outputs = model.forward_with_aux(inputs)
    hook.remove()
    assert outputs["logits"].shape == (2, 5, 64, 64)
    assert len(captured) == 1
    assert captured[0].shape[1] == 2
    assert torch.allclose(captured[0].sum(dim=1), torch.ones_like(captured[0][:, 0]))
    outputs["logits"].mean().backward()
    gradient = model.cmcr.residual_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_ms_cmcr_is_registered_as_a_distinct_model():
    model = build_module_model("gated_rezero_ms_cmcr", encoder_weights=None)
    assert isinstance(model, GatedReZeroMSCMCRResNet50)
    assert model.experiment_identity["cross_modal_scales"] == (4, 8)


def test_ms_cmcr_frozen_loader_leaves_only_pyramid_trainable():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "gated_rezero.pth"
        parent = GatedReZeroResNet50(encoder_weights=None)
        torch.save(parent.state_dict(), checkpoint)
        enhanced = GatedReZeroMSCMCRResNet50(encoder_weights=None)
        report = load_frozen_cmcr_base(
            enhanced, str(checkpoint), torch.device("cpu")
        )
        assert report["missing_keys"]
        assert all(key.startswith("cmcr.") for key in report["missing_keys"])
        assert all(
            parameter.requires_grad == name.startswith("cmcr.")
            for name, parameter in enhanced.named_parameters()
        )
        assert sum(parameter.numel() for parameter in enhanced.cmcr.parameters()) == 92663


def test_ms_cmcr_seed42_config_binds_validated_parent_and_protocol():
    import json

    path = (
        PROJECT_ROOT
        / "configs/v6_overlap40_frozen_gated_rezero_ms_cmcr_batch4_seed42.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["module"] == "gated_rezero_ms_cmcr"
    assert config["cross_modal_scales"] == [4, 8]
    assert config["scale_fusion"] == "spatial_softmax"
    assert config["expected_trainable_parameter_count"] == 92663
    assert config["expected_init_checkpoint_sha256"] == (
        "d6fd9b1a940d707ff64fe05383e9326420a7a586acf0182cb38c7f5a776b0ed4"
    )
    assert config["learning_rate"] == 1e-4
    assert config["batch_size"] == 4 and config["accum_steps"] == 1
    assert config["automatic_test_evaluation"] is False


def test_autodl_runner_builds_the_module_named_by_config():
    config = {
        "parent_model": "gated_rezero",
        "module": "gated_rezero_ms_cmcr",
    }
    parent, enhanced = build_frozen_experiment_models(config, build_module_model)
    assert isinstance(parent, GatedReZeroResNet50)
    assert isinstance(enhanced, GatedReZeroMSCMCRResNet50)
    assert sum(parameter.numel() for parameter in enhanced.cmcr.parameters()) == 92663
