import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.dynamic_snake import DynamicSnakeConv2d
from models.module_models import DSConvResNet50, GatedBoundaryResNet50
from train_module_experiment import masks_to_boundaries


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
