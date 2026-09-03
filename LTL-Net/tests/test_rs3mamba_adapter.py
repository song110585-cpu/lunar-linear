from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.rs3mamba_official_adapter import _expand_input_conv


def test_expand_input_conv_preserves_rgb_and_mean_initializes_extra_channels():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel()).view_as(conv.weight))
        conv.bias.copy_(torch.arange(4, dtype=torch.float32))

    expanded = _expand_input_conv(conv, 5)

    assert expanded.in_channels == 5
    torch.testing.assert_close(expanded.weight[:, :3], conv.weight)
    expected = conv.weight.mean(dim=1, keepdim=True)
    torch.testing.assert_close(expanded.weight[:, 3:4], expected)
    torch.testing.assert_close(expanded.weight[:, 4:5], expected)
    torch.testing.assert_close(expanded.bias, conv.bias)


def test_expand_input_conv_returns_same_layer_when_channels_already_match():
    conv = nn.Conv2d(5, 4, kernel_size=3)
    assert _expand_input_conv(conv, 5) is conv
