"""Adapter for the official RS3Mamba implementation used as a comparison model.

The upstream source is intentionally not copied into this repository.  Callers
must provide a checkout of https://github.com/sstary/SSRS at the pinned commit
recorded by the experiment config.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


OFFICIAL_BACKBONE = "resnet18.fb_swsl_ig1b_ft_in1k"


def _official_rs3mamba_root(source_dir: str | Path) -> Path:
    source = Path(source_dir).expanduser().resolve()
    candidates = (source, source / "RS3Mamba")
    for candidate in candidates:
        if (candidate / "model" / "RS3Mamba.py").is_file():
            return candidate
    raise FileNotFoundError(
        "找不到官方 RS3Mamba 源码；--rs3mamba-source-dir 应指向 SSRS 仓库"
        f"或其 RS3Mamba 子目录: {source}"
    )


def _import_official_model(source_dir: str | Path):
    root = _official_rs3mamba_root(source_dir)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("model.RS3Mamba")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "导入官方 RS3Mamba 失败。请安装其 CUDA 依赖 mamba-ssm、"
            "causal-conv1d、einops、monai 和 timm。"
        ) from exc
    return module.RS3Mamba


def _checkpoint_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"预训练权重不是 state_dict: {type(payload).__name__}")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def _load_matching_state(
    module: nn.Module, checkpoint_path: str | Path, *, skip_prefixes: tuple[str, ...] = ()
) -> dict[str, Any]:
    source = _checkpoint_state(checkpoint_path)
    target = module.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and not key.startswith(skip_prefixes)
    }
    module.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "source_tensors": len(source),
        "skipped_or_unmatched_tensors": len(source) - len(matched),
    }


def _load_vmamba_state(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    source = _checkpoint_state(checkpoint_path)
    target = model.state_dict()
    skipped = {
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "patch_embed.norm.weight",
        "patch_embed.norm.bias",
    }
    matched: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for key, value in source.items():
        if key in skipped:
            continue
        mapped = f"vssm_encoder.{key}"
        match = re.search(r"layers\.(\d+)\.downsample", mapped)
        if match:
            stage = match.group(1)
            mapped = mapped.replace(
                f"layers.{stage}.downsample", f"downsamples.{stage}"
            )
        if mapped not in target:
            continue
        if target[mapped].shape != value.shape:
            shape_mismatches.append(mapped)
            continue
        matched[mapped] = value
    if shape_mismatches:
        raise RuntimeError(f"VMamba 权重形状不匹配: {shape_mismatches[:5]}")
    model.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "source_tensors": len(source),
        "skipped_or_unmatched_tensors": len(source) - len(matched),
    }


def _expand_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    if conv.groups != 1 or conv.in_channels != 3 or in_channels < 3:
        raise ValueError(
            f"只支持把普通 RGB Conv2d 扩展到 >=3 通道，实际为 "
            f"in_channels={conv.in_channels}, groups={conv.groups}"
        )
    expanded = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        expanded.weight[:, :3].copy_(conv.weight)
        channel_mean = conv.weight.mean(dim=1, keepdim=True)
        expanded.weight[:, 3:].copy_(channel_mean.expand(-1, in_channels - 3, -1, -1))
        if conv.bias is not None:
            expanded.bias.copy_(conv.bias)
    return expanded


def build_rs3mamba(
    source_dir: str | Path,
    *,
    in_channels: int = 5,
    classes: int = 5,
    resnet_checkpoint: str | Path | None = None,
    vmamba_checkpoint: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the official network and adapt both RGB stems to five channels."""
    official_model = _import_official_model(source_dir)
    model = official_model(
        backbone_name=OFFICIAL_BACKBONE,
        pretrained=False,
        num_classes=classes,
    )
    loading: dict[str, Any] = {}
    if resnet_checkpoint is not None:
        loading["resnet18"] = _load_matching_state(
            model.backbone, resnet_checkpoint, skip_prefixes=("fc.",)
        )
    if vmamba_checkpoint is not None:
        loading["vmamba_tiny"] = _load_vmamba_state(model, vmamba_checkpoint)

    cnn_stem = _expand_input_conv(model.conv1, in_channels)
    model.conv1 = cnn_stem
    model.backbone.conv1 = cnn_stem
    model.stem[0] = _expand_input_conv(model.stem[0], in_channels)
    loading["input_adaptation"] = {
        "from_channels": 3,
        "to_channels": in_channels,
        "adapted_stems": 2,
        "new_channel_initialization": "mean_of_rgb_kernels",
    }
    return model, loading
