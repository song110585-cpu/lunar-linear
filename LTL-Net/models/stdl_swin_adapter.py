"""Adapter for the historical STDL-Net SwinV2 segmentation architecture."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STDL_MODEL_FILE = REPOSITORY_ROOT / "STDL-Net" / "models" / "swinv2unet.py"


def _load_stdl_module():
    module_name = "lunar_linear_stdl_swinv2unet"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not STDL_MODEL_FILE.is_file():
        raise FileNotFoundError(f"STDL Swin model file not found: {STDL_MODEL_FILE}")
    spec = importlib.util.spec_from_file_location(module_name, STDL_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load STDL Swin model: {STDL_MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_stdl_swinv2(
    variant: str,
    encoder_weights: str | None = "imagenet22k_to_1k",
) -> nn.Module:
    """Build the unmodified STDL architecture with a fixed full-tuning identity."""
    normalized = variant.strip().lower()
    if normalized not in {"small", "base"}:
        raise ValueError(f"unsupported STDL SwinV2 variant: {variant}")
    stdl = _load_stdl_module()
    model = stdl.Swin_LCSRB_DeformablePSP_FPNPAN(
        size=normalized,
        img_size=512,
        num_classes=5,
        in_channels=5,
        pretrained=encoder_weights is not None,
        use_strip_pooling=False,
        use_coord_attention=False,
        use_local_cnn=False,
        use_dem_guided=False,
        use_deep_supervision=False,
    )
    model.experiment_identity = {
        "architecture": "Swin_LCSRB_DeformablePSP_FPNPAN",
        "backbone": f"swinv2_{normalized}",
        "variant": normalized,
        "freeze_stages": 0,
        "img_size": 512,
        "in_channels": 5,
        "pretrained": encoder_weights is not None,
        "pretrained_load_report": getattr(
            model.backbone, "pretrained_load_report", None
        ),
        "disabled_optional_modules": [
            "strip_pooling",
            "coord_attention",
            "local_cnn",
            "dem_guided",
            "deep_supervision",
        ],
    }
    return model
