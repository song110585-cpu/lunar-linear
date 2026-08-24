"""Controlled ResNet50 module experiments built on one DeepLabV3+ base."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

from .dynamic_snake import DynamicLineRefinement


def _conv_norm_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _DeepLabResNet50Base(nn.Module):
    """Shared encoder/decoder/head so only the experimental module varies."""

    def __init__(self, encoder_weights: str | None = "imagenet", classes: int = 5) -> None:
        super().__init__()
        base = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            in_channels=5,
            classes=classes,
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

    def _features(self, x: torch.Tensor):
        features = self.encoder(x)
        return features, self.decoder(features)


class DeepLabResNet50(_DeepLabResNet50Base):
    """Unmodified DeepLabV3+-ResNet50 control for the module training protocol."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, decoded = self._features(x)
        return self.segmentation_head(decoded)


class DSConvResNet50(_DeepLabResNet50Base):
    """DeepLabV3+-ResNet50 plus residual Dynamic Snake refinement."""

    def __init__(
        self,
        encoder_weights: str | None = "imagenet",
        classes: int = 5,
        hidden_channels: int = 64,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
    ) -> None:
        super().__init__(encoder_weights=encoder_weights, classes=classes)
        self.refinement = DynamicLineRefinement(
            channels=256,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            extend_scope=extend_scope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, decoded = self._features(x)
        return self.segmentation_head(self.refinement(decoded))


class SemanticGatedBoundaryRefinement(nn.Module):
    """Use deep semantics to gate noisy 1/2-resolution shape features."""

    def __init__(
        self,
        detail_channels: int = 64,
        semantic_channels: int = 256,
        shape_channels: int = 32,
    ) -> None:
        super().__init__()
        self.detail_projection = _conv_norm_relu(detail_channels, shape_channels, 3)
        self.semantic_projection = _conv_norm_relu(semantic_channels, shape_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * shape_channels, shape_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.shape_refinement = nn.Sequential(
            _conv_norm_relu(shape_channels, shape_channels, 3),
            _conv_norm_relu(shape_channels, shape_channels, 3),
        )
        self.boundary_head = nn.Conv2d(shape_channels, 1, 1)
        self.semantic_fusion = nn.Sequential(
            nn.Conv2d(
                semantic_channels + shape_channels,
                semantic_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(semantic_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, detail: torch.Tensor, semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.detail_projection(detail)
        semantic_high = F.interpolate(
            self.semantic_projection(semantic),
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(torch.cat((detail, semantic_high), dim=1))
        shape = self.shape_refinement(detail * gate)
        boundary_logits = self.boundary_head(shape)
        shape_low = F.interpolate(shape, size=semantic.shape[-2:], mode="area")
        refined = semantic + self.semantic_fusion(torch.cat((semantic, shape_low), dim=1))
        return refined, boundary_logits


class GatedBoundaryResNet50(_DeepLabResNet50Base):
    """DeepLabV3+-ResNet50 plus semantic-gated shape stream and boundary head."""

    def __init__(
        self,
        encoder_weights: str | None = "imagenet",
        classes: int = 5,
        shape_channels: int = 32,
    ) -> None:
        super().__init__(encoder_weights=encoder_weights, classes=classes)
        self.boundary_refinement = SemanticGatedBoundaryRefinement(
            detail_channels=self.encoder.out_channels[1],
            semantic_channels=256,
            shape_channels=shape_channels,
        )

    def forward_with_aux(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, decoded = self._features(x)
        refined, boundary_logits = self.boundary_refinement(features[1], decoded)
        return {
            "logits": self.segmentation_head(refined),
            "boundary_logits": boundary_logits,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)["logits"]


def build_module_model(name: str, encoder_weights: str | None = "imagenet") -> nn.Module:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"deeplab", "deeplabv3plus", "deeplab_resnet50"}:
        return DeepLabResNet50(encoder_weights=encoder_weights)
    if normalized in {"dsconv", "dsconv_resnet50"}:
        return DSConvResNet50(encoder_weights=encoder_weights)
    if normalized in {"gated_boundary", "gated_boundary_resnet50"}:
        return GatedBoundaryResNet50(encoder_weights=encoder_weights)
    raise ValueError(f"unknown module model: {name}")
