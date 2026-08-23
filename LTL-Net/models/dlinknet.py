"""D-LinkNet semantic-segmentation model with an SMP encoder.

This is a real D-LinkNet-style architecture: a pretrained encoder, a cascaded
dilated-convolution center block, additive skip connections, and a lightweight
transpose-convolution decoder.  It must not be confused with SMP ``Linknet``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch.encoders import get_encoder


class DBlock(nn.Module):
    """Cascaded dilated center block used by D-LinkNet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dilate1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.dilate4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.dilate8 = nn.Conv2d(channels, channels, 3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d4 = self.relu(self.dilate4(d2))
        d8 = self.relu(self.dilate8(d4))
        return x + d1 + d2 + d4 + d8


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden_channels = max(in_channels // 4, 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DLinkNet(nn.Module):
    """D-LinkNet with configurable SMP encoder; formal experiments use ResNet50."""

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
        classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        channels = list(self.encoder.out_channels)
        if len(channels) != 6:
            raise ValueError(
                f"DLinkNet requires five encoder stages, got out_channels={channels}"
            )
        c1, c2, c3, c4, c5 = channels[1:]
        self.center = DBlock(c5)
        self.decoder4 = DecoderBlock(c5, c4)
        self.decoder3 = DecoderBlock(c4, c3)
        self.decoder2 = DecoderBlock(c3, c2)
        self.decoder1 = DecoderBlock(c2, c1)
        final_channels = max(c1 // 2, 16)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c1, final_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, final_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, classes, 3, padding=1),
        )

    @staticmethod
    def _add_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x + skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        e1, e2, e3, e4, e5 = features[1:]
        center = self.center(e5)
        d4 = self._add_skip(self.decoder4(center), e4)
        d3 = self._add_skip(self.decoder3(d4), e3)
        d2 = self._add_skip(self.decoder2(d3), e2)
        d1 = self._add_skip(self.decoder1(d2), e1)
        logits = self.final(d1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

