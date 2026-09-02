"""PIDNet-S adapted to five-channel, five-class lunar segmentation.

Architecture adapted from the official PIDNet implementation:
https://github.com/XuJiacong/PIDNet
MIT License, Copyright (c) 2022 Jiacong Xu.

The comparison model uses the official inference graph (the P/I/D branches and
boundary-attention fusion) and returns full-resolution semantic logits.  The
common project loss is used instead of PIDNet's dataset-specific auxiliary
training losses so that comparison models share one optimization objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, classes: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(hidden_channels, classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.relu(self.bn1(x), inplace=True))
        return self.conv2(F.relu(self.bn2(x), inplace=True))


class PAPPM(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int, out_channels: int) -> None:
        super().__init__()

        def scale(pool: nn.Module) -> nn.Sequential:
            return nn.Sequential(
                pool,
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            )

        self.scale0 = scale(nn.Identity())
        self.scales = nn.ModuleList(
            [
                scale(nn.AvgPool2d(5, stride=2, padding=2)),
                scale(nn.AvgPool2d(9, stride=4, padding=4)),
                scale(nn.AvgPool2d(17, stride=8, padding=8)),
                scale(nn.AdaptiveAvgPool2d((1, 1))),
            ]
        )
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(4 * branch_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                4 * branch_channels,
                4 * branch_channels,
                3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(5 * branch_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(5 * branch_channels, out_channels, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        local = self.scale0(x)
        pooled = [
            F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False)
            + local
            for branch in self.scales
        ]
        # Grouped PAPPM convolution can produce non-finite FP16 outputs on some
        # CUDA/cuDNN combinations even when its inputs and weights are finite.
        # Keep only this numerically sensitive operation in FP32; the rest of
        # PIDNet still follows the shared AMP training protocol.
        with torch.autocast(device_type=x.device.type, enabled=False):
            processed = self.scale_process(torch.cat(pooled, dim=1).float())
        return self.compression(torch.cat((local, processed), dim=1)) + self.shortcut(x)


class PagFM(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y_query = F.interpolate(
            self.f_y(y), size=size, mode="bilinear", align_corners=False
        )
        similarity = torch.sigmoid(
            torch.sum(self.f_x(x) * y_query, dim=1, keepdim=True)
        )
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return (1.0 - similarity) * x + similarity * y


class LightBag(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(
        self, detail: torch.Tensor, context: torch.Tensor, boundary: torch.Tensor
    ) -> torch.Tensor:
        attention = torch.sigmoid(boundary)
        detail_add = self.conv_p((1.0 - attention) * context + detail)
        context_add = self.conv_i(context + attention * detail)
        return detail_add + context_add


class PIDNetSmall(nn.Module):
    """Official PIDNet-S inference graph with configurable input/classes."""

    def __init__(self, in_channels: int = 5, classes: int = 5) -> None:
        super().__init__()
        planes, ppm_planes, head_planes = 32, 96, 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, 2)
        self.layer2 = self._make_layer(BasicBlock, planes, 2 * planes, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 2 * planes, 4 * planes, 3, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 4 * planes, 8 * planes, 3, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 8 * planes, 8 * planes, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(4 * planes, 2 * planes, 1, bias=False),
            nn.BatchNorm2d(2 * planes, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(8 * planes, 2 * planes, 1, bias=False),
            nn.BatchNorm2d(2 * planes, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(2 * planes, planes)
        self.pag4 = PagFM(2 * planes, planes)
        self.layer3_p = self._make_layer(BasicBlock, 2 * planes, 2 * planes, 2)
        self.layer4_p = self._make_layer(BasicBlock, 2 * planes, 2 * planes, 2)
        self.layer5_p = self._make_layer(Bottleneck, 2 * planes, 2 * planes, 1)

        self.layer3_d = self._make_single_layer(BasicBlock, 2 * planes, planes)
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.layer5_d = self._make_layer(Bottleneck, 2 * planes, 2 * planes, 1)
        self.diff3 = nn.Sequential(
            nn.Conv2d(4 * planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(8 * planes, 2 * planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * planes, momentum=BN_MOMENTUM),
        )
        self.spp = PAPPM(16 * planes, ppm_planes, 4 * planes)
        self.dfm = LightBag(4 * planes)
        self.final_layer = SegmentHead(4 * planes, head_planes, classes)
        self._initialize()

    @staticmethod
    def _make_layer(
        block: type[nn.Module],
        in_channels: int,
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        expansion = block.expansion
        downsample = None
        if stride != 1 or in_channels != channels * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, channels * expansion, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(channels * expansion, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [
            block(in_channels, channels, stride, downsample)
        ]
        current_channels = channels * expansion
        for index in range(1, blocks):
            layers.append(
                block(
                    current_channels,
                    channels,
                    no_relu=index == blocks - 1,
                )
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _make_single_layer(
        block: type[nn.Module], in_channels: int, channels: int
    ) -> nn.Module:
        downsample = None
        if in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion, 1, bias=False),
                nn.BatchNorm2d(channels * block.expansion, momentum=BN_MOMENTUM),
            )
        return block(in_channels, channels, downsample=downsample, no_relu=True)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        branch_size = (input_size[0] // 8, input_size[1] // 8)
        x = self.layer1(self.conv1(x))
        x = self.relu(self.layer2(self.relu(x)))
        detail = self.layer3_p(x)
        boundary = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        detail = self.pag3(detail, self.compression3(x))
        boundary = boundary + F.interpolate(
            self.diff3(x), size=branch_size, mode="bilinear", align_corners=False
        )

        x = self.relu(self.layer4(x))
        detail = self.layer4_p(self.relu(detail))
        boundary = self.layer4_d(self.relu(boundary))
        detail = self.pag4(detail, self.compression4(x))
        boundary = boundary + F.interpolate(
            self.diff4(x), size=branch_size, mode="bilinear", align_corners=False
        )

        detail = self.layer5_p(self.relu(detail))
        boundary = self.layer5_d(self.relu(boundary))
        context = F.interpolate(
            self.spp(self.layer5(x)),
            size=branch_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.final_layer(self.dfm(detail, context, boundary))
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("PIDNet checkpoint does not contain a state_dict")
    result: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        normalized = str(key)
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        replacements = {
            "layer3_.": "layer3_p.",
            "layer4_.": "layer4_p.",
            "layer5_.": "layer5_p.",
            "spp.scale1.": "spp.scales.0.",
            "spp.scale2.": "spp.scales.1.",
            "spp.scale3.": "spp.scales.2.",
            "spp.scale4.": "spp.scales.3.",
        }
        for old, new in replacements.items():
            if normalized.startswith(old):
                normalized = new + normalized[len(old) :]
                break
        if isinstance(value, torch.Tensor):
            result[normalized] = value
    return result


def load_pidnet_imagenet_weights(
    model: PIDNetSmall, checkpoint_path: str | Path
) -> dict[str, int]:
    """Load official 3-channel ImageNet weights and mean-init channels 4-5."""
    source = _unwrap_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    adapted = 0
    for key, value in source.items():
        if key == "conv1.0.weight" and key in target:
            expected = target[key]
            if value.ndim == 4 and value.shape[1] == 3 and expected.shape[1] == 5:
                expanded = expected.clone()
                expanded[:, :3] = value
                expanded[:, 3:] = value.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1)
                loaded[key] = expanded
                adapted += 1
                continue
        if key in target and value.shape == target[key].shape:
            loaded[key] = value
    target.update(loaded)
    if adapted != 1 or len(loaded) < 50:
        raise RuntimeError(
            "PIDNet-S checkpoint is incompatible: "
            f"loaded_tensors={len(loaded)}, adapted_input_tensors={adapted}"
        )
    model.load_state_dict(target, strict=True)
    return {"loaded_tensors": len(loaded), "adapted_input_tensors": adapted}
