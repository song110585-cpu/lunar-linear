"""Differentiable 2D Dynamic Snake Convolution.

Adapted for this project from the official DSCNet implementation:
https://github.com/YaoleiQi/DSCNet (MIT License, Copyright 2025 Yaolei Qi).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 16) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DynamicSnakeConv2d(nn.Module):
    """One oriented dynamic-snake convolution with unchanged output size."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if morph not in (0, 1):
            raise ValueError("morph must be 0 (horizontal) or 1 (vertical)")
        self.kernel_size = kernel_size
        self.extend_scope = float(extend_scope)
        self.morph = morph
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1, bias=False),
            nn.GroupNorm(kernel_size, 2 * kernel_size),
            nn.Tanh(),
        )
        if morph == 0:
            kernel, stride = (kernel_size, 1), (kernel_size, 1)
        else:
            kernel, stride = (1, kernel_size), (1, kernel_size)
        self.collapse = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, bias=False
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.ReLU(inplace=True)

    def _continuous_offsets(self, raw: torch.Tensor) -> torch.Tensor:
        centre = self.kernel_size // 2
        zeros = torch.zeros_like(raw[:, centre : centre + 1])
        left = torch.flip(
            torch.cumsum(torch.flip(raw[:, :centre], dims=(1,)), dim=1), dims=(1,)
        )
        right = torch.cumsum(raw[:, centre + 1 :], dim=1)
        return torch.cat((left, zeros, right), dim=1) * self.extend_scope

    def _coordinate_grid(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        dtype, device = x.dtype, x.device
        centre = self.kernel_size // 2
        y_coord = torch.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
        x_coord = torch.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)
        spread = torch.arange(
            -centre, centre + 1, dtype=dtype, device=device
        ).view(1, self.kernel_size, 1, 1)

        if self.morph == 0:
            grid_y = (y_coord + offsets).expand(batch, -1, height, width)
            grid_x = (x_coord + spread).expand(batch, -1, height, width)
            grid_y = grid_y.permute(0, 2, 1, 3).reshape(batch, height * self.kernel_size, width)
            grid_x = grid_x.permute(0, 2, 1, 3).reshape(batch, height * self.kernel_size, width)
        else:
            grid_y = (y_coord + spread).expand(batch, -1, height, width)
            grid_x = (x_coord + offsets).expand(batch, -1, height, width)
            grid_y = grid_y.permute(0, 2, 3, 1).reshape(batch, height, width * self.kernel_size)
            grid_x = grid_x.permute(0, 2, 3, 1).reshape(batch, height, width * self.kernel_size)

        grid_x = 2.0 * grid_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(height - 1, 1) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_y, raw_x = self.offset(x).chunk(2, dim=1)
        raw = raw_y if self.morph == 0 else raw_x
        grid = self._coordinate_grid(x, self._continuous_offsets(raw))
        sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return self.activation(self.norm(self.collapse(sampled)))


class DynamicLineRefinement(nn.Module):
    """Residual multi-view line refinement at the decoder's 1/4 seam."""

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 64,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.standard = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.horizontal = DynamicSnakeConv2d(
            hidden_channels, hidden_channels, kernel_size, extend_scope, morph=0
        )
        self.vertical = DynamicSnakeConv2d(
            hidden_channels, hidden_channels, kernel_size, extend_scope, morph=1
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        refined = self.fuse(
            torch.cat(
                (self.standard(reduced), self.horizontal(reduced), self.vertical(reduced)),
                dim=1,
            )
        )
        return self.activation(x + refined)
