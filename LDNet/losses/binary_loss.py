"""
二分类 Loss: BCE + Dice

BCE: 逐像素监督, 确保背景不炸
Dice: 全局监督, 缓解 98% 背景压倒 → 推高 Recall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(logits, targets, smooth=1.0):
    """二分类 Dice Loss.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) float 0/1
    Returns:
        scalar loss
    """
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


class BinaryLoss(nn.Module):
    """BCE + Dice 联合 Loss.

    Args:
        bce_weight: BCE 权重 (默认 1.0)
        dice_weight: Dice 权重 (默认 1.0)
        pos_weight: BCE 正样本权重, 缓解正负不平衡 (默认 2.0)
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, binary_target):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            binary_target: (B, 1, H, W) float 0/1
        Returns:
            total_loss: scalar
            loss_dict: {'bce': ..., 'dice': ..., 'total': ...}
        """
        loss_bce = self.bce(logits, binary_target)
        loss_dice = binary_dice_loss(logits, binary_target)

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        return total, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'total': total.item(),
        }


def label_to_binary(mask):
    """5 类 mask (0-4) → 二值 mask (0/1).

    Args:
        mask: (B, H, W) or (H, W) int64, 值 0-4
    Returns:
        binary: same shape, 0=背景, 1=线性构造
    """
    return (mask > 0).long()
