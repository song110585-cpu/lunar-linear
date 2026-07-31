"""
双分支 Loss 模块

分支1 (Binary): BCE + Dice — 监督"有线/没线"
分支2 (4-Class): CE (仅前景像素) — 监督"什么类型"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(logits, targets, smooth=1.0):
    """二分类 Dice Loss.

    Args:
        logits: (B, 1, H, W) raw logits
        targets: (B, 1, H, W) float 0/1
        smooth: 平滑项
    Returns:
        scalar loss
    """
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


class DualBranchLoss(nn.Module):
    """双分支联合 Loss.

    分支1: BCE + Dice, 在全部像素上计算, GT 二值化 (0→0, 1-4→1)
    分支2: CE, 仅在 GT>0 的像素上计算, 类别映射 1-4 → 0-3

    Args:
        bce_weight:  分支1 BCE 权重
        dice_weight: 分支1 Dice 权重
        ce_weight:   分支2 CE 权重
        class_weights: 分支2 的 4 类权重 (WR, Rille, Fault, Graben), 默认等权
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, ce_weight=1.0,
                 class_weights=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        if class_weights is not None:
            self.register_buffer('class_weights',
                                 torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, b1_logit, b2_logit, target):
        """
        Args:
            b1_logit: (B, 1, H, W) 分支1 二分类 logits
            b2_logit: (B, 4, H, W) 分支2 四分类 logits
            target:   (B, H, W)   GT 标签, 值 0-4

        Returns:
            total_loss: scalar
            loss_dict:   {'bce': ..., 'dice': ..., 'ce': ...}
        """
        device = b1_logit.device

        # ── 分支1: Binary GT ──
        binary_gt = (target > 0).float().unsqueeze(1)  # (B, 1, H, W)

        loss_bce = self.bce(b1_logit, binary_gt)
        loss_dice = binary_dice_loss(b1_logit, binary_gt)

        # ── 分支2: CE 仅前景像素 ──
        fg_mask = target > 0  # (B, H, W)
        if fg_mask.any():
            # GT 1,2,3,4 → 0,1,2,3
            class_gt = (target[fg_mask] - 1).long()           # (N,)
            # b2_logit: (B, 4, H, W) → (B, H, W, 4) → (N, 4)
            class_logits = b2_logit.permute(0, 2, 3, 1)[fg_mask]  # (N, 4)
            loss_ce = F.cross_entropy(class_logits, class_gt,
                                      weight=self.class_weights)
        else:
            loss_ce = torch.tensor(0.0, device=device)

        # ── 总 Loss ──
        total = (self.bce_weight * loss_bce
                 + self.dice_weight * loss_dice
                 + self.ce_weight * loss_ce)

        return total, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'ce': loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce,
            'total': total.item(),
        }
