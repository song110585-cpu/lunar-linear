"""
LDNet: Lunar Linear Detection Network (双分支架构)

分支1 (Binary):  "线在哪？" → 1ch sigmoid, 只管前景/背景检测
分支2 (4-Class): "什么类型？" → 4ch logits, 只在线性区域做精细分类

推理时合并: final_pred = argmax(b2) × (b1 > 0.5), 背景自动归零

Gate: 分支1 的 sigmoid 作为分支2 conv_fusion 的额外通道,
     让分支2 知道"哪里是分支1 认为有线的区域"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from backbone import swin_v2_LCSRB, PSPModule, FPNPAN


class LDNet(nn.Module):
    """Lunar Linear Detection Network — 双分支解耦架构.

    Args:
        size: backbone 规模 ('tiny' | 'small' | 'base')
        img_size: 输入 tile 尺寸 (默认 512)
        in_channels: 输入通道数 (默认 5: WAC+DEM+Slope+TPI+曲率)
        pretrained: 是否加载 ImageNet-22k 预训练权重
    """

    def __init__(self, size='small', img_size=512, in_channels=5, pretrained=True):
        super().__init__()

        # ── 共享 Backbone ──
        self.backbone = swin_v2_LCSRB(
            size=size, img_size=img_size, in_chans=in_channels, pretrained=pretrained)

        # 特征通道数
        variant = size.lower()
        if 'base' in variant:
            feature_channels = [256, 512, 1024, 1024]
        else:  # tiny / small
            feature_channels = [192, 384, 768, 768]
        fpn_out = feature_channels[0]

        # ═══════════════════════════════════════════
        # 分支1: Binary Detection — "线在哪？"
        # ═══════════════════════════════════════════
        self.ppm1 = PSPModule(feature_channels[-1])
        self.fpnpan1 = FPNPAN(feature_channels, fpn_out=fpn_out)
        self.conv_fusion1 = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Conv2d(fpn_out, 1, kernel_size=3, padding=1)

        # ═══════════════════════════════════════════
        # 分支2: 4-Class Classification — "什么类型？"
        # ═══════════════════════════════════════════
        # conv_fusion2 多 1 个通道: 分支1 的 sigmoid gate
        self.ppm2 = PSPModule(feature_channels[-1])
        self.fpnpan2 = FPNPAN(feature_channels, fpn_out=fpn_out)
        self.conv_fusion2 = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out + 1, fpn_out,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )
        self.head2 = nn.Conv2d(fpn_out, 4, kernel_size=3, padding=1)

    def freeze_backbone_stages(self, num_stages=2):
        """冻结 backbone 前 num_stages 个阶段 (含 patch_embed)."""
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        for i in range(min(num_stages, len(self.backbone.layers))):
            for p in self.backbone.layers[i].parameters():
                p.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f'[freeze] {frozen}/{total} params frozen '
              f'(前 {num_stages} 阶段 + patch_embed)')

    def forward(self, x):
        """Forward pass, 返回两个分支的原始输出.

        Returns:
            b1_logit: (B, 1, H, W) 二分类 logits
            b2_logit: (B, 4, H, W) 四分类 logits (0=WR, 1=Rille, 2=Fault, 3=Graben)
        """
        input_size = (x.size()[2], x.size()[3])

        # ── 共享 Backbone 特征 ──
        feats = self.backbone.extra_features(x)
        # feats: [f0@1/8, f1@1/16, f2@1/32, f3@1/32]

        # ═══ 分支1: Binary ═══
        f1 = feats.copy()
        f1[-1] = self.ppm1(f1[-1])
        f1 = self.fpnpan1(f1)
        f1_fused = self.conv_fusion1(torch.cat(f1, dim=1))
        b1_logit = self.head1(f1_fused)
        b1_prob = torch.sigmoid(b1_logit)

        # ═══ 分支2: 4-Class (Gate 来自分支1) ═══
        f2 = feats.copy()
        f2[-1] = self.ppm2(f2[-1])
        f2 = self.fpnpan2(f2)
        # 把 b1_prob 缩放到 f2[0] 的尺寸, 拼接到 conv_fusion
        gate = F.interpolate(b1_prob, size=f2[0].shape[-2:],
                             mode='bilinear', align_corners=True)
        f2_cat = torch.cat(f2 + [gate], dim=1)
        f2_fused = self.conv_fusion2(f2_cat)
        b2_logit = self.head2(f2_fused)

        # ── 上采样到原始尺寸 ──
        b1_logit = F.interpolate(b1_logit, size=input_size,
                                 mode='bilinear', align_corners=True)
        b2_logit = F.interpolate(b2_logit, size=input_size,
                                 mode='bilinear', align_corners=True)

        return b1_logit, b2_logit

    def predict(self, x):
        """推理模式: 直接返回合并后的 5 类预测.

        Returns:
            pred: (B, H, W) int64, 值 0-4
        """
        b1_logit, b2_logit = self.forward(x)
        return merge_branches(b1_logit, b2_logit)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(
            self.ppm1.parameters(), self.fpnpan1.parameters(),
            self.conv_fusion1.parameters(), self.head1.parameters(),
            self.ppm2.parameters(), self.fpnpan2.parameters(),
            self.conv_fusion2.parameters(), self.head2.parameters(),
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def merge_branches(b1_logit, b2_logit, threshold=0.5):
    """合并双分支输出为 5 类预测.

    Args:
        b1_logit: (B, 1, H, W) 二分类 logits
        b2_logit: (B, 4, H, W) 四分类 logits
        threshold: 二分类阈值

    Returns:
        pred: (B, H, W) int64, 值 0-4
    """
    binary_mask = (torch.sigmoid(b1_logit) > threshold).squeeze(1)  # (B, H, W)
    class_pred = b2_logit.argmax(dim=1) + 1                         # (B, H, W), 1-4
    return (class_pred * binary_mask).long()
