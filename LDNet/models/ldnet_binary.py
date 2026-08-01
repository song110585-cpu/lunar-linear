"""
LDNet-Binary: 单分支二分类 — Step 1 "线 vs 背景"

复用 STDL-Net R50 验证过的 decoder 架构:
  SwinV2-LCSRB → PSPModule → FPNPAN → conv_fusion → head(1ch)

Loss: BCE + Dice, 目标: 高 Recall, 所有线性构造不漏.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from backbone import swin_v2_LCSRB, PSPModule, FPNPAN


class LDNetBinary(nn.Module):
    """单分支二分类模型 — 只判断"有线/没线".

    Args:
        size: backbone 规模 ('tiny' | 'small' | 'base')
        img_size: 输入 tile 尺寸 (默认 512)
        in_channels: 输入通道数 (默认 5)
        pretrained: 是否加载预训练权重
    """

    def __init__(self, size='small', img_size=512, in_channels=5, pretrained=True):
        super().__init__()

        self.backbone = swin_v2_LCSRB(
            size=size, img_size=img_size, in_chans=in_channels, pretrained=pretrained)

        variant = size.lower()
        if 'base' in variant:
            feature_channels = [256, 512, 1024, 1024]
        else:
            feature_channels = [192, 384, 768, 768]
        fpn_out = feature_channels[0]

        self.ppm = PSPModule(feature_channels[-1])
        self.fpnpan = FPNPAN(feature_channels, fpn_out=fpn_out)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(fpn_out, 1, kernel_size=3, padding=1)

    def freeze_backbone_stages(self, num_stages=2):
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
        input_size = (x.size()[2], x.size()[3])

        feats = self.backbone.extra_features(x)
        feats = list(feats)
        feats[-1] = self.ppm(feats[-1])
        feats = self.fpnpan(feats)
        x = self.conv_fusion(torch.cat(feats, dim=1))
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x  # (B, 1, H, W) logits

    def predict(self, x, threshold=0.5):
        """返回二值 mask: 0=背景, 1=线性构造."""
        return (torch.sigmoid(self.forward(x)) > threshold).squeeze(1).long()

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(
            self.ppm.parameters(), self.fpnpan.parameters(),
            self.conv_fusion.parameters(), self.head.parameters(),
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
