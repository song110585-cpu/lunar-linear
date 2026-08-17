"""
LTL-Net: 细线保持解码器 (Thin-line Preserving Decoder, TPD)

主干: ResNet50 (smp, ImageNet 预训练, output_stride=16)
自研 decoder: 在 DeepLabV3+ 基础上, 额外接入 encoder 的 1/2 分辨率浅层特征 (features[1]),
             用于恢复 1-2px 的极细线性构造 (断层/月溪/皱脊)。

核心动机 (见 实验现状总结 R58 的「Swin patch 化丢细线」诊断):
  输入 512×512 时, 1-2px 的断层在 encoder 1/4 特征(128×128)里只剩 0.25-0.5px(亚像素),
  下采样两次后信号已被稀释。DeepLabV3+ 只用到 1/4(features[2]), 细线信息已丢失大半。
  唯一补救是接入 1/2 分辨率特征(features[1], 256×256), 此时细线仍有 0.5-1px 可辨。

与 DeepLabV3+ 的唯一结构差异:
  DeepLabV3+: ASPP → 上采样到 1/4 → 融合 features[2] → 4×上采样输出
  TPD:        ASPP → 上采样到 1/4 → 融合 features[2] → 上采样到 1/2 → 融合 features[1] → 2×上采样输出
  (TPD 多了一级 1/2 细节融合, 最终上采样从 4× 降到 2×, 细线保留更好)
"""
import torch
from torch import nn
import torch.nn.functional as F

from segmentation_models_pytorch.encoders import get_encoder
# 复用 smp 的 ASPP / SeparableConv2d (与 DeepLabV3+ 完全同构, 保证消融唯一变量是「多接 f1」)
from segmentation_models_pytorch.decoders.deeplabv3.decoder import ASPP, SeparableConv2d


class TPDDecoder(nn.Module):
    """细线保持解码器 (Thin-line Preserving Decoder)

    encoder 输出 6 个特征 (depth=5, output_stride=16, resnet50):
        features[0] 512×512×5   输入
        features[1] 256×256×64  1/2   (conv1 后, 细线细节)  ← TPD 新增接入
        features[2] 128×128×256 1/4   (layer1 后, 低层语义)  ← DeepLabV3+ 原有
        features[3] 64×64×512   1/8
        features[4] 32×32×1024  1/16
        features[5] 32×32×2048  1/16  (layer4, dilation)
    """

    def __init__(
        self,
        encoder_channels,           # list, 如 [5, 64, 256, 512, 1024, 2048]
        encoder_depth=5,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        aspp_separable=True,
        aspp_dropout=0.5,
        highres_detail_channels=16,  # f1(1/2, 64ch) 降维后的细节通道数
    ):
        super().__init__()

        # ---- ASPP (与 DeepLabV3+ 完全一致) ----
        self.aspp = nn.Sequential(
            ASPP(
                encoder_channels[-1], out_channels, atrous_rates,
                separable=aspp_separable, dropout=aspp_dropout,
            ),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # ASPP 输出上采样: output_stride=16 且 depth=5 时 4× (1/16 → 1/4)
        scale_factor = 4 if output_stride == 16 and encoder_depth > 3 else 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        # ---- 低层语义特征 features[2] (1/4, 256ch) → 48ch (同 DeepLabV3+) ----
        self.block1 = nn.Sequential(
            nn.Conv2d(encoder_channels[2], 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(48 + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # ---- 新增: 高分辨率细节特征 features[1] (1/2, 64ch) → highres_detail_channels ----
        self.detail_block = nn.Sequential(
            nn.Conv2d(encoder_channels[1], highres_detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_detail_channels),
            nn.ReLU(),
        )
        # ---- 新增: 1/2 分辨率融合层 ----
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fuse_block = nn.Sequential(
            SeparableConv2d(highres_detail_channels + out_channels, out_channels,
                            kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, features):
        # 1) ASPP 处理最深特征, 上采样到 1/4
        aspp = self.aspp(features[-1])
        aspp = self.up(aspp)

        # 2) 低层语义融合 (同 DeepLabV3+), 得到 1/4 的 256ch
        low = self.block1(features[2])
        fused = self.block2(torch.cat([aspp, low], dim=1))

        # 3) 上采样到 1/2, 接入高分辨率细节特征 features[1]
        fused = self.up2(fused)
        detail = self.detail_block(features[1])
        out = self.fuse_block(torch.cat([fused, detail], dim=1))
        return out


class LTLNet(nn.Module):
    """LTL-Net 主体: ResNet50 主干 + 细线保持解码器 (TPD) + 分类头"""

    def __init__(
        self,
        encoder_name='resnet50',
        encoder_depth=5,
        encoder_weights='imagenet',
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        in_channels=5,
        classes=5,
        highres_detail_channels=16,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.decoder = TPDDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            highres_detail_channels=highres_detail_channels,
        )
        # 分类头: decoder 输出 1/2 分辨率, 1×1 卷积后 2× 上采样到原图
        self.head = nn.Conv2d(decoder_channels, classes, kernel_size=1)
        self.up_final = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)
        x = self.up_final(x)
        return x


if __name__ == '__main__':
    # 本地尺寸自检 (CPU)
    model = LTLNet(encoder_name='resnet50', in_channels=5, classes=5)
    model.eval()
    x = torch.randn(1, 5, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f'输入: {tuple(x.shape)}  输出: {tuple(y.shape)}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
