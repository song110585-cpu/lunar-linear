# LTL-Net — Lunar Tectonic Linear feature Net

月球线性构造（5 类）语义分割，**自己的模型**。

## 定位

- **Backbone**：ResNet50（`segmentation_models_pytorch` 现成，ImageNet 预训练）
- **Decoder**：自研细线增强 decoder（待设计）
- **任务**：Background / Wrinkle Ridge(皱脊) / Rille(月溪) / Fault(断层) / Graben(地堑)
- **输入**：5 通道 [WAC, DEM, Slope, TPI, Profile Curvature]

## 目录

```
LTL-Net/
├── models/      自研 decoder（待实现）
├── scripts/     训练/评估脚本（train_baseline.py 跑对比基线）
├── utils/       metrics.py（评估工具，复制自 STDL-Net）
├── datasets/    MyDataset.py（数据集类，复制自 STDL-Net）
└── configs/     训练配置
```

## 当前阶段

1. **跑基线摸清底线**：`scripts/train_baseline.py --model Unet/PSPNet/Linknet/DeepLabV3Plus`
2. **锁定 backbone**（ResNet50）
3. **设计自研 decoder + 细线模块**，超过 DeepLabV3+ 基线（前景 mIoU 0.674）

## 与 STDL-Net 的关系

- `STDL-Net/` 是参考/对比代码（Swin 等），保留不动
- `LTL-Net/` 是**我的模型**，只放 ResNet50 + 自研 decoder
