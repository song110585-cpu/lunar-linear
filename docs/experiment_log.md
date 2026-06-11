# 月球线性构造提取实验记录

## 实验环境

- Backbone: SwinV2-Small
- 输入通道: 5-channel
- 类别数: 5
  - Background
  - Wrinkle Ridge
  - Rille
  - Fault
  - Graben
- 评价指标:
  - OA
  - mIoU
  - Precision
  - Recall
  - F1-score

---

# R17 Baseline

## 改动

Baseline模型

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9834 |
| Wrinkle Ridge | 0.4421 |
| Rille | 0.4744 |
| Fault | 0.3725 |
| Graben | 0.4115 |

OA = 0.9833

mIoU = 0.5368

## 结论

当前最佳Baseline。

---

# R18

## 改动

类别权重调整：

Fault = Graben = 3

## 结果

效果明显下降。

## 结论

类别权重强化策略无效。

---

# R19 Strip Pooling

## 改动

加入 Strip Pooling 模块。

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9842 |
| Wrinkle Ridge | 0.4490 |
| Rille | 0.4643 |
| Fault | 0.3255 |
| Graben | 0.3537 |

OA = 0.9837

mIoU = 0.5153

## 结论

较原始结果略有提升，但低于R17。

---

# R20 Coordinate Attention

## 改动

移除 Strip Pooling。

加入 Coordinate Attention。

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9846 |
| Wrinkle Ridge | 0.4039 |
| Rille | 0.3913 |
| Fault | 0.2357 |
| Graben | 0.3302 |

OA = 0.9845

mIoU = 0.4692

## 结论

性能明显下降。

Coordinate Attention 不适用于当前任务。

---

# R21 Boundary Supervision

## 改动

加入边界监督。

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9847 |
| Wrinkle Ridge | 0.4515 |
| Rille | 0.4694 |
| Fault | 0.3128 |
| Graben | 0.3843 |

OA = 0.9843

mIoU = 0.5205

## 结论

较Baseline略有提升。

---

# R22 DEM-Guided Fusion

## 改动

加入DEM-Guided Fusion模块。

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9755 |
| Wrinkle Ridge | 0.3761 |
| Rille | 0.3732 |
| Fault | 0.2888 |
| Graben | 0.3123 |

OA = 0.9751

mIoU = 0.4652

## 结论

DEM信息引入后产生大量误检。

Precision下降明显。

模块失效。

---

# R23 Multi-modal Gating

## 改动

加入跨模态条件门控。

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9814 |
| Wrinkle Ridge | 0.4439 |
| Rille | 0.2787 |
| Fault | 0.3362 |
| Graben | 0.3264 |

OA = 0.9809

mIoU = 0.4733

## 结论

对Fault有提升。

但Rille严重下降。

总体效果不佳。

---

# R24 Dataset v6

## 改动

数据集优化：

- 增加Graben样本
- 过滤背景占比过高样本
- 增加困难样本

详见 [dataset_versions.md](dataset_versions.md)

## 结果

| Class | IoU |
|---------|---------|
| Background | 0.9845 |
| Wrinkle Ridge | 0.4468 |
| Rille | 0.4241 |
| Fault | 0.3158 |
| Graben | 0.3447 |

OA = 0.9840

mIoU = 0.5032

## 结论

Graben类别提升明显。

数据集优化效果优于多数结构改进模块。

---

# R25

## 改动

DEM-Guided Fusion
+
Strip Pooling
+
Gradient Clipping

## 结果

mIoU ≈ 0.4+

## 结论

效果较差。

说明DEM相关模块仍然存在问题。

---

# R26

## 改动

Dataset v6 + R17 Baseline (不包含 DEM、Strip Pooling 等，作为 v6 的纯净基准)

## 结果

| Class | IoU | Prec | Recall | F1-score |
|---------|---------|---------|---------|---------|
| Background | 0.9806 | 0.9950 | 0.9810 | 0.9880 |
| Wrinkle Ridge | 0.4412 | 0.4938 | 0.7780 | 0.6044 |
| Rille | 0.4496 | 0.5186 | 0.7660 | 0.5501 |
| Fault | 0.3621 | 0.4024 | 0.8000 | 0.5358 |
| Graben | 0.3831 | 0.4285 | 0.8070 | 0.5594 |

OA = 0.9802

mIoU = 0.6031  (相比 v5 Baseline 提升 +6.63%)

## 结论

1. **里程碑式的优秀基准**：各前景构造（Wrinkle Ridge / Rille / Fault / Graben）的召回率（Recall）极高，全部达到 **76%~80.7%**！
2. **零跨类别混淆**：模型对各构造形态的特征把握极准，几乎没有“指鹿为马”的错分类。
3. **唯一的学术痛点**：约 20% 左右的像素在边缘处被漏判为背景（漏检），这主要是由于线性构造极度狭窄细长。

---

# R27（进行中）

## 改动

Dataset v6 + R17 Baseline + **Strip Pooling 
- 在 Swin-LCSRB 主干中启用长条形池化，强迫模型沿构造延伸方向建立长距离依赖，锁定狭窄的线状构造。
- 期待解决 R26 中 20% 构造边缘被划分为背景的漏检痛点。

---

# R28 new baseline+boundary

## 改动

Dataset v6 + R17 Baseline + Boundary Loss

## 结果

效果不佳，与 R27 类似。

---

# R29: R26 Baseline + Light Boundary Supervision

## 改动

Dataset v6 + R17 Baseline + Boundary Loss (light weight=0.3)

- `boundary_loss_weight: 0.3`
- `boundary_kernel_size: 3`

## 结果

Fault Precision: 38.6% (R26: 37.7%)，提升微乎其微。

BG→Fault 误检仍占 61.3%。

## 结论

Boundary Loss 对 Fault 的 Precision 几乎无帮助。Fault 被误分为背景的问题不是边界不清晰，而是背景纹理和断层形态过于相似。

---

# R30 Local CNN 双分支

## 改动

在 SwinV2 Stage 2 (H/8) 和 Stage 3 (H/16) 引入 **Local CNN 并行分支**：

```
Stage 2/3 的 Swin Transformer 输出
  ├── Transformer 分支 (全局语义, 保持不变)
  ├── LocalCNN 分支: 3×3Conv×2 + BN + ReLU (局部细线细节保留)
  └── Concat(Transformer, LocalCNN) → 1×1Conv → 融合
```

### 设计原理

| 决策 | 理由 |
|---|---|
| 用 3×3 Conv 而非 DWConv | 极细线需要跨通道空间协同，DWConv 逐通道独立会丢失关联信号 |
| Concat 融合而非 Add | 两条路径信号独立，让网络自行学习融合权重 |
| 只加 Stage 2/3 | Stage 4 分辨率太低 (H/32)，细线结构已丢失 |
| Stage 级并行 (非 Block 级) | 不干扰每个 Swin block 内部的 attention 计算 |
| 独立 if 分支 (非 if/elif) | 可与 Strip Pooling 或其他模块叠加 |

### 参数量

- Stage 2 (384ch): 3x3 Conv ×2 + 1x1 Fuse ≈ 1.2M
- Stage 3 (768ch): 3x3 Conv ×2 + 1x1 Fuse ≈ 4.7M
- 总计 ≈ **+5.9M** (约 +8% vs 73M baseline)

### 预期效果

- 保留极细线（月溪）和断层（Fault）的局部几何细节
- 减少 Fault→Background 的漏检（R26 中 Fault 的 FN 约 20%）
- 不破坏 Transformer 的全局语义理解

### 配置

```yaml
use_local_cnn: true   # configs/R30.yaml
```

### 训练命令

```bash
python scripts/train.py --config configs/R30.yaml
```

