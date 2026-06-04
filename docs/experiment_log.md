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

# R26（计划）

## 改动

Dataset v6

+

R17 Baseline
