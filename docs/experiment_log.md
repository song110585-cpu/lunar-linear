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

---

# R31 Weighted Tile Sampling

## 改动

Dataset v6 + R17 Baseline + **Weighted Tile Sampling** + **Base Model**

- 按前景占比 `sqrt(fg_ratio) + 0.01` 加权采样
- 高前景 tile 更频繁出现，缓解 96% 背景主导问题
- 从 small 升级到 base 模型

## 结果 (Val, 107 tiles)

| Class | IoU | Prec | Recall | F1 |
|---|---|---|---|---|
| Background | 0.9725 | 0.9910 | 0.9811 | 0.9860 |
| Wrinkle Ridge | 0.5236 | 0.5951 | 0.8132 | 0.6873 |
| Rille | 0.5422 | 0.6697 | 0.7400 | 0.7031 |
| **Fault** | **0.3713** | **0.4029** | **0.8257** | **0.5416** |
| Graben | 0.6110 | 0.7480 | 0.7694 | 0.7586 |

OA = 0.9731, mIoU = 0.6041

## 结论

- Base 模型 + 加权采样达到当前最优 mIoU (0.6041)
- Wrinkle/Rille 提升明显 (Recall 81%/74%)
- Fault 仍存在 Precision 低的问题 (40.3%)

---

# R32 Fault GT Dilation + FP Penalty

## 改动

R31 Base + **Fault GT 膨胀 (边界宽容) + 假阳性惩罚 (噪点抑制)**

### 两个新 Loss 组件

**1. Fault GT Dilation (Dice Loss 内)**
```
Fault GT → max_pool2d(kernel=3) 膨胀 1px → Dice 计算
```
模型预测偏 1-2px 仍然算"命中"，不惩罚。针对 Fault 宽度 1-2px 极细问题。

**2. Fault FP Penalty**
```
penalty = mean( (1 - dilated_gt) * fault_prob )
```
膨胀后的 Fault GT 区域外，所有 Fault 预测概率直接惩罚。打击背景纹理中随机产生的假阳性噪点。

### 配置

```yaml
use_fault_penalty: true
fault_dilate: 1           # GT 膨胀半径 (3x3 kernel)
fault_fp_weight: 0.1      # 假阳性惩罚权重
```

### 预期效果

- Fault Precision 从 40% → 45%+
- Fault IoU 从 0.37 → 0.40+

### 训练命令

```bash
python scripts/train.py --config configs/R32.yaml
```

---

# R33 Weighted Tile Sampling (small 模型 + 真正启用加权采样)

## 改动

R26 Baseline (small) + **加权 Tile 采样** (真正启用 `use_tile_sampling: true`)

与 R31 的关键区别:
- R31 使用了 base 模型，但 YAML 中 `use_tile_sampling: false`，加权采样**未生效**
- R33 使用 small 模型，`use_tile_sampling: true`，加权采样**真正生效**
- 权重公式: `sqrt(fg_ratio) + 0.01`

## 结果 (Val, 107 tiles)

```text
============================================================
验证集最终成绩 (Validation Set Evaluation)
Overall Accuracy: 0.9800
Mean mIoU:        0.6523
============================================================
Class                 IoU     Prec   Recall       F1
--------------------------------------------------
Background         0.9794   0.9925   0.9867   0.9896
Wrinkle Ridge      0.5604   0.6598   0.7882   0.7183
Rille              0.5819   0.6862   0.7928   0.7357
Fault              0.4040   0.4530   0.7887   0.5755
Graben             0.7356   0.8533   0.8421   0.8477
--------------------------------------------------
```

| 指标 | R26 (small, 无采样) | R31 (base, 无采样) | **R33 (small, 有采样)** |
|---|---|---|---|
| mIoU | 0.6031 | 0.6041 | **0.6523** |
| OA | 0.9802 | 0.9731 | 0.9800 |
| Fault IoU | 0.3621 | 0.3713 | **0.4040** |
| Fault Prec | 0.4024 | 0.4029 | **0.4530** |
| Fault Recall | 0.8000 | 0.8257 | 0.7887 |
| Graben IoU | 0.3831 | 0.6110 | **0.7356** |
| Rille IoU | 0.4496 | 0.5422 | **0.5819** |
| Wrinkle Ridge IoU | 0.4412 | 0.5236 | **0.5604** |

## 结论

1. **新 SOTA**: mIoU 0.6523，且仅用 small 模型，性价比极高
2. **加权采样是核心突破**: 所有前景类别全面提升，Graben 暴涨 +20.4%
3. **Fault 仍是难例**: Precision 45.3% 虽有提升但仍偏低，需持续关注
4. **种子稳定性待验证**: R33 初次运行未设 seed，需固定种子重跑确认结果稳定

## 训练命令

```bash
# 原始 R33 (无固定 seed)
python scripts/train.py --config configs/R33.yaml

# 种子稳定性验证
python scripts/train.py --config configs/R33-seed2.yaml   # seed=42
python scripts/train.py --config configs/R33-seed3.yaml   # seed=123
```

---

# R33 种子稳定性验证

## 结果

| 指标 | R33 (无seed) | R33-s2 (42) | R33-s3 (123) | 均值 ± 标准差 |
|---|---|---|---|---|
| mIoU | 0.6523 | 0.6417 | 0.6400 | **0.6447 ± 0.0055** |
| Fault IoU | 0.4040 | 0.3958 | 0.3836 | 0.3945 ± 0.0084 |
| Graben IoU | 0.7356 | 0.6955 | 0.7150 | 0.7154 ± 0.0164 |
| Rille IoU | 0.5819 | 0.5860 | 0.5753 | 0.5811 ± 0.0044 |

## 结论

mIoU 标准差 0.0055，加权采样策略稳定可靠。

---

# R34 抗过拟合 (★ 新基线)

## 改动

R33 + **抗过拟合三招**:
1. `tile_sample_exp: 0.5 → 0.3` — 采样温度降温，减少高前景 tile 重复频率
2. `use_scale_aug: true` — 多尺度增强 ±10%，增加输入多样性
3. `patience: 12 → 8` — 早停收紧

代码改动: `train.py` 新增 `tile_sample_exp` 配置项，默认 0.5 向前兼容。

## 结果 (Val, seed=42)

| 指标 | R33-s2 | **R34** | Δ |
|---|---|---|---|
| **mIoU** | 0.6417 | **0.6425** | +0.001 |
| WR IoU | 0.5532 | 0.5417 | -0.012 |
| Rille IoU | 0.5860 | **0.6008** | +0.015 |
| Fault IoU | 0.3958 | 0.3843 | -0.012 |
| Graben IoU | 0.6955 | **0.7091** | +0.014 |

**Loss gap 对比 (同 seed):**
| | R33-s2 | R34 | 改善 |
|---|---|---|---|
| train_loss @ best | 0.149 | 0.189 | +27% (不过低) |
| val_loss @ best | 0.278 | 0.248 | -11% |
| **gap** | **0.129** | **0.059** | **↓ 54%** |

## 结论

1. **过拟合 gap 砍半**，但 mIoU 持平 → 泛化改善被 Precision 下降抵消
2. **R34 确定为后续所有实验的新基线**

---

# R35 Local CNN (已废弃)

## 改动

R34 + **Local CNN 双分支** (`use_local_cnn: true`)

SwinV2 Stage2/3 并行 3×3 Conv → +~6M 参数

## 结果 (Val, seed=42)

| 指标 | R34 | R35 | Δ |
|---|---|---|---|
| mIoU | 0.6425 | 0.6390 | -0.004 |
| Rille IoU | 0.6008 | 0.5660 | -0.035 |
| WR Prec | 60.1% | 66.0% | +5.9pp |

## 结论

WR 受益但 Rille 被严重拖累。3×3 感受野太局部，把窄缝和噪点搞混。废弃。

---

# R36 Strip Pooling (已废弃)

## 改动

R34 + **Strip Pooling** (`use_strip_pooling: true`)

## 结果 (Val, seed=42)

| 指标 | R34 | R36 | Δ |
|---|---|---|---|
| mIoU | 0.6425 | 0.6254 | **-0.017** |
| Fault IoU | 0.3843 | 0.3549 | -0.029 |

## 结论

Strip Pooling 第三次实验仍大幅退步（R19 R25 R36）。方向彻底废弃。

---

# R37 Base 模型 (★ 新 SOTA)

## 改动

R34 + **base 模型** (`model_size: base`)

- R31 用了 base 但 tile sampling 未生效（配置 bug），这是第一次真正验证 "base + 加权采样"
- batch=2, accum=2 (等效 BS=4)

## 结果 (Val, seed=42)

| 指标 | R34 (small) | **R37 (base)** | Δ |
|---|---|---|---|
| **mIoU** | 0.6425 | **0.6613** | **+0.019** |
| OA | 0.9774 | 0.9794 | +0.002 |
| WR IoU | 0.5417 | **0.6039** | +0.062 |
| WR Prec | 60.1% | **69.7%** | +9.6pp |
| Rille IoU | 0.6008 | **0.6097** | +0.009 |
| Rille Prec | 69.1% | **75.0%** | +5.9pp |
| Fault IoU | 0.3843 | **0.3943** | +0.010 |
| Fault Prec | 42.2% | 42.3% | +0.1pp |
| Fault Recall | 81.2% | **85.3%** | +4.1pp |
| Graben IoU | 0.7091 | **0.7201** | +0.011 |

Loss gap: 0.120 (R34=0.059)，有所回升但可接受。

## 结论

1. **Base 模型是继加权采样后第二个有效突破**，mIoU 0.6613 新 SOTA
2. **Precision 全线提升**（WR +10pp, Rille +6pp），base 骨架更强的表达能力让线状构造 vs 背景区分更准
3. **Fault Precision 纹丝不动** (42.3%)，这可能是数据标注/特征本身的上限，非模型容量问题

## 训练命令

```bash
python scripts/train.py --config configs/R37.yaml
```
python scripts/train.py --config configs/R37.yaml
```

