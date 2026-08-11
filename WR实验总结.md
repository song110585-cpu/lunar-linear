# WR（皱脊）二分类实验 — 新方向总结

> 最后更新：2026-08-10

---

## 1. 数据已就绪

929 条 WR 面要素，5 区域，2019 tiles (100m) / 3183 tiles (59m Shade)：

| 区域 | Features | 100m Tiles | 59m Shade Tiles |
|------|----------|:---:|:---:|
| Mare Imbrium | 212 | 420 | 1224 |
| Mare Serenitatis | 179 | 270 | 780 |
| Mare Tranquillitatis | 140 | 132 | 380 |
| Marius Hills | 137 | 240 | 294 |
| Oceanus Procellarum | 261 | 957 | 509 |
| **合计** | **929** | **2019** | **3183** |

两种划分策略：

| | Mixed | Holdout |
|------|-------|---------|
| **Train** | 5区混合 | 东部4区 |
| **Val/Test** | 5区混合 | 风暴洋分层 50:50 |
| **目的** | 跟文献 0.71 对标 | 测跨区域泛化 |

---

## 2. 模型与超参

- Swin-UNet (SwinV2-LCSRB + DeformablePSP + FPNPAN), model_size=small (~72M)
- BCE + Dice Loss, pos_weight=10 (class_weights=[0.1, 1.0])
- lr=5e-5, batch=2, accum=2, freeze_stages=1
- 关闭所有结构模块（纯净基线）
- Tile-level weighted sampling (exp=0.5)
- 早停 patience=15

---

## 3. R1: 5ch 100m (2026-08-10)

输入: [WAC, DEM, Slope, TPI, Profile Curvature] @ 100m

| 指标 | Mixed (5区混合) | Holdout (东部→风暴洋) |
|------|:---:|:---:|
| **Train tiles** | 1615 | 1062 |
| **Val WR IoU (最佳)** | 0.4876 (epoch 57) | 0.4931 (epoch 35) |
| **Test WR IoU** | 0.4741 | 0.4920 |
| Val mIoU | 0.7399 | 0.7434 |
| Test mIoU | 0.7325 | 0.7429 |
| 总 Epoch | 72 | 50 (早停) |

### 空间泄漏分析

```
泄漏缺口 = Mixed Val - Holdout Val = 0.4876 - 0.4931 = -0.0055 ≈ 0
```

未观察到明显的空间数据泄漏。Holdout 的跨区域泛化良好。

### 关键发现：分辨率不匹配

- 标注时使用 **59m** Shade + DEM
- 训练时使用 **100m** 5ch 数据
- 标签被降采样到 100m 栅格化，边界信息损失约 **40%**
- **这可能是 R1 精度偏低的核心原因**（不限于 5 分类时期，所有之前实验都受此影响）

---

## 4. 文献对标：Lu et al. (2025)

发表于 RAA，WR 二分类 **IoU = 0.716**。

| | Lu et al. | R1 |
|------|:---:|:---:|
| 模型 | DBR-Net (双 ResNet-34 + ACFF) | Swin-UNet (Transformer) |
| 输入 | **2ch**: DEM + Aspect(方差滤波) | 5ch: WAC+DEM+Slope+TPI+Curv |
| 分辨率 | **59m** | **100m** |
| 划分 | 8:2 随机 | 8:1:1 随机 |
| 关键创新 | Aspect 通道 + 方差滤波 + 双分支 | — |

### 可借鉴的

1. **Aspect + 方差滤波**：WR 边缘处坡向一致性高，方差滤波增强边缘信号
2. **59m 分辨率**：训练分辨率匹配标注分辨率
3. Slope/TPI/Curvature 都是从 DEM 导出的——Swin Transformer 有能力自己学，不是必需的独立信息

---

## 5. R2 实验计划：验证分辨率 + 通道优化

### 5.1 根本原因假设

R1 低精度(0.49)的主要瓶颈是 **100m 分辨率导致标签边界损失 40%**，而非模型或通道问题。

### 5.2 两阶段验证

#### 阶段 A: Shade 59m 单通道（进行中）

| | R1 | R2-Shade |
|------|:---:|:---:|
| 通道 | 5 | **1** (Shade) |
| 分辨率 | 100m | **59m** |
| 数据集 | wr_dataset_mixed/holdout | wr_dataset_shade_mixed/holdout |
| Config | R_wr_serenitatis.yaml | R_wr_shade_1ch.yaml |
| Kaggle | changyasong/yuanssy | changyasong/yuanssy |

**目的**: 隔离分辨率变量。如果 59m 单通道就超过 100m 5ch，证明分辨率是主要瓶颈。

#### 阶段 B: 3ch 59m（脚本待写）

| # | 通道 | 理由 |
|:---:|------|------|
| 1 | Hillshade | 标注时的光学参考，替换 WAC |
| 2 | DEM | 原生 59m DEMmerge (LOLA+Kaguya) |
| 3 | Aspect(方差滤波) | Lu et al. 验证的边缘信息，独立于 DEM |

不再使用 Slope/TPI/Curvature——Swin Transformer 能从 DEM 自学导数特征，不是独立信息。对标 Lu et al. 的 DEM+Aspect 双通道。

### 5.3 新增数据资产

| 文件 | 说明 |
|------|------|
| `Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif` | 全局 DEM @ 59m, 22.7GB |
| `SLDEM2015_512_60S_60N_000_360.JP2` | SLDEM2015 @ 59m, 5.5GB (备用) |

### 5.4 目标

| 阶段 | WR IoU | 说明 |
|------|:---:|------|
| R1 (5ch 100m) | 0.49 | 基线 |
| R2-Shade (1ch 59m) | 0.50-0.55 | 测分辨率收益 |
| R2-3ch (3ch 59m) | 0.55-0.65 | 最优通道组合 |
| 最终目标 | **0.65-0.70** | 逼近 Lu et al. 0.716 |

---

## 6. 脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/generate_5ch.py` | 5ch 100m TIFF (R1) |
| `scripts/prepare_wr_dataset.py` | 100m tile 数据集 |
| `scripts/generate_shade_1ch.py` | Shade 归一化 1ch 59m |
| `scripts/prepare_shade_dataset.py` | Shade 59m tile 数据集 |
| `scripts/generate_6ch_59m.py` | 6ch 59m TIFF (待改 3ch) |
| `configs/R_wr_serenitatis.yaml` | R1 训练配置 (in_channels=5) |
| `configs/R_wr_shade_1ch.yaml` | Shade 训练配置 (in_channels=1) |
| `notebooks/wr_mixed.ipynb` / `wr_holdout.ipynb` | R1 Kaggle notebook |
| `notebooks/wr_shade_mixed.ipynb` / `wr_shade_holdout.ipynb` | R2-Shade Kaggle notebook |
