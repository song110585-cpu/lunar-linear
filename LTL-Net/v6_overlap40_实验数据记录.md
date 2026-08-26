# v6 overlap40 实验数据记录

> 建立日期：2026-08-24
> 适用范围：`dataset_v6_random811_overlap40` 上已经完成的正式模型与模块实验
> 主指标：前景四类平均交并比 `mIoU_fg`（WR、Rille、Fault、Graben）

## 1. 文档边界与更新规则

本文件是**数据台账**，只记录：数据协议、实验配置、数值结果、证据路径、可比性限制和基于结果作出的结论。

从本文件建立之日起：

- Kaggle/AutoDL 上传、运行命令、排错过程、代码修改和操作决策写入 `实验操作记录.md`；
- 数值结果只写入本文件，不再混入操作日志；
- 只有检查过下载到本地的 `metrics/history/config/checkpoint` 后，结果才标记为“原始产物已核验”；
- 仅由旧总结或人工报告得到、但本地缺少原始产物的结果，必须标记为“旧文档记录”；
- Val 与 Test 严格分表；没有 Test 结果的模块不得按 Test 收益表述；
- 同一模型的重跑必须保留为独立行，不覆盖旧结果，并写明 checkpoint 选择指标；
- 后续主模型筛选以 `mIoU_fg` 为主，但同时检查逐类 IoU、Precision、Recall、F1、训练—验证差距和曲线，不能只看单个汇总值。

证据等级：

- **A：原始产物已核验**——本地存在并已检查结果文件；
- **B：旧文档记录**——有现有总结中的数值，但本地未找到对应原始产物；
- **C：未完成**——只有计划或代码，没有可登记的正式结果。

旧的 `实验现状总结.md` 保留为历史混合档案，不回写、不覆盖；从现在开始以本文件作为 v6 overlap40 数值的唯一主台账。

## 2. 数据集与评价口径

### 2.1 数据配置

| 项目 | 值 |
|---|---:|
| 数据集 | `dataset_v6_random811_overlap40` |
| Tile 尺寸 | 512 × 512 |
| Stride | 307 |
| 名义相邻重叠 | 约 40% |
| Train / Val / Test | 1598 / 200 / 200 |
| 总 Tile 数 | 1998 |
| Train / Val / Test 纯背景 | 514 / 64 / 62 |
| 总纯背景 | 640 / 1998（32.03%） |
| 输入通道 | 5：WAC、DEM、Slope、TPI、Profile Curvature |
| 类别 | 0 Background；1 WR；2 Rille；3 Fault；4 Graben |
| 固定随机种子 | 42（本表所列首轮实验） |
| 正式训练轮数 | 80 epochs |
| 类别权重 | `[0.15, 1.0, 2.73, 1.98, 2.12]` |

Train-only 归一化统计：

```text
mean = [0.15665339073973303, 0.6052870962271574, 0.22171011101838023,
        0.5087022443378417, 0.46687463729626205]
std  = [0.07239327406001447, 0.35159567816693277, 0.23999408652260576,
        0.18305312443820845, 0.18653673179588806]
```

### 2.2 协议独立性限制

数据文件、标签范围、配准、数量分层和 Train-only mean/std 审计通过，但 Test 不是空间独立测试集：

| 审计项 | Val | Test |
|---|---:|---:|
| 与 Train 存在像素相交的 Tile | 99.5% | 100% |
| Tile 足迹被 Train 覆盖 | 87.67% | 87.52% |
| Test 对象也出现在 Train | — | WR 98.62%；Rille 96.30%；Fault 98.68%；Graben 100% |
| 跨 split 单对最大共享 | — | 最高 99.41% |

因此，本文件中的 Test 结果只能称为**同研究区、重叠 Tile 的内部参考结果/乐观上界**。它可以用于当前固定协议下的工程筛选，但不能单独支撑未见构造、对象独立或空间泛化结论。

另有 320 张 Tile 带不超过 5% 的 NoData 边缘；现有加载链会把其中 Train/Val/Test 约 3931/467/446 个正标签像素改为背景。该问题对所有当前模型共享，但属于待修复的数据口径缺陷。

审计证据：`results/data_audit_v6_random_overlap40/protocol_audit_2026-08-24.md`、`results/data_audit_v6_random_overlap40/audit_report.json`。

## 3. 已完成实验总表

### 3.1 Test 结果

以下表只比较真正执行过 Test 的完整模型。两个新模块尚未执行 Test，不进入本表。

| 实验 | 骨干/结构 | 选模指标 | 最佳 epoch | Test loss | mIoU_all | mIoU_fg | mF1_all | WR IoU | Rille IoU | Fault IoU | Graben IoU | 证据 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| U-Net | ResNet50 | 未在本地产物核验 | — | 0.212511 | 0.7436 | 0.6824 | 0.8470 | 0.6775 | 0.7458 | 0.6088 | 0.6974 | B |
| DeepLabV3+（旧正式运行） | ResNet50 | `val_mIoU_all` | 74 | 0.140064 | 0.752125 | 0.692916 | 0.853089 | 0.715951 | 0.727798 | 0.615757 | 0.712158 | A |
| LTL-Net + TPD | ResNet50 | `val_mIoU_all` | 本地产物未记录 | 0.140573 | 0.755200 | 0.696738 | 0.855141 | 0.716364 | 0.745345 | 0.617287 | 0.707957 | A |
| DeepLabV3+（统一选模重跑） | ResNet50 | `val_mIoU_fg` | 76 | 0.12883 | 0.7531 | 0.6942 | 0.8538 | 0.7202 | 0.7222 | 0.6183 | 0.7159 | B |
| D-LinkNet-R50（adapted） | ResNet50 | `val_mIoU_fg` | 59 | 0.29488 | 0.6795 | 0.6030 | 0.7988 | 0.6079 | 0.6785 | 0.5012 | 0.6245 | B |

说明：统一选模 DeepLabV3+ 是后续比较应采用的正式基线，但其下载产物当前不在本地，因此证据暂为 B；旧 DeepLabV3+ 保留用于追溯，不与重跑结果合并。

### 3.2 Test 详细指标（本地原始产物）

| 实验 | Background IoU | mPrecision_all | mRecall_all | mF1_all | Accuracy |
|---|---:|---:|---:|---:|---:|
| DeepLabV3+（旧正式运行） | 0.988962 | 0.815997 | 0.895794 | 0.853089 | 0.989205 |
| LTL-Net + TPD | 0.989046 | 0.821621 | 0.893442 | 0.855141 | 0.989287 |

逐类 Precision / Recall / F1：

| 实验 | 指标 | Background | WR | Rille | Fault | Graben |
|---|---|---:|---:|---:|---:|---:|
| DeepLabV3+（旧正式运行） | Precision | 0.996380 | 0.788662 | 0.785560 | 0.699666 | 0.809719 |
|  | Recall | 0.992528 | 0.885917 | 0.908241 | 0.836986 | 0.855296 |
|  | F1 | 0.994450 | 0.834465 | 0.842457 | 0.762190 | 0.831884 |
| LTL-Net + TPD | Precision | 0.996425 | 0.783952 | 0.802720 | 0.707723 | 0.817283 |
|  | Recall | 0.992568 | 0.892578 | 0.912494 | 0.828492 | 0.841079 |
|  | F1 | 0.994493 | 0.834746 | 0.854094 | 0.763361 | 0.829010 |

### 3.3 Val 结果与过拟合诊断

| 实验 | 物理 batch / 累积 | 最佳 Val epoch | 最佳 Val mIoU_fg | 同 epoch Train mIoU_fg | Train−Val | 最低 Val loss（epoch） | 最终 Val mIoU_fg | 峰值−最终 | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepLabV3+（旧正式运行） | 4 / 1 | 74 | 0.703326 | 0.798314 | 0.094988 | 0.104420（19） | 0.700091 | 0.003234 | A |
| DeepLabV3+（统一选模重跑） | 4 / 1 | 76 | 0.6983 | — | — | — | — | — | B |
| D-LinkNet-R50（adapted） | 2 / 2 | 59 | 0.5925 | — | — | — | — | — | B |
| DSConv / Dynamic Snake | 2 / 2 | 69 | 0.633572 | 0.787906 | 0.154334 | 0.132158（15） | 0.590991 | 0.042580 | A |
| Gated Boundary | 2 / 2 | 66 | 0.664128 | 0.793032 | 0.128904 | 0.167811（28） | 0.652185 | 0.011943 | A |
| DeepLabV3+（模块脚本 batch2 控制） | 2 / 2 | 75 | 0.621450 | 0.753669 | 0.132218 | 0.149193（21） | 0.516317 | 0.105133 | A |
| Gated，boundary weight=0 | 2 / 2 | 79 | 0.669135 | 0.792398 | 0.123263 | 0.128081（25） | 0.646148 | 0.022987 | A |
| Gated，boundary weight=0（公平 batch4） | 4 / 1 | 78 | **0.718582** | 0.811205 | 0.092623 | 0.108832（11） | 0.714034 | 0.004549 | A |

模块运行的有效 batch 均为 4，但物理 batch 为 2；DeepLab 基线物理 batch 为 4。BatchNorm 统计不同，因此当前模块结果不是完全严格的单变量消融。

## 4. 两个新模块的完整记录

### 4.1 DSConv / Dynamic Snake refinement

**结构与配置**

| 项目 | 值 |
|---|---|
| 模型 | DeepLabV3+-ResNet50 + residual Dynamic Snake refinement |
| 模块位置 | DeepLab decoder 的 256 通道输出与 segmentation head 之间 |
| 模块参数 | hidden channels 64；kernel size 9；extend scope 1.0 |
| 参数量 | 26,882,845 |
| Seed / epochs | 42 / 80 |
| Batch | physical 2；gradient accumulation 2；effective 4 |
| Optimizer 学习率 | 5e-5 |
| 选模指标 | `val_mIoU_fg` |
| 自动 Test | 否 |
| Git commit | `f543485e62afee05b357ba9d80b5d9b8d9fc7c85` |
| Kaggle 数据路径 | `/kaggle/input/datasets/yuanssy/datav6-overlap40/dataset_v6_random811_overlap40` |

**最佳 Val（epoch 69）**

| 指标 | Background | WR | Rille | Fault | Graben | 汇总 |
|---|---:|---:|---:|---:|---:|---:|
| IoU | 0.985398 | 0.688323 | 0.628326 | 0.517234 | 0.700404 | mIoU_all 0.703937；mIoU_fg 0.633572 |
| Precision | 0.995048 | 0.794998 | 0.676037 | 0.599856 | 0.752358 | mean 0.763659 |
| Recall | 0.990254 | 0.836861 | 0.899020 | 0.789706 | 0.910254 | mean 0.885219 |
| F1 | 0.992645 | 0.815393 | 0.771744 | 0.681811 | 0.823809 | mean 0.817080 |

最佳 epoch 的 Val loss 为 0.162305，Accuracy 为 0.985669。最低 Val loss 出现在 epoch 15（0.132158），最终 epoch 的 Train/Val `mIoU_fg` 为 0.791124/0.590991，差距扩大到 0.200132；最终 Train/Val loss 为 0.017759/0.308284。曲线表现为明显过拟合。

### 4.2 Semantic-Gated Boundary refinement

**结构与配置**

| 项目 | 值 |
|---|---|
| 模型 | DeepLabV3+-ResNet50 + semantic-gated shape stream + boundary head |
| 模块位置 | 以 ResNet 1/2 分辨率浅层特征为 detail，以 decoder 输出为 semantic；深层语义门控形状流后回融 decoder |
| Shape channels | 32 |
| Boundary loss weight | 0.2 |
| 参数量 | 27,396,374 |
| Seed / epochs | 42 / 80 |
| Batch | physical 2；gradient accumulation 2；effective 4 |
| Optimizer 学习率 | 5e-5 |
| 选模指标 | `val_mIoU_fg` |
| 自动 Test | 否 |
| Git commit | `f543485e62afee05b357ba9d80b5d9b8d9fc7c85` |
| Kaggle 数据路径 | `/kaggle/input/datasets/changyasong/datav6-overlap40/dataset_v6_random811_overlap40` |

**最佳 Val（epoch 66）**

| 指标 | Background | WR | Rille | Fault | Graben | 汇总 |
|---|---:|---:|---:|---:|---:|---:|
| IoU | 0.986182 | 0.651318 | 0.710600 | 0.553845 | 0.740748 | mIoU_all 0.728539；mIoU_fg 0.664128 |
| Precision | 0.993510 | 0.808984 | 0.808008 | 0.653257 | 0.806639 | mean 0.814080 |
| Recall | 0.992577 | 0.769687 | 0.854956 | 0.784457 | 0.900677 | mean 0.860471 |
| F1 | 0.993043 | 0.788846 | 0.830819 | 0.712871 | 0.851069 | mean 0.835330 |

最佳 epoch 的 Val total/semantic/boundary loss 为 0.235605/0.207843/0.138809，Accuracy 为 0.986471。最低 Val total loss 出现在 epoch 28（0.167811），最终 epoch 的 Train/Val `mIoU_fg` 为 0.797030/0.652185，差距为 0.144844；最终 Val total loss 为 0.310275。该模块同样过拟合，但峰值后的退化小于 DSConv。

### 4.3 模块相对基线的结果

| 模块 | 最佳 Val mIoU_fg | 相对统一选模 DeepLab（0.6983，证据 B） | 相对本地旧 DeepLab 曲线峰值（0.703326，证据 A） | 当前判断 |
|---|---:|---:|---:|---|
| DSConv | 0.633572 | -0.064728（-6.47 pp） | -0.069754（-6.98 pp） | 当前配置无正收益，且过拟合明显 |
| Gated Boundary | 0.664128 | -0.034172（-3.42 pp） | -0.039198（-3.92 pp） | 当前配置无正收益，且存在过拟合 |

这两个判断只针对当前实现、训练配置和 seed=42。由于未运行 Test，不能写成 Test 下降；由于物理 batch 不一致，也不能外推为对应模块思想普遍无效。当前证据足以决定：**暂不把这两个配置并入总体架构、不补 Test；但在把退化归因于模块结构前，必须先完成同物理 batch 的 DeepLab 控制。**

### 4.4 退化原因诊断（2026-08-24）

#### 当前证据能回答什么

训练时虽然累计了 5×5 confusion matrix，但保存结果前只提取了逐类 IoU、Precision、Recall 和 F1，原始矩阵没有写入产物。因此当前可以判断某一类别更偏向 FP 或 FN，却不能判断具体的错误类别对，例如不能区分“Background→Fault”和“Graben→Fault”。

按 TP 归一化后的错误负担：

| 模块 | 类别 | FP/TP | FN/TP | 主要现象 |
|---|---|---:|---:|---|
| DSConv | WR | 0.258 | 0.195 | 相对均衡，误检略多 |
| DSConv | Rille | 0.479 | 0.112 | 明显过预测 |
| DSConv | Fault | 0.667 | 0.266 | 最严重的过预测 |
| DSConv | Graben | 0.329 | 0.099 | 高召回、误检偏多 |
| Gated Boundary | WR | 0.236 | 0.299 | 漏检偏多 |
| Gated Boundary | Rille | 0.238 | 0.170 | 误检略多 |
| Gated Boundary | Fault | 0.531 | 0.275 | 明显过预测 |
| Gated Boundary | Graben | 0.240 | 0.110 | 高召回、误检偏多 |

DSConv 在 80 个逐轮对齐点中从未超过旧 DeepLab；同为 epoch 69 时，两者 Val `mIoU_fg` 分别为 0.633572/0.701625，相差 6.81 pp。Gated 在 epoch 66 的 Train `mIoU_fg=0.793032`，几乎等于 DeepLab 的 0.795541，但 Val 为 0.664128/0.698903，相差 3.48 pp，说明其主要问题是泛化而不是训练集拟合不足。

后 20 epoch 的 Val `mIoU_fg` 波动标准差：

| DeepLab（physical batch 4） | DSConv（physical batch 2） | Gated（physical batch 2） |
|---:|---:|---:|
| 0.00117 | 0.01426 | 0.00777 |

epoch 31–80 的 Val loss 变异系数分别为 0.070、0.202、0.321。两个模块在学习率已很低时仍明显波动，优先提示小物理 batch 下的 BatchNorm running statistics 不稳定；这是强证据支持的假设，但尚未由控制实验确认。

#### Gated Boundary 的特定问题

最佳 epoch 的 boundary loss 加权贡献为 `0.2×0.138809=0.027762`，约占 total loss 的 11.78%；semantic CE 本身仍有 0.207843，明显高于同 epoch DeepLab 的 0.148503，因此退化不能只解释为“多加了一个 loss”。

当前 boundary target 会先做 5×5 膨胀，再降采样至 1/2 分辨率。合成细线检查显示，原图宽 1–6 px 的线在降采样后其前景像素都会 100% 落入 boundary target；对本任务而言，辅助任务容易从“学轮廓”退化为“学习膨胀后的整条细线带”。这与 Fault/Graben 高 Recall、低 Precision 的现象一致，但仍需预测图和原始混淆矩阵确认。

此外，`semantic_fusion` 直接以随机初始化分支执行 `semantic + residual`，末层没有零初始化或从 0 开始的残差系数；边界辅助梯度也会经过 shape stream 回传至 decoder/encoder，存在语义任务与类别无关边界任务冲突的可能。

#### DSConv 的特定问题

DSConv 最终 Train/Val `mIoU_fg` 为 0.791124/0.590991，差距 20.01 pp；前景各类均为 FP 负担大于 FN，符合动态采样将响应向线周围扩散的表现。

其 kernel size 9 的累计动态偏移理论上可达约 4 个 decoder 像素，折合输入约 16 px；偏移没有显式幅度正则，越界采样又使用 zero padding。残差融合的最后 BatchNorm 同样未零初始化，因此模型从 epoch 1 就不是 DeepLab 的近似恒等起点。暂未发现坐标轴交换或输出尺寸错误，且模块只增加约 19.8 万参数，所以“参数量过大”不是优先解释。

#### 必须先完成的最小诊断

1. 不碰 Test，使用已保存的 DeepLab、DSConv、Gated 最佳 checkpoint 在同一 Val 上补算原始/行归一 5×5 confusion matrix、前景/背景二元矩阵、逐 Tile 指标和固定 error map。
2. 在模块训练脚本下补跑纯 DeepLab：同一 Kaggle 数据副本、physical batch 2、accumulation 2、seed 42、同损失与 `val_mIoU_fg` 选模，用来分离 BatchNorm/batch 混杂。
3. 核对两个 Kaggle 账号数据副本的 manifest、normalization 和文件哈希；当前仅凭同名叶目录不能证明字节一致。
4. Gated 只做一个 `boundary_weight=0` 控制，区分融合结构退化与边界辅助监督冲突；DSConv 只做恒等初始化/较小 offset 控制。完成这些控制前不继续叠加新的边界或动态形变模块。

新模块方向由混淆矩阵决定：若 Background→Foreground FP 主导，优先做语义/模态一致性抑制假阳性；若前景类别互相混淆，优先做双视图地形—影像融合或类别上下文；若 FN 和断裂主导，才做方向上下文或拓扑连续性；只有边界带指标明显更差时，才继续边界模块。

### 4.5 Gated 控制与公平 batch4 复验（2026-08-25）

本节结果均只使用 Train/Val，按 `val_mIoU_fg` 选模，未自动评估 Test。新运行保存的 `dataset_protocol.json`、`dataset_summary.json`、`normalization_stats.json`、`tile_manifest.csv` 四个 SHA-256 与冻结配置完全一致，因此两个新控制实验的数据元文件版本可核验一致。

| 实验 | physical batch / accum | Boundary weight | 最佳 epoch | Val mIoU_all | Val mIoU_fg |
|---|---:|---:|---:|---:|---:|
| DeepLab 模块脚本控制 | 2 / 2 | — | 75 | 0.694051 | 0.621450 |
| Gated | 2 / 2 | 0.2 | 66 | 0.728539 | 0.664128 |
| Gated 无边界辅助损失 | 2 / 2 | 0 | 79 | 0.732670 | 0.669135 |
| 旧 DeepLab 正式基线 | 4 / 1 | — | 74 | 0.760405 | 0.703326 |
| **Gated 无边界辅助损失，公平 batch4** | **4 / 1** | **0** | **78** | **0.772777** | **0.718582** |

Gated batch4 最佳 Val 逐类结果：

| 指标 | Background | WR | Rille | Fault | Graben |
|---|---:|---:|---:|---:|---:|
| IoU | 0.989558 | 0.763677 | 0.758315 | 0.595611 | 0.756727 |
| Precision | 0.995810 | 0.852603 | 0.829691 | 0.697337 | 0.828129 |
| Recall | 0.993695 | 0.879836 | 0.898113 | 0.803263 | 0.897715 |
| F1 | 0.994751 | 0.866006 | 0.862547 | 0.746561 | 0.861519 |

同为 physical batch4、seed=42 时，Gated 无边界辅助损失相对旧 DeepLab 的 Val `mIoU_fg` 从 0.703326 提升到 0.718582，即 **+0.015257（+1.53 pp）**；`mIoU_all` 提升 **+1.24 pp**。因此 Gated 融合登记为**单种子有效候选模块1**。它尚未经过多种子复验，不能写成稳定增益或最终 Test 增益。

关闭边界辅助损失后，batch2 Gated 相对原 batch2 Gated 提升约 0.50 pp；由于旧运行没有保存数据指纹，该差值作为支持性证据，不单独作严格因果结论。`boundary_weight=0` 时产物仍记录原始 boundary loss，但 `total loss=semantic loss`，该项未参与反向优化。

物理 batch 对结果影响显著：模块脚本 DeepLab 的 batch2 `mIoU_fg=0.621450`，旧 batch4 DeepLab 为 0.703326；Gated 无边界辅助损失从 batch2 的 0.669135 升至 batch4 的 0.718582。后续正式模块筛选统一使用 physical batch4，不再把 batch2 模块结果与 batch4 基线直接比较。

Gated batch4 在 epoch 78 达峰值，最终 epoch 为 0.714034，回落 0.45 pp；后20轮 Val `mIoU_fg` 标准差约 0.00167、极差约 0.00545。最佳 epoch 的 Train−Val gap 为 9.26 pp，仍有过拟合，但稳定性明显优于旧 batch2 Gated。当前最弱类仍是 Fault（IoU 0.595611），模块2方向必须结合统一 Val 混淆矩阵进一步确定。

### 4.6 CMCR 独立消融与 Gated+CMCR 组合（2026-08-25）

两项实验均使用 overlap40、ResNet50、seed42、80 epochs、physical batch4/accum1、学习率 5e-5、`val_mIoU_fg` 选模，不自动评估 Test。两份结果均已下载并核验 `config.json`、`metrics.json`、80轮 `history.csv/json`、曲线及 `best_model.pth`；Git commit 均为 `2389489`。CMCR 以零初始化的5类残差 logit 接入，对应模型初始输出严格等于母模型。

| 实验 | 唯一结构变量 | 最佳 epoch | Val mIoU_all | Val mIoU_fg | 相对旧 DeepLab batch4 | 相对 Gated batch4 |
|---|---|---:|---:|---:|---:|---:|
| DeepLab+CMCR（B） | DeepLab 增加 CMCR | 78 | 0.765646 | 0.709767 | +0.006441（+0.64 pp） | -0.008815（-0.88 pp） |
| Gated+CMCR（A+B） | Gated 增加 CMCR | 74 | 0.764456 | 0.708320 | +0.004994（+0.50 pp） | -0.010262（-1.03 pp） |

逐类最佳 Val IoU：

| 模型 | WR | Rille | Fault | Graben |
|---|---:|---:|---:|---:|
| Gated（A） | 0.763688 | 0.758272 | 0.595609 | 0.756727 |
| CMCR（B） | 0.766884 | 0.749997 | 0.573634 | 0.748465 |
| Gated+CMCR（A+B） | 0.748177 | 0.759213 | 0.582196 | 0.743673 |

A+B 相对 A 的变化为 WR -1.55 pp、Rille +0.09 pp、Fault -1.34 pp、Graben -1.31 pp。只有 Rille 基本持平，其余三类均退化，因此不能把组合结果解释为某一弱类换取整体性能。

统一 FP32 Val 诊断复现了保存指标（B 0.709745，A+B 0.708315，A 0.718574；差异仅为浮点/累计精度）。前景/背景二元混淆统计：

| 模型 | Background→Foreground FP | Foreground→Background FN | 容差2 px边界 F1 |
|---|---:|---:|---:|
| Gated（A） | 320,316 | 212,183 | 0.728015 |
| CMCR（B） | 356,423 | 196,140 | 0.704784 |
| Gated+CMCR（A+B） | 339,562 | 221,421 | 0.707684 |

CMCR 单独相对 A 表现为更高召回、更多背景假阳性；组合相对 CMCR 虽减少16,861个 FP，却增加25,281个 FN。更关键的是，组合相对 A 同时增加19,246个 FP和9,238个 FN，且容差2 px边界 F1下降2.03 pp，说明联合训练没有保留任一分支的互补优势。Gated gate 均值从 A 的0.47235变为组合的0.47166，CMCR gate 均值从 B 的0.51394变为组合的0.51786，极端激活比例均很低；没有发现简单的 gate 饱和或分支完全关闭。

训练曲线方面，A/B/A+B 最佳 epoch 的 Train−Val `mIoU_fg` gap 分别为9.26/9.07/9.33 pp，均存在相近程度的过拟合。后20轮 Val `mIoU_fg` 标准差分别为0.00167/0.00153/0.00236，A+B 波动最大；最终相对各自峰值分别回落0.45/0.33/0.51 pp。组合退化不能只归因于一般过拟合，因为其误差结构和边界指标也系统性差于 A。

结论：CMCR 登记为**单种子、小幅正收益候选模块2**，但与 Gated 联合从头训练存在负交互。当前总体架构保留 Gated（A），淘汰现有 A+B 组合；CMCR 可作为独立消融证据，不能宣称其提高了最终 Gated 模型。若继续验证 CMCR，只允许做“冻结已训练 A、仅训练零初始化 CMCR 残差”的两阶段控制，不能重复无约束联合训练。

### 4.7 FEC 独立消融与 Gated+FEC 组合（2026-08-26）

Foreground Evidence Calibration（FEC）受 FarSeg（CVPR 2020）显式前景建模与抑制遥感复杂背景假警报的思想启发，但不是对 FarSeg 的 F-S relation 或 F-A optimization 的复现。当前实现从 decoder 特征预测 Background/Foreground 二元证据，以有界、零初始化的残差校准背景与四类前景 logits；辅助前景 BCE 权重为0.1。两项实验均为 overlap40、ResNet50、seed42、80 epochs、physical batch4/accum1、学习率5e-5、`val_mIoU_fg` 选模，不评估 Test，Git commit 为 `982d36d`。

下载产物存在目录命名放反：本地 `results/v6_overlap40/FEC/` 内实际 `model=gated_fec`（A+C），而 `results/v6_overlap40/gate+FEC/` 内实际 `model=deeplab_fec`（C）。以下判断一律以 `config.json/metrics.json` 的模型字段为准，不按文件夹名推断。

| 实验 | 最佳 epoch | Val mIoU_all | Val mIoU_fg | 相对旧 DeepLab batch4 | 相对 Gated batch4 |
|---|---:|---:|---:|---:|---:|
| DeepLab+FEC（C） | 74 | 0.766086 | 0.710308 | +0.006982（+0.70 pp） | -0.008274（-0.83 pp） |
| Gated+FEC（A+C） | 74 | 0.772246 | 0.717935 | +0.014609（+1.46 pp） | -0.000647（-0.06 pp） |

逐类 Val IoU 的统一 FP32 复算：

| 模型 | WR | Rille | Fault | Graben |
|---|---:|---:|---:|---:|
| Gated（A） | 0.763688 | 0.758272 | 0.595609 | 0.756727 |
| FEC（C） | 0.762095 | 0.735767 | 0.583905 | 0.759391 |
| Gated+FEC（A+C） | 0.759906 | 0.751541 | 0.595409 | 0.764742 |

A+C 相对 A 为 WR -0.38 pp、Rille -0.67 pp、Fault -0.02 pp、Graben +0.80 pp。它没有改善设计时瞄准的 Fault，只把部分收益转移到 Graben，整体未超过 A。

前景/背景二元混淆与边界指标：

| 模型 | Background→Foreground FP | Foreground→Background FN | 容差2 px边界 F1 |
|---|---:|---:|---:|
| Gated（A） | 320,316 | 212,183 | 0.728015 |
| FEC（C） | 354,564 | 196,195 | 0.705161 |
| Gated+FEC（A+C） | 322,421 | 213,713 | 0.722651 |

C 单独相对 A 增加34,248个 FP、减少15,988个 FN，实际行为是以更多背景误报换取前景召回，而不是抑制假阳性。A+C 相对 A 同时增加2,105个 FP和1,530个 FN，容差2 px边界 F1下降0.54 pp。checkpoint 中二者学得的有界校准强度分别为 C 0.35843、A+C 0.22855，说明校准分支已参与决策，并非保持零初始化或完全关闭。

训练曲线方面，C/A+C 最佳 epoch 的 Train−Val `mIoU_fg` gap 分别为9.16/8.96 pp，后20轮标准差为0.00162/0.00185，最终较峰值回落0.49/0.32 pp；属于与 A 相近的轻度过拟合和稳定波动，不能解释其方向性误差变化。前景辅助损失在最佳 epoch 的加权贡献分别约0.00833/0.00922，没有压过语义主损失。

结论：FEC 登记为**单种子、小幅正收益候选模块C**，但现有 A+C 没有组合增益且未实现预期的总体假阳性抑制。当前总体架构仍保留 A，淘汰现有 A+C；不得写成“FEC 提升最终模型”。不建议继续扫描 foreground weight，因为当前方向性证据显示问题不只是损失权重，而是二元证据在联合语义优化下形成了召回—精度重新分配。

### 4.8 冻结 Gated 后训练 CMCR 的两阶段控制（2026-08-26）

该实验加载已核验的 Gated batch4、`boundary_weight=0` checkpoint（SHA-256 `f75fc145626053b7a5da39905295c289a6fc8665f1c555a2729c83ef0586891f`），冻结 Gated 母模型，仅训练零初始化 CMCR 的42,133个参数。数据为同一 overlap40 指纹，ResNet50、seed42、physical batch4/accum1、学习率1e-4、最多40 epochs、patience8、`val_mIoU_fg` 选模，不评估 Test；Git commit 为 `982d36d`。训练在第37轮早停，最佳为第29轮。

统一 FP32 Val 诊断独立复算 `mIoU_fg=0.721621`，与训练保存值0.721608仅差0.0013 pp。与同一母 checkpoint 的统一诊断结果比较：

| 模型 | Val mIoU_all | Val mIoU_fg | WR IoU | Rille IoU | Fault IoU | Graben IoU |
|---|---:|---:|---:|---:|---:|---:|
| Gated（A） | 0.772771 | 0.718574 | 0.763688 | 0.758272 | 0.595609 | 0.756727 |
| 冻结A后训练CMCR（A→B） | 0.775255 | 0.721621 | 0.764534 | 0.761133 | 0.599586 | 0.761233 |
| A→B 相对 A | +0.002484 | **+0.003048（+0.30 pp）** | +0.000846 | +0.002861 | +0.003977 | +0.004506 |

四个前景类别 IoU 全部提高，组合收益不是牺牲 Fault 换取其他类别。误差结构显示 CMCR 主要进行了精度—召回再平衡：

| 模型 | Background→Foreground FP | Foreground→Background FN | 容差2 px边界 F1 | 容差4 px边界 F1 |
|---|---:|---:|---:|---:|
| Gated（A） | 320,316 | 212,183 | 0.728015 | 0.855276 |
| 冻结A后训练CMCR（A→B） | 295,348 | 225,546 | 0.737319 | 0.858433 |
| 变化 | -24,968 | +13,363 | +0.009304 | +0.003157 |

相对 A，A→B 减少约7.8%的前景假阳性，同时增加约6.3%的前景漏检；四类 Precision 均上升、Recall 均下降，但最终四类 IoU 与多尺度边界 F1 均改善。CMCR gate 均值0.51487、标准差0.11239，极低/极高激活比例均不足0.1%，没有饱和或关闭。

第1轮 Val `mIoU_fg=0.720337`，已经高于 A；随后缓慢上升并在第29轮达到0.721608，之后8轮没有再创新高而早停，因此最佳值不是孤立尖峰。但最佳轮 Train−Val `mIoU_fg` gap 为11.40 pp，Train loss约0.0131而 Val loss约0.1619，存在明显泛化差距。该差距部分继承自冻结母模型，仍要求用第二个 CMCR 随机初始化/数据顺序复验，不能据单种子宣称稳定增益。

结论：此前 A+B 从头联合训练较 A 下降1.03 pp，而冻结 A、仅训练 B 后较 A 提升0.30 pp并首次达到 Val `mIoU_fg>0.72`。这支持“CMCR 有条件有效，主要问题是联合共适应冲突”，当前保留**两阶段 A→B**作为最佳总体架构候选；其训练规程必须作为方法的一部分明确报告，不能与从头联合 A+B 混写。待完成第二种子复验与架构冻结后，才能进行一次最终 Test。

## 5. 当前可支持的模型结论

1. 统一选模重跑的 DeepLabV3+-ResNet50 是当前主基线：Test `mIoU_fg=0.6942`（证据 B，待补下载产物）。
2. LTL-Net+TPD 的旧单种子 Test `mIoU_fg=0.696738`，相对旧 DeepLab 为 +0.003822；增益只有 0.38 pp，主要来自 Rille（+1.75 pp），Graben 反而下降 0.42 pp，因此 TPD 不能登记为稳定有效模块。
3. U-Net 的 Test `mIoU_fg=0.6824`，低于统一 DeepLab 约 1.18 pp；但 Rille IoU 0.7458，显示结构存在类别偏好。
4. D-LinkNet-R50 adapted 的 Test `mIoU_fg=0.6030`，比统一 DeepLab 低 9.12 pp；由于其约 1.80 亿参数、物理 batch=2 且是 R50 适配版，只能说明当前适配和配置较差，不能推断原始 D-LinkNet 方法普遍较差。
5. DSConv 与旧 Gated Boundary 的 batch2 配置不进入总体架构；同物理 batch2 控制证明二者的旧结果受到严重 batch/BN 混杂，不能作为模块思想无效的证据。
6. Gated 无边界辅助损失在 physical batch4、seed42 上取得 Val `mIoU_fg=0.718582`，相对同为 batch4 的旧 DeepLab 峰值提升 1.53 pp，登记为单种子有效候选模块1；尚未完成多种子和最终冻结 Test，不能称为稳定增益。
7. CMCR 单独取得 Val `mIoU_fg=0.709767`，相对旧 DeepLab batch4 提升0.64 pp，登记为单种子、小幅正收益候选模块2。
8. Gated+CMCR 仅为0.708320，较 Gated 下降1.03 pp，并同时增加前景 FP 与 FN；现有联合结构淘汰，当前最佳总体模型仍是 Gated。
9. FEC 单独取得 Val `mIoU_fg=0.710308`，相对旧 DeepLab batch4提升0.70 pp；但 Gated+FEC 为0.717935，较 Gated低0.06 pp，且没有改善 Fault 或总体假阳性，因此现有 A+C 淘汰。
10. 冻结已训练 Gated、仅训练 CMCR 的两阶段 A→B 取得统一复算 Val `mIoU_fg=0.721621`，较 A 提升0.30 pp，四个前景类别 IoU 均提高；当前登记为单种子最佳总体架构候选，尚需第二种子复验，不能写成稳定或 Test 增益。

## 6. 证据路径与缺失项

### 6.1 已核验的本地原始产物

- DeepLabV3+ 旧正式运行：`results/v6_overlap40/Deeplab_seed42_formal180/`
- LTL-Net + TPD：`results/v6_overlap40/LTLNet_seed42_formal180/`
- DSConv：`results/v6_overlap40/M2/DSConv/`
- Gated Boundary：`results/v6_overlap40/M3/gated/`
- DeepLab batch2 控制：`results/v6_overlap40/deeplab_resnet50_batch2_control/`
- Gated 无边界损失 batch2：`results/v6_overlap40/gated_boundary_resnet50_no_boundary/`
- Gated 无边界损失 batch4：`results/v6_overlap40/gated_boundary_resnet50_bw0_batch4/`
- CMCR 单独：`results/v6_overlap40/CMCR/`
- Gated+CMCR：`results/v6_overlap40/gate+CMCR/`
- 统一 Val 诊断（含混淆矩阵、边界指标和错误图）：`results/v6_overlap40/val_diagnostics/{gated_without_boundary_loss_batch4,CMCR,gate+CMCR}/`
- FEC 两项产物（目录名放反，以内部模型字段为准）：`results/v6_overlap40/{FEC,gate+FEC}/`
- FEC 统一 Val 诊断：`results/v6_overlap40/val_diagnostics/{deeplab_fec,gated_fec}/`
- 冻结 Gated 后训练 CMCR：`results/v6_overlap40/gated_cmcr_resnet50_frozen_gated_seed42_batch4/`
- 冻结 A→B 统一 Val 诊断：`results/v6_overlap40/gated_cmcr_resnet50_frozen_gated_seed42_batch4/val_diagnostics/`
- 数据协议审计：`results/data_audit_v6_random_overlap40/`

### 6.2 仅有旧文档记录、待补原始产物

- U-Net / ResNet50 seed=42；
- DeepLabV3+ 统一 `val_mIoU_fg` 选模重跑；
- D-LinkNet-R50 adapted。



## 7. 后续追加模板

每个新实验按以下字段追加，先核验产物再写结论：

```text
实验名 / run_name：
模型、骨干、模块位置：
唯一变量与对照：
数据集路径与版本：
Git commit：
seed / epochs：
physical batch / accumulation / effective batch：
optimizer / learning rate / loss / class weights：
checkpoint 选择指标与最佳 epoch：
最佳 Val：mIoU_all、mIoU_fg、各类 IoU、Precision、Recall、F1、loss：
同 epoch Train mIoU_fg 与 Train−Val gap：
Test 是否执行：
若执行 Test：mIoU_all、mIoU_fg、各类 IoU、Precision、Recall、F1、loss：
本地证据目录：
证据等级：A / B / C：
可比性限制：
结论：保留 / 淘汰 / 需复验：
```
