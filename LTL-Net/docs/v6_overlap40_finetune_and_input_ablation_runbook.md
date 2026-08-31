# v6 overlap40：联合微调与输入影像消融运行说明

## 1. 实验边界

- 所有实验只使用 Train 训练、Val 选模。
- runner 仅核验 Test 文件数量，不读取 Test 影像、标签或指标。
- 两项联合微调从同一完整 A′→CMCR seed1337 checkpoint 开始。
- 六项影像消融均为 DeepLabV3+-ResNet50、seed42、physical batch4、80 epochs。
- 原始 GeoTIFF 不修改、不复制；无效通道在归一化后置零（等价于训练集均值填充）。

## 2. 联合微调 F0/F1

共同初始checkpoint：

```text
SHA-256: 012728c1f4deb888eff474fd49d6ea1478fb7803f61348bce8f3b0c2e79e3c04
初始Val mIoU-FG: 0.7176050991
```

| 实验 | Notebook | 唯一损失变量 |
|---|---|---:|
| F0 | `notebooks/autodl_v6_overlap40_joint_finetune_rezero_cmcr_ce_seed1337.ipynb` | `lovasz_weight=0.0` |
| F1 | `notebooks/autodl_v6_overlap40_joint_finetune_rezero_cmcr_lovasz02_seed1337.ipynb` | `lovasz_weight=0.2` |

两项均冻结ResNet50编码器和全部BatchNorm，只训练Decoder、分割头、A′与CMCR，共3,918,988个参数。每个Notebook只允许按实际服务器位置修改：

```python
INIT_CHECKPOINT = Path('.../best_model.pth')
```

判断顺序：

1. epoch 0不在`0.7176050991±0.0001`内，停止并检查checkpoint/数据。
2. F1必须同时超过当前checkpoint和F0。
3. F1相对F0低于0.10 pp，不登记为有效增益。
4. F1相对F0达到0.20 pp后，再生成seed42复验，不提前查看Test。

### F1 seed42复验

seed1337中F1相对F0提高0.4835 pp，进入seed42复验。使用：

```text
notebooks/autodl_v6_overlap40_joint_finetune_rezero_cmcr_lovasz02_seed42.ipynb
```

seed42初始checkpoint必须为独立训练的A′→CMCR完整权重：

```text
SHA-256: 2920e0a91b4d9998069e70ee4137669462e6e41bb6753c1749277616dd657e7f
初始Val mIoU-FG: 0.7114108652
```

除seed及其匹配的初始checkpoint外，训练协议与seed1337 F1完全相同。Loss曲线用于诊断过拟合和概率置信度，不改变预注册的Val mIoU-FG选模规则。若seed42相对自身epoch 0仍为正增益，则锁定F1协议；若退化，再补跑seed42 F0定位是联合微调还是Lovasz导致。

## 3. DeepLab输入影像消融 I0-I3

| 实验 | channel_mode | 有效通道 | AutoDL Notebook | Kaggle Notebook |
|---|---|---|---|---|
| I0 | `full` | WAC+DEM+Slope+TPI+Curvature | `notebooks/autodl_v6_overlap40_deeplab_input_full_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_full_seed42.ipynb` |
| I1 | `wac_only` | WAC | `notebooks/autodl_v6_overlap40_deeplab_input_wac_only_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_wac_only_seed42.ipynb` |
| I2 | `terrain_only` | DEM+Slope+TPI+Curvature | `notebooks/autodl_v6_overlap40_deeplab_input_terrain_only_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_terrain_only_seed42.ipynb` |
| I3 | `wac_dem` | WAC+DEM | `notebooks/autodl_v6_overlap40_deeplab_input_wac_dem_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_wac_dem_seed42.ipynb` |
| I4 | `wac_dem_slope` | WAC+DEM+Slope | `notebooks/autodl_v6_overlap40_deeplab_input_wac_dem_slope_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_wac_dem_slope_seed42.ipynb` |
| I5 | `wac_dem_slope_tpi` | WAC+DEM+Slope+TPI | `notebooks/autodl_v6_overlap40_deeplab_input_wac_dem_slope_tpi_seed42.ipynb` | `notebooks/kaggle_v6_overlap40_deeplab_input_wac_dem_slope_tpi_seed42.ipynb` |

六项共同条件：

```text
model=deeplab
encoder=resnet50
encoder_weights=imagenet
seed=42
epochs=80
batch_size=4
accum_steps=1
learning_rate=5e-5
selection_metric=val_mIoU_fg
```

必须按`config.json`中的`channel_mode`核验结果，不根据下载文件夹名称猜测。

分析时至少报告：mIoU-FG、四个前景类别IoU、Precision、Recall。解释关系：

- I0−I1：四通道地形信息在WAC基础上的总体贡献。
- I0−I2：WAC在地形信息基础上的总体贡献。
- I3−I1：DEM在WAC基础上的贡献。
- I0−I3：Slope、TPI与Profile Curvature组合的额外贡献。
- I4−I3：Slope在WAC+DEM基础上的顺序边际贡献。
- I5−I4：TPI在WAC+DEM+Slope基础上的顺序边际贡献。
- I0−I5：Profile Curvature在前四通道基础上的顺序边际贡献。

I1、I3、I4、I5、I0构成逐步累加主序列；I2必须作为无WAC对照保留，避免把顺序边际贡献误写为各通道的独立贡献。

## 4. 推荐运行顺序

```text
服务器1：F0
服务器2：F1

F0/F1运行期间或之后：I0、I1、I2、I3可在独立计算资源上并行
```

先比较F0/F1决定最终模型优化方向；输入影像消融与模块选择互不依赖，可以同时进行。

## 5. 最终模型的快速通道敏感性（可选）

训练结束后，可用同一个最终checkpoint在Val上执行归一化均值遮挡：

```powershell
python scripts/evaluate_segmentation.py `
  --model gated_rezero_cmcr `
  --data-dir "E:\月球_dataset\dataset\dataset_v6_random811_overlap40" `
  --checkpoint "D:\path\to\best_model.pth" `
  --output-dir "D:\path\to\val_drop_wac" `
  --split val `
  --channel-mode drop_wac `
  --batch-size 2 `
  --num-workers 0
```

可选值还包括`drop_dem`、`drop_slope`、`drop_tpi`、`drop_curvature`、`wac_only`和`terrain_only`。该结果只能表述为“通道遮挡敏感性”，不能代替I0-I3重新训练消融。
