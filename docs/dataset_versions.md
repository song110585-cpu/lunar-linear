# 数据集版本记录

---

## Dataset v5

- 训练集: 3893 张
- 测试集: 304 张
- 分辨率: 512×512
- 格式: TIF (5-channel)
- 通道: WAC, DEM, Slope, TPI, Curvature

### 问题

- 背景样本占比过高
- Graben 类别样本不足
- 缺少困难样本

---

## Dataset v6

总样本数: 1983

正样本: 1531

背景样本: 452

### 类别统计

| Class | Count |
|---------|---------|
| Wrinkle Ridge | 525 |
| Rille | 302 |
| Fault | 312 |
| Graben | 393 |
| Crater Chain | 382 |

### 改动

- 增加Grab样本
- 过滤背景占比过高样本
- 增加困难样本

### 归一化参数

```
CHANNEL_MEAN = [0.1475, 0.5435, 0.2545, 0.4909, 0.4547]
CHANNEL_STD  = [0.0911, 0.3890, 0.2680, 0.2118, 0.2185]
```

### 数据划分

- Train split: ~90% (按 `valid_tiles_train_split.txt`)
- Val: ~10% (按 `valid_tiles_val.txt`)
- Test: 独立区域 (按 `valid_tiles_test.txt`)
- 划分方式: 按主导类别分层随机采样
