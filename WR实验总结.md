# WR（皱脊）二分类实验 — 新方向总结

> 最后更新：2026-08-08

---

## 1. 数据已就绪（2026-08-08）

929 条 WR 面要素，5 区域，2019 tiles，两种划分策略：

| 区域 | Features | Tiles | 占比 |
|------|----------|-------|------|
| Mare Imbrium | 212 | 420 | 21% |
| Mare Serenitatis | 179 | 270 | 13% |
| Mare Tranquillitatis | 140 | 132 | 7% |
| Marius Hills | 137 | 240 | 12% |
| Oceanus Procellarum | 261 | 957 | **47%** |

| | Mixed | Holdout |
|------|-------|---------|
| **Train** | 5区混合 1615 | 东部4区 1062 |
| **Val** | 5区混合 201 | 风暴洋分层 478 |
| **Test** | 5区混合 203 | 风暴洋分层 479 |
| **Val/Test WR 平衡** | 随机 | 各 74 含 WR，密度 ~0.93% |
| **目的** | 跟文献 0.71 对标 | 测跨区域泛化 |

## 4. 模型与超参

- Swin-UNet, 5ch, BCE+Dice+pos_weight=5.0, lr=5e-5, batch=2, accum=2
- 关闭所有结构模块，纯净基线
- DeepLabV3+ 作为对比

## 5. 下一步

1. 上传 Kaggle 训练 Mixed 和 Holdout
2. 对比 Val IoU 差距 → 量化空间泄漏
3. 全月推理
