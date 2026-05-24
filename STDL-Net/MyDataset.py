"""
5 通道月球影像 + 多类别标签 Dataset
通道顺序: [WAC, DEM, Slope, TPI, 剖面曲率]
类别约定: 0=背景, 1=皱脊, 2=月溪, 3=断层, 4=地堑
"""
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# 归一化参数 (dataset_v3 训练集统计, 数据已预先归一化到 [0, 1])
CHANNEL_MEAN = [0.1758, 0.5991, 0.2652, 0.4936, 0.4552]
CHANNEL_STD  = [0.0784, 0.3981, 0.2708, 0.2165, 0.2238]

class MyDataset(Dataset):
    """
    images_dir 和 masks_dir 下同名 .tif 一一对应.
    返回 (image_tensor, mask_tensor, img_stem):
        image_tensor: (C, H, W) float32
        mask_tensor:  (H, W)    int64
        img_stem:     文件名(不含扩展名)
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.mean = np.array(mean if mean is not None else CHANNEL_MEAN, dtype=np.float32)
        self.std = np.array(std if std is not None else CHANNEL_STD, dtype=np.float32)

        # image 和 mask 同名, 以 mask 目录为基准
        self.filenames = sorted(
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        )

        # 简单校验
        img_set = set(os.listdir(images_dir))
        missing = [f for f in self.filenames if f not in img_set]
        if missing:
            raise FileNotFoundError(
                f"mask 目录下有 {len(missing)} 个文件在 image 目录中找不到, "
                f"前 5 个: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]
        stem = os.path.splitext(fname)[0]

        # ---- 读 5 通道影像 ----
        with rasterio.open(os.path.join(self.images_dir, fname)) as src:
            image = src.read().astype(np.float32)  # (C, H, W)

        # 将 nodata (-3.4e38 等极端值) 和 NaN/Inf 替换为 0
        bad = ~np.isfinite(image) | (image < -1e10)
        image[bad] = 0.0

        # 按通道归一化 (mean=0, std=1 时等于不做归一化)
        image = (image - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        # ---- 读标签 ----
        with rasterio.open(os.path.join(self.masks_dir, fname)) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # 双保险: 把所有不在 0-4 范围的值都合并到 0（处理 99, 9 等异常值）
        mask[(mask > 4) | (mask < 0)] = 0
        # 5 类: 0=背景, 1=皱脊, 2=月溪, 3=断层, 4=地堑

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor, stem
if __name__ == "__main__":
    import random

    IMG_DIR = r"E:\月球_dataset\Research area\train\dataset_v3\image"
    MASK_DIR = r"E:\月球_dataset\Research area\train\dataset_v3\mask"

    print("正在初始化数据集...")
    dataset = MyDataset(IMG_DIR, MASK_DIR)
    print(f"数据集加载成功，共有 {len(dataset)} 张切片")
