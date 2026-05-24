import os, numpy as np
import rasterio
from collections import Counter

DIRS = {
    'train_mask': r"E:\月球_dataset\baseline模型结果\lunar-dataset\dataset\train\mask",
    'test_mask':  r"E:\月球_dataset\baseline模型结果\lunar-dataset\dataset\test\mask",
}

for name, d in DIRS.items():
    files = [f for f in os.listdir(d) if f.lower().endswith('.tif')]
    total = Counter()
    n_with_graben = 0
    for f in files:
        with rasterio.open(os.path.join(d, f)) as src:
            m = src.read(1)
        u, c = np.unique(m, return_counts=True)
        for k, v in zip(u, c):
            total[int(k)] += int(v)
        if 4 in u:
            n_with_graben += 1
    print(f'\n=== {name} ({len(files)} files) ===')
    for k in sorted(total):
        print(f'  class {k}: {total[k]:>14,} pixels')
    print(f'  含 Graben(4) 的样本: {n_with_graben}/{len(files)}')