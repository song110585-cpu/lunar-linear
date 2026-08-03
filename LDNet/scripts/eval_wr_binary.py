"""
用现有 5 分类模型评估 WR 二分类的物理上限
直接用 backbone.py 里的模型类
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models']:
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MyDataset import MyDataset
from backbone import Swin_LCSRB_DeformablePSP_FPNPAN


def main():
    import sys as _sys
    ckpt_path = _sys.argv[1] if len(_sys.argv) > 1 else r"E:\月球_dataset\output\result10\result\result\best_small.pth"
    data_root = r"E:\月球_dataset\dataset\datasetv5"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # 直接用 backbone.py 里的模型类 (和 checkpoint 匹配)
    model = Swin_LCSRB_DeformablePSP_FPNPAN(
        size='small', img_size=512, num_classes=5, in_channels=5, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k: v for k, v in state.items() if not k.startswith(('sp2.', 'sp3.'))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if len(missing) == 0:
        print("All keys matched! Checkpoint fully loaded.")
    model.eval()

    test_img = os.path.join(data_root, 'test', 'image')
    test_msk = os.path.join(data_root, 'test', 'mask')
    test_data = MyDataset(test_img, test_msk)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test tiles: {len(test_data)}")

    # 评估
    hist_5x5 = torch.zeros(5, 5, dtype=torch.float64)
    with torch.no_grad():
        for img, label, _ in tqdm(test_iter, desc='Eval'):
            img, label = img.to(device), label.to(device)
            pred = model(img).argmax(dim=1)
            for c in range(5):
                mask = (label == c)
                for pc in range(5):
                    hist_5x5[c, pc] += (pred[mask] == pc).sum().item()

    print(f"\nConfusion matrix (row=GT, col=Pred):")
    names = ['Bg', 'WR', 'Rille', 'Fault', 'Graben']
    print(f"{'':>8} {'Bg':>8} {'WR':>8} {'Rille':>8} {'Fault':>8} {'Graben':>8}")
    for i in range(5):
        row_str = ' '.join([f'{hist_5x5[i,j].item():>8.0f}' for j in range(5)])
        print(f"{names[i]:>8} {row_str}")

    # WR 二分类指标
    tp = hist_5x5[1, 1].item()
    fp = hist_5x5[0, 1].item() + hist_5x5[2, 1].item() + hist_5x5[3, 1].item() + hist_5x5[4, 1].item()
    fn = hist_5x5[1, 0].item() + hist_5x5[1, 2].item() + hist_5x5[1, 3].item() + hist_5x5[1, 4].item()
    eps = 1e-10
    print(f"\n{'='*50}")
    print(f"  WR Binary (class 1 vs rest)")
    print(f"  TP={tp:.0f}  FP={fp:.0f}  FN={fn:.0f}")
    print(f"  IoU:       {tp/(tp+fp+fn+eps):.4f}")
    print(f"  Recall:    {tp/(tp+fn+eps):.4f}")
    print(f"  Precision: {tp/(tp+fp+eps):.4f}")
    print(f"  F1:        {2*tp/(2*tp+fp+fn+eps):.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
