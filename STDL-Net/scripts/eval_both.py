"""
同时评估 Swin-UNet 和 DeepLabV3+ 的二分类 Test 结果
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models']:
    sys.path.insert(0, os.path.join(_root, _sub))

from MyDataset import MyDataset
from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN
import metrics as _m


def eval_model(model, data_iter, device, name):
    hist = torch.zeros(2, 2, dtype=torch.float64)
    with torch.no_grad():
        for img, label, _ in tqdm(data_iter, desc=name):
            img = img.to(device)
            label = (label > 0).long().to(device)
            pred = model(img).argmax(dim=1)
            hist += _m.multiclass_confusion(pred, label, 2).double()

    m = _m.metrics_from_hist(hist)
    iou = m['iou_per_class']
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  BG IoU: {iou[0]:.4f}  Line IoU: {iou[1]:.4f}  mIoU: {m['miou']:.4f}")
    prec = m.get('precision_per_class', [0,0])
    rec  = m.get('recall_per_class', [0,0])
    if prec:
        print(f"  Line Precision: {prec[1]:.4f}  Recall: {rec[1]:.4f}")
    print(f"{'='*50}")
    return iou[1], m['miou']


def main():
    data_root = r'E:\月球_dataset\dataset\datasetv5'
    test_img = os.path.join(data_root, 'test', 'image')
    test_msk = os.path.join(data_root, 'test', 'mask')
    test_data = MyDataset(test_img, test_msk)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test tiles: {len(test_data)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Swin-UNet ----
    print("\nLoading Swin-UNet...")
    model_swin = Swin_LCSRB_DeformablePSP_FPNPAN(
        size='small', img_size=512, num_classes=2, in_channels=5, pretrained=False).to(device)
    ckpt_swin = r'E:\月球_dataset\binary_experiments\swin_unet\best_small.pth'
    state = torch.load(ckpt_swin, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith(('sp2.', 'sp3.'))}
    model_swin.load_state_dict(state, strict=False)
    model_swin.eval()
    eval_model(model_swin, test_iter, device, "Swin-UNet (R_binary)")

    # ---- DeepLabV3+ ----
    print("\nLoading DeepLabV3+...")
    model_dlv3 = smp.DeepLabV3Plus(
        encoder_name='resnet50', encoder_weights=None, in_channels=5, classes=2).to(device)
    ckpt_dlv3 = r'E:\月球_dataset\binary_experiments\deeplabv3p\best_resnet50.pth'
    model_dlv3.load_state_dict(torch.load(ckpt_dlv3, map_location=device))
    model_dlv3.eval()
    eval_model(model_dlv3, test_iter, device, "DeepLabV3+ (ResNet50)")

    print("\nDone.")


if __name__ == '__main__':
    main()
