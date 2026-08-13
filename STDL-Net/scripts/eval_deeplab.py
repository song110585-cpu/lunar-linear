"""
用 DeepLabV3+ checkpoint 对随机8:1:1 test 集评估 (5分类).
用法: python STDL-Net/scripts/eval_deeplab.py "E:\月球_dataset\output\deeplab\_result_20260812_102552.pth"
"""
import os, sys, argparse
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['utils', 'models', 'datasets']:
    sys.path.insert(0, os.path.join(_root, _sub))

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from MyDataset import MyDataset
import metrics

CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help='DeepLabV3+ .pth checkpoint 路径')
    parser.add_argument('--data-dir', default=r'E:\月球_dataset\dataset\datasetv5_random811')
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模型
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50', encoder_weights=None, in_channels=5, classes=5).to(device)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f'Model loaded: {args.ckpt}')

    # 数据
    split_dir = os.path.join(args.data_dir, args.split)
    data = MyDataset(os.path.join(split_dir, 'image'), os.path.join(split_dir, 'mask'))
    data_iter = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    print(f'{args.split} set: {len(data)} tiles')

    # 评估
    hist = torch.zeros(5, 5, dtype=torch.float64)
    with torch.no_grad():
        for img, label, _ in tqdm(data_iter, desc=f'{args.split} Eval'):
            img, label = img.to(device), label.to(device)
            pred = model(img).argmax(dim=1)
            hist += metrics.multiclass_confusion(pred, label, 5).double()

    tm = metrics.metrics_from_hist(hist)
    print(f'\n{"="*60}')
    print(f'{args.split.upper()} Set Results')
    print(f'Accuracy: {tm["accuracy"]:.4f}  mIoU: {tm["miou"]:.4f}')
    print(f'mF1: {tm["mf1"]:.4f}  mPrec: {tm["mprecision"]:.4f}  mRec: {tm["mrecall"]:.4f}')
    print(f'{"="*60}')
    print(f'{"Class":<18} {"IoU":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}')
    print(f'{"-"*50}')
    for c in range(5):
        print(f'{CLASS_NAMES[c]:<18} {tm["iou_per_class"][c]:>8.4f} '
              f'{tm["precision_per_class"][c]:>8.4f} '
              f'{tm["recall_per_class"][c]:>8.4f} '
              f'{tm["f1_per_class"][c]:>8.4f}')
    print(f'{"-"*50}')

    # 混淆矩阵 (行=GT, 列=Pred)
    cm = hist.numpy().astype(np.int64)
    print(f'\n混淆矩阵 (行=真实, 列=预测)')
    print(f'{"":>16}', end='')
    for n in CLASS_NAMES:
        print(f'{n[:6]:>10}', end='')
    print(f'\n{"-"*66}')
    for i, n in enumerate(CLASS_NAMES):
        print(f'{n:<16}', end='')
        for j in range(5):
            print(f'{cm[i][j]:>10,}', end='')
        print()


if __name__ == '__main__':
    main()
