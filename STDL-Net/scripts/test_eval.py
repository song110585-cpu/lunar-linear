r"""
用 best checkpoint 对独立测试集做评估，打印 mIoU 等指标。
用法: python STDL-Net/scripts/test_eval.py 'E:\月球_dataset\output\result45'
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ['datasets', 'utils', 'models']:
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MyDataset import MyDataset
from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN
import metrics

# 默认本地 v8 路径
_DEFAULT_DATA = r"E:\月球_dataset\dataset\datasetv8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='结果目录路径 (递归搜索 best_*.pth)')
    parser.add_argument('--split', choices=['test', 'val'], default='test',
                        help='评估哪个集合 (default: test)')
    parser.add_argument('--data-dir', default=None,
                        help=f'数据集根目录 (default: {_DEFAULT_DATA})')
    parser.add_argument('--val-list', default=None,
                        help='val 的 valid_list 文件 (仅 --split val 时使用)')
    args = parser.parse_args()

    data_dir = args.data_dir or _DEFAULT_DATA

    if args.split == 'val':
        IMG_DIR = os.path.join(data_dir, 'train', 'image')
        MASK_DIR = os.path.join(data_dir, 'train', 'mask')
        VALID_LIST = args.val_list or r'E:\月球_dataset\dataset\dataset_analysis\valid_tiles_val_scene.txt'
        LABEL = '验证集成绩 (Val - Mare Serenitatis 下半部)'
    else:
        IMG_DIR = os.path.join(data_dir, 'test', 'image')
        MASK_DIR = os.path.join(data_dir, 'test', 'mask')
        VALID_LIST = None
        LABEL = '独立测试集成绩 (Test Set Evaluation)'

    # 找最新的 best checkpoint
    best_pth = None
    candidates = []
    for root, dirs, files in os.walk(args.result_dir):
        for f in files:
            if f.startswith('best_'):
                fp = os.path.join(root, f)
                candidates.append((os.path.getmtime(fp), fp))
        for d in dirs:
            if d.startswith('best_'):
                dp = os.path.join(root, d)
                inner = os.path.join(dp, 'data.pkl')
                if os.path.isfile(inner):
                    candidates.append((os.path.getmtime(dp), dp))
    if not candidates:
        print('ERROR: 找不到 best_*.pth')
        return
    candidates.sort(reverse=True)
    best_pth = candidates[0][1]
    print(f'Checkpoint: {best_pth}')

    # 确定 model_size
    model_size = os.path.basename(best_pth).replace('best_', '')
    if model_size.endswith('.pth'):
        model_size = model_size.replace('.pth', '')
    if not model_size:
        model_size = 'base'
    print(f'Model size: {model_size}')

    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    load_path = best_pth
    if os.path.isdir(best_pth):
        load_path = best_pth  # torch.load handles dirs with zip format
    state = torch.load(load_path, map_location='cpu', weights_only=False)
    has_local_cnn = any(k.startswith('lc2.') for k in state.keys())
    has_strip_pooling = any(k.startswith('sp2.') for k in state.keys())

    model = Swin_LCSRB_DeformablePSP_FPNPAN(
        size=model_size, num_classes=5, in_channels=5, pretrained=False,
        use_dem_guided=False, use_strip_pooling=has_strip_pooling, use_coord_attention=False,
        use_local_cnn=has_local_cnn)
    state = {k: v for k, v in state.items() if not k.startswith(('sp2.', 'sp3.'))}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # 加载数据集
    eval_data = MyDataset(IMG_DIR, MASK_DIR, valid_list_file=VALID_LIST)
    eval_iter = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=0)
    print(f'{args.split} set: {len(eval_data)} tiles')

    # 评估
    eval_hist = torch.zeros(5, 5, dtype=torch.float64)
    with torch.no_grad():
        for img, label, name in tqdm(eval_iter, desc=f'{args.split} Eval'):
            img, label = img.to(device), label.to(device)
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits.argmax(dim=1)
            eval_hist += metrics.multiclass_confusion(pred, label, 5).double()

    tm = metrics.metrics_from_hist(eval_hist)
    CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
    print(f'\n{"="*60}')
    print(f'{LABEL}')
    print(f'Overall Accuracy: {tm["accuracy"]:.4f}')
    print(f'Mean mIoU:        {tm["miou"]:.4f}')
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
    cm = eval_hist.numpy().astype(np.int64)
    print(f'\n混淆矩阵 (行=真实标签, 列=预测标签)')
    print(f'{"":>16}', end='')
    for name in CLASS_NAMES:
        print(f'{name[:6]:>10}', end='')
    print(f'\n{"-"*66}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'{name:<16}', end='')
        for j in range(5):
            print(f'{cm[i][j]:>10,}', end='')
        print()
    # 归一化 (每行 / 行总和)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums * 100
    print(f'\n归一化混淆矩阵 (行百分比)')
    print(f'{"":>16}', end='')
    for name in CLASS_NAMES:
        print(f'{name[:6]:>10}', end='')
    print(f'\n{"-"*66}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'{name:<16}', end='')
        for j in range(5):
            print(f'{cm_norm[i][j]:>9.1f}%', end='')
        print()


if __name__ == '__main__':
    main()
