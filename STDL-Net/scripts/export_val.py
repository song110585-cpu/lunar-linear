"""
用 best_small.pth 对验证集做推理，生成 pred_mask PNG。
用法:python STDL-Net/scripts/export_val.py "E:\月球_dataset\output\result47"
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ['datasets', 'utils', 'models']:
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from MyDataset import MyDataset
from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN

VAL_IMG_DIR = r"E:\月球_dataset\Research area\train\dataset_v9\image"
VAL_MASK_DIR = r"E:\月球_dataset\Research area\train\dataset_v9\mask"
VAL_VALID_LIST = r"E:\月球_dataset\dataset\datasetv9\pretrain\train_list.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='结果目录路径')
    args = parser.parse_args()

    # 找 best_*.pth 和 history.json — 遍历所有子目录, 选最新的
    best_pth = None
    history_path = None
    model_size = None
    candidates = []
    for root, dirs, files in os.walk(args.result_dir):
        # 常规 .pth 文件
        pth_files = sorted([f for f in files if f.startswith('best_')])
        for pf in pth_files:
            fp = os.path.join(root, pf)
            candidates.append((os.path.getmtime(fp), fp, root))
        # Kaggle 下载时自动解压的目录 (best_base/ 内含 data.pkl)
        for d in dirs:
            if d.startswith('best_'):
                dp = os.path.join(root, d)
                inner = os.path.join(dp, 'data.pkl')
                if os.path.isfile(inner):
                    candidates.append((os.path.getmtime(dp), dp, root))
    if candidates:
        candidates.sort(reverse=True)  # 按修改时间倒序, 最新的排第一
        _, best_pth, root = candidates[0]
        model_size = os.path.basename(best_pth).replace('best_', '').replace('.pth', '')
        all_files = os.listdir(root)
        json_candidates = [f for f in all_files if f.replace(' ', '') in ('history.json', 'history.txt', 'history')]
        if not json_candidates:
            # 上级目录也找一下
            parent = os.path.dirname(root)
            if os.path.isdir(parent):
                json_candidates = [f for f in os.listdir(parent) if f.replace(' ', '') in ('history.json', 'history.txt', 'history')]
                if json_candidates:
                    history_path = os.path.join(parent, json_candidates[0])
                    root = parent
        if not json_candidates:
            history_path = None
        else:
            history_path = os.path.join(root, json_candidates[0])
        print(f'  Found: {os.path.basename(best_pth)} (model_size={model_size}) '
              f'[{len(candidates)} total, using latest]')

    if not best_pth:
        print(f'ERROR: 找不到 best_*.pth')
        return

    if not history_path:
        print(f'ERROR: 找不到 history.json')
        return

    # 读取 best epoch
    import json
    with open(history_path) as f:
        h = json.load(f)
    best_idx = int(np.argmax(h['val_miou'] if 'val_miou' in h else h['test_miou']))
    best_epoch = h['epoch'][best_idx]
    best_miou = float(h['val_miou'][best_idx] if 'val_miou' in h else h['test_miou'][best_idx])

    output_dir = os.path.join(os.path.dirname(best_pth),
                              f'best_epoch_{best_epoch:02d}_val_miou_{best_miou:.4f}',
                              'pred_mask')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Best epoch: {best_epoch}, val_mIoU: {best_miou:.4f}')
    print(f'Output: {output_dir}')

    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Auto-detect: 从 state_dict 键判断模块配置
    # 兼容 Kaggle 自动解压: 目录格式直接传目录路径给 torch.load
    load_path = best_pth
    state = torch.load(load_path, map_location='cpu', weights_only=False)
    has_local_cnn = any(k.startswith('lc2.') for k in state.keys())
    has_strip_pooling = any(k.startswith('sp2.') for k in state.keys())
    print(f'  Detected: LocalCNN={has_local_cnn}, StripPooling={has_strip_pooling}')

    # 从 checkpoint 推断 img_size
    img_size = 256
    for k in state.keys():
        if 'layers.0' in k and 'attn_mask' in k:
            nW = state[k].shape[0]
            img_size = int(nW ** 0.5) * 16 * 4  # nW → resolution → img_size
            break
    print(f'  Inferred img_size={img_size}')

    model = Swin_LCSRB_DeformablePSP_FPNPAN(
        img_size=img_size, size=model_size, num_classes=5, in_channels=5,
        pretrained=False, use_dem_guided=False,
        use_strip_pooling=has_strip_pooling, use_coord_attention=False,
        use_local_cnn=has_local_cnn)
    state = {k: v for k, v in state.items() if not k.startswith(('sp2.', 'sp3.'))}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    print('Model loaded.')

    # 数据集
    val_data = MyDataset(VAL_IMG_DIR, VAL_MASK_DIR, valid_list_file=VAL_VALID_LIST)
    val_iter = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    print(f'Validation set: {len(val_data)} tiles')

    # 推理
    count = 0
    with torch.no_grad():
        for img, label, name in tqdm(val_iter, desc='Exporting val preds'):
            img = img.to(device)
            pred = model(img).argmax(dim=1)
            pred_np = pred[0].cpu().numpy().astype(np.uint8)
            stem = name[0]
            Image.fromarray(pred_np, mode='L').save(os.path.join(output_dir, f'{stem}.png'))
            count += 1

    print(f'Done! {count} pred_masks saved to {output_dir}')


if __name__ == '__main__':
    main()
