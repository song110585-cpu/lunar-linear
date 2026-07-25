"""
用 best checkpoint 对 test 集生成 pred_mask PNG + 评估指标.
用法: python STDL-Net/scripts/export_test.py "E:\月球_dataset\output\result48"
"""
import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models']:
    sys.path.insert(0, os.path.join(_root, _sub))

import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from MyDataset import MyDataset
from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN
import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='结果目录')
    parser.add_argument('--data-dir', default=r'E:\月球_dataset\dataset\datasetv5')
    parser.add_argument('--out', default=None, help='输出目录 (默认: result_dir/test_preds/)')
    args = parser.parse_args()

    # 找 best checkpoint
    best_pth = None
    history_path = None
    candidates = []
    for root, dirs, files in os.walk(args.result_dir):
        for f in files:
            if f.startswith('best_'):
                candidates.append((os.path.getmtime(os.path.join(root, f)), os.path.join(root, f)))
    if not candidates:
        print('ERROR: 找不到 best_*.pth')
        return
    candidates.sort(reverse=True)
    best_pth = candidates[0][1]
    model_size = os.path.basename(best_pth).replace('best_', '').replace('.pth', '')
    print(f'Checkpoint: {best_pth}  (model={model_size})')

    # 找 history.json
    for root, dirs, files in os.walk(args.result_dir):
        for f in files:
            if f == 'history.json':
                history_path = os.path.join(root, f)
                break
    if history_path:
        with open(history_path) as f:
            h = json.load(f)
        best_idx = int(np.argmax(h['val_miou']))
        best_epoch = h['epoch'][best_idx]
        best_val = h['val_miou'][best_idx]
        print(f'Best: epoch={best_epoch}, val_mIoU={best_val:.4f}')

    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(best_pth, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    img_size = 512
    for k in state.keys():
        if 'layers.0' in k and 'attn_mask' in k:
            img_size = int(state[k].shape[0] ** 0.5) * 16 * 4
            break
    print(f'img_size={img_size}')

    has_sp = any(k.startswith('sp2.') for k in state.keys())
    has_lc = any(k.startswith('lc2.') for k in state.keys())
    model = Swin_LCSRB_DeformablePSP_FPNPAN(
        img_size=img_size, size=model_size, num_classes=5, in_channels=5,
        pretrained=False, use_dem_guided=False,
        use_strip_pooling=has_sp, use_coord_attention=False, use_local_cnn=has_lc)
    state = {k: v for k, v in state.items() if not k.startswith(('sp2.', 'sp3.'))}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # 加载 test 数据
    test_dir = os.path.join(args.data_dir, 'test')
    test_data = MyDataset(os.path.join(test_dir, 'image'), os.path.join(test_dir, 'mask'))
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    print(f'Test set: {len(test_data)} tiles')

    # 输出目录
    out_dir = args.out or os.path.join(args.result_dir, 'test_preds')
    mask_dir = os.path.join(out_dir, 'pred_mask')
    os.makedirs(mask_dir, exist_ok=True)

    # 推理 + 评估
    eval_hist = torch.zeros(5, 5, dtype=torch.float64)
    with torch.no_grad():
        for img, label, name in tqdm(test_iter, desc='Test'):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            logits = pred[0] if isinstance(pred, tuple) else pred
            pred = logits.argmax(dim=1)
            eval_hist += metrics.multiclass_confusion(pred, label, 5).double()
            # 保存 pred mask
            png = pred[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(png, mode='L').save(os.path.join(mask_dir, f'{name[0]}.png'))

    tm = metrics.metrics_from_hist(eval_hist)
    CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
    print(f'\n{"="*60}')
    print(f'Test Set Results')
    print(f'Accuracy: {tm["accuracy"]:.4f}  mIoU: {tm["miou"]:.4f}')
    print(f'{"="*60}')
    print(f'{"Class":<18} {"IoU":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}')
    for c in range(5):
        print(f'{CLASS_NAMES[c]:<18} {tm["iou_per_class"][c]:>8.4f} '
              f'{tm["precision_per_class"][c]:>8.4f} '
              f'{tm["recall_per_class"][c]:>8.4f} '
              f'{tm["f1_per_class"][c]:>8.4f}')
    print(f'\nSaved {len(test_data)} pred_masks → {mask_dir}')


if __name__ == '__main__':
    main()
