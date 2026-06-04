"""
本地可视化脚本 - 直接读取 Kaggle 下载的 result 文件夹生成分析图表

使用方法:
    1. 修改下面 RESULT_DIR 为你的 result 文件夹路径
    2. 直接运行本脚本: python visualize_result.py
    3. 输出图表保存在 RESULT_DIR/visualization/ 下
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

#   只需要改这里！把路径改成你下载的 result 文件夹

RESULT_DIR = r"E:\月球_dataset\baseline模型结果\result26"  # R13结果

# 测试集 GT mask 目录 (一般不用改, 自动搜索)
TEST_MASK_DIR = r"E:\月球_dataset\Research area\test\dataset_v6\mask"

# 测试集影像目录 (可选, 用于显示 WAC/DEM/Slope)
TEST_IMG_DIR = r"E:\月球_dataset\Research area\test\dataset_v6\image"

# ========== TTA 配置 ==========
TTA_ENABLED = True                # 设为 True 启用 TTA 推理
MODEL_PATH = r"E:\月球_dataset\baseline模型结果\result26\result\result\best_small.pth"   # best_small.pth 路径
MODEL_SIZE = 'small'
USE_STRIP_POOLING = False         # R19 模型=True; R17/R18 baseline 模型=False
TTA_OUTPUT_DIR = ''                # 留空则自动设为 RESULT_DIR/tta_pred_mask

# ========== 配置 ==========
NUM_CLASSES = 5
CLASS_NAMES = ['Background', 'Wrinkle Ridge', 'Rille', 'Fault', 'Graben']
CLASS_NAMES_CN = ['背景', '皱脊', '月溪', '断层', '地堑']
CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 100, 255],
    [0, 200, 0],
    [255, 255, 0],
], dtype=np.uint8)
CLASS_COLORS_PLT = ['black', 'red', 'dodgerblue', 'green', 'orange']


def mask_to_color(mask):
    return CLASS_COLORS[np.clip(mask.astype(np.int64), 0, NUM_CLASSES - 1)]


def error_map(gt, pred):
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    gt_fg, pr_fg = gt > 0, pred > 0
    out[gt_fg & pr_fg] = [0, 200, 0]       # TP
    out[gt_fg & (~pr_fg)] = [255, 0, 0]    # FN
    out[(~gt_fg) & pr_fg] = [255, 165, 0]  # FP
    return out


def get_legend_patches(iou_per_class):
    patches = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        patches.append(mpatches.Patch(
            color=color / 255.0,
            label=f'{name} (IoU={iou_per_class[i]:.3f})'
        ))
    return patches


def compute_metrics(pred_dir, gt_dir):
    """计算混淆矩阵和各类 IoU"""
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    per_image = []
    names = []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f'找到 {len(pred_files)} 张预测图')

    for fname in pred_files:
        pred = np.array(Image.open(os.path.join(pred_dir, fname)))
        gt_path = os.path.join(gt_dir, fname.replace('.png', '.tif'))
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

        # 尝试读取 tif
        try:
            import rasterio
            with rasterio.open(gt_path) as src:
                gt = src.read(1).astype(np.int64)
        except:
            gt = np.array(Image.open(gt_path)).astype(np.int64)

        gt[(gt > 4) | (gt < 0)] = 0

        # 累计混淆矩阵
        valid = (gt >= 0) & (gt < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
        hist += np.bincount(
            gt[valid] * NUM_CLASSES + pred[valid],
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        # 单图 mIoU
        h = np.bincount(
            gt[valid] * NUM_CLASSES + pred[valid],
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        iou = np.diag(h) / (h.sum(0) + h.sum(1) - np.diag(h) + 1e-10)
        per_image.append(float(np.mean(iou)))
        names.append(fname.replace('.png', ''))

    # 全局指标
    iou_per_class = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist) + 1e-10)
    precision = np.diag(hist) / (hist.sum(0) + 1e-10)
    recall = np.diag(hist) / (hist.sum(1) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = np.diag(hist).sum() / (hist.sum() + 1e-10)
    miou = float(np.mean(iou_per_class))

    return {
        'hist': hist,
        'iou_per_class': iou_per_class,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'miou': miou,
        'per_image_miou': np.array(per_image),
        'names': names,
    }


def plot_training_curves(history, save_dir):
    ep = history['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(ep, history['train_loss'], 'o-', ms=3, label='Train')
    axes[0, 0].plot(ep, history['test_loss'], 's-', ms=3, label='Test')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history['train_miou'], 'o-', ms=3, label='Train')
    axes[0, 1].plot(ep, history['test_miou'], 's-', ms=3, label='Test')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('mIoU Curve'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if 'train_iou_per_class' in history:
        arr = np.array(history['train_iou_per_class'])
        for c in range(min(NUM_CLASSES, arr.shape[1])):
            axes[1, 0].plot(ep, arr[:, c], 'o-', ms=2, lw=1.5,
                            color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Train Per-Class IoU'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    if 'test_iou_per_class' in history:
        arr = np.array(history['test_iou_per_class'])
        for c in range(min(NUM_CLASSES, arr.shape[1])):
            axes[1, 1].plot(ep, arr[:, c], 's-', ms=2, lw=1.5,
                            color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_title('Test Per-Class IoU'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f'  保存: training_curves.png')


def plot_confusion_matrix(hist, save_dir):
    row_sums = hist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    hist_norm = hist / row_sums * 100

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(hist_norm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (%)')

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = hist_norm[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'  保存: confusion_matrix.png')


def plot_iou_bar(metrics, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#333333', '#FF0000', '#0064FF', '#00C800', '#FFFF00']
    bars = ax.bar(CLASS_NAMES, metrics['iou_per_class'], color=colors, edgecolor='black')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (mIoU={metrics["miou"]:.4f})')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=metrics['miou'], color='gray', linestyle='--', label=f'mIoU={metrics["miou"]:.4f}')
    for bar, val in zip(bars, metrics['iou_per_class']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_per_class.png'), dpi=150)
    plt.close()
    print(f'  保存: iou_per_class.png')


def plot_miou_distribution(metrics, save_dir):
    per_image_miou = metrics['per_image_miou']
    names = metrics['names']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(per_image_miou, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(per_image_miou), color='red', linestyle='--',
                    label=f'mean={np.mean(per_image_miou):.4f}')
    axes[0].set_xlabel('mIoU')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-image mIoU Distribution')
    axes[0].legend()

    worst_10_idx = np.argsort(per_image_miou)[:10]
    axes[1].barh(range(10), per_image_miou[worst_10_idx], color='salmon', edgecolor='black')
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([names[i] for i in worst_10_idx], fontsize=8)
    axes[1].set_xlabel('mIoU')
    axes[1].set_title('Worst 10 images')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'miou_distribution.png'), dpi=150)
    plt.close()
    print(f'  保存: miou_distribution.png')


def plot_worst_predictions(metrics, pred_dir, gt_dir, test_img_dir, save_dir, n=4):
    """可视化最差的 n 张"""
    per_image_miou = metrics['per_image_miou']
    names = metrics['names']
    worst_idx = np.argsort(per_image_miou)[:n]

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    for row, idx in enumerate(worst_idx):
        stem = names[idx]
        pred = np.array(Image.open(os.path.join(pred_dir, f'{stem}.png')))

        # 读 GT
        gt_path = os.path.join(gt_dir, f'{stem}.tif')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f'{stem}.png')
        try:
            import rasterio
            with rasterio.open(gt_path) as src:
                gt = src.read(1).astype(np.int64)
        except:
            gt = np.array(Image.open(gt_path)).astype(np.int64)
        gt[(gt > 4) | (gt < 0)] = 0

        # 读 WAC (5通道的第一个)
        wac = None
        if test_img_dir:
            img_path = os.path.join(test_img_dir, f'{stem}.tif')
            if os.path.exists(img_path):
                try:
                    import rasterio
                    with rasterio.open(img_path) as src:
                        wac = src.read(1).astype(np.float32)
                    wac = (wac - wac.min()) / (wac.max() - wac.min() + 1e-8)
                except:
                    pass

        if wac is not None:
            axes[row, 0].imshow(wac, cmap='gray')
        else:
            axes[row, 0].imshow(np.zeros_like(pred), cmap='gray')
        axes[row, 0].set_title(f'WAC - {stem}\nmIoU={per_image_miou[idx]:.4f}', fontsize=8)
        axes[row, 1].imshow(mask_to_color(gt.astype(np.uint8)))
        axes[row, 1].set_title('Ground Truth')
        axes[row, 2].imshow(mask_to_color(pred))
        axes[row, 2].set_title('Prediction')
        axes[row, 3].imshow(error_map(gt.astype(np.uint8), pred))
        axes[row, 3].set_title('Error')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Worst {n} Predictions', fontsize=16)
    fig.legend(handles=get_legend_patches(metrics['iou_per_class']),
               loc='lower center', ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'worst_predictions.png'), dpi=150)
    plt.close()
    print(f'  保存: worst_predictions.png')


def plot_top_foreground(metrics, pred_dir, gt_dir, test_img_dir, save_dir, n=4):
    """可视化前景最多的 n 张"""
    names = metrics['names']

    # 计算每张图的前景像素数
    fg_counts = []
    for stem in names:
        gt_path = os.path.join(gt_dir, f'{stem}.tif')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f'{stem}.png')
        try:
            import rasterio
            with rasterio.open(gt_path) as src:
                gt = src.read(1)
        except:
            gt = np.array(Image.open(gt_path))
        fg_counts.append(np.sum(gt > 0))

    top_idx = np.argsort(fg_counts)[::-1][:n]

    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    for row, idx in enumerate(top_idx):
        stem = names[idx]
        pred = np.array(Image.open(os.path.join(pred_dir, f'{stem}.png')))

        gt_path = os.path.join(gt_dir, f'{stem}.tif')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f'{stem}.png')
        try:
            import rasterio
            with rasterio.open(gt_path) as src:
                gt = src.read(1).astype(np.int64)
        except:
            gt = np.array(Image.open(gt_path)).astype(np.int64)
        gt[(gt > 4) | (gt < 0)] = 0

        wac, dem, slope = None, None, None
        if test_img_dir:
            img_path = os.path.join(test_img_dir, f'{stem}.tif')
            if os.path.exists(img_path):
                try:
                    import rasterio
                    with rasterio.open(img_path) as src:
                        bands = src.read()  # (C, H, W)
                    wac = bands[0].astype(np.float32)
                    wac = (wac - wac.min()) / (wac.max() - wac.min() + 1e-8)
                    if bands.shape[0] > 1:
                        dem = bands[1].astype(np.float32)
                        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
                    if bands.shape[0] > 2:
                        slope = bands[2].astype(np.float32)
                        slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
                except:
                    pass

        if wac is not None:
            axes[row, 0].imshow(wac, cmap='gray')
        axes[row, 0].set_title(f'WAC - {stem}', fontsize=7)
        axes[row, 1].imshow(mask_to_color(gt.astype(np.uint8)))
        axes[row, 1].set_title('Ground Truth')
        axes[row, 2].imshow(mask_to_color(pred))
        axes[row, 2].set_title('Prediction')
        axes[row, 3].imshow(error_map(gt.astype(np.uint8), pred))
        axes[row, 3].set_title('Error (R=miss, O=false alarm)')
        if dem is not None:
            axes[row, 4].imshow(dem, cmap='terrain')
        axes[row, 4].set_title('DEM')
        if slope is not None:
            axes[row, 5].imshow(slope, cmap='hot')
        axes[row, 5].set_title('Slope')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Test Predictions (top foreground samples)', fontsize=16)
    fig.legend(handles=get_legend_patches(metrics['iou_per_class']),
               loc='lower center', ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'predictions_top_fg.png'), dpi=150)
    plt.close()
    print(f'  保存: predictions_top_fg.png')


def tta_inference(model_path, model_size, test_img_dir, output_dir):
    """
    TTA 推理: 原图 + 水平翻转 + 垂直翻转 + 180°旋转, 取 softmax 平均
    预计提升 1~3% mIoU
    """
    import torch
    import sys
    # 确保能 import 模型代码
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    for p in [code_dir, project_dir,
              os.path.join(project_dir, 'models'),
              os.path.join(project_dir, 'datasets')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from swinv2unet import Swin_LCSRB_DeformablePSP_FPNPAN
    from MyDataset import MyDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'  TTA device: {device}')

    # 创建模型 (跳过预训练权重加载, 因为本地可能没有预训练文件)
    _orig_torch_load = torch.load
    def _dummy_load(*args, **kwargs):
        return {}
    torch.load = _dummy_load  # 临时替换, 跳过 swinv2unet 内部的预训练加载
    try:
        model = Swin_LCSRB_DeformablePSP_FPNPAN(
            size=model_size, num_classes=NUM_CLASSES, in_channels=5, pretrained=False,
            use_strip_pooling=USE_STRIP_POOLING,
        )
    finally:
        torch.load = _orig_torch_load  # 恢复

    # 加载真正的微调权重
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f'  模型加载完成: {model_path}')

    # 数据集
    test_data = MyDataset(images_dir=test_img_dir, masks_dir=TEST_MASK_DIR)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for img, label, name in test_iter:
            img = img.to(device)
            # 4-fold TTA
            logits_list = []

            # 1) 原图
            with torch.amp.autocast('cuda'):
                logits_list.append(torch.softmax(model(img), dim=1))

            # 2) 水平翻转
            img_hflip = img.flip(-1)
            with torch.amp.autocast('cuda'):
                out = torch.softmax(model(img_hflip), dim=1)
            logits_list.append(out.flip(-1))

            # 3) 垂直翻转
            img_vflip = img.flip(-2)
            with torch.amp.autocast('cuda'):
                out = torch.softmax(model(img_vflip), dim=1)
            logits_list.append(out.flip(-2))

            # 4) 180° 旋转
            img_rot180 = img.flip(-1).flip(-2)
            with torch.amp.autocast('cuda'):
                out = torch.softmax(model(img_rot180), dim=1)
            logits_list.append(out.flip(-2).flip(-1))

            # 平均概率
            avg_probs = torch.stack(logits_list, dim=0).mean(dim=0)
            pred = avg_probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            # 保存
            Image.fromarray(pred, mode='L').save(
                os.path.join(output_dir, f'{name[0]}.png'))

            count += 1
            if count % 50 == 0:
                print(f'  TTA 进度: {count}/{len(test_data)}')

    print(f'  TTA 完成! 共 {count} 张, 保存到: {output_dir}')
    return output_dir


def main():
    result_dir = RESULT_DIR
    gt_dir = TEST_MASK_DIR
    test_img_dir = TEST_IMG_DIR

    print(f'结果目录: {result_dir}')
    save_dir = os.path.join(result_dir, 'visualization')
    os.makedirs(save_dir, exist_ok=True)

    # TTA 推理 (可选)
    tta_pred_dir = None
    if TTA_ENABLED and MODEL_PATH and os.path.exists(MODEL_PATH):
        print('\n[TTA] 启用 Test-Time Augmentation (4-fold: orig + hflip + vflip + rot180)')
        tta_output = TTA_OUTPUT_DIR if TTA_OUTPUT_DIR else os.path.join(result_dir, 'tta_pred_mask')
        tta_pred_dir = tta_inference(MODEL_PATH, MODEL_SIZE, test_img_dir, tta_output)
    elif TTA_ENABLED:
        print('WARNING: TTA 已启用但 MODEL_PATH 无效, 跳过 TTA')

    # 自动搜索 pred_mask 目录
    pred_dir = None
    for root, dirs, files in os.walk(result_dir):
        if 'pred_mask' in dirs:
            candidate = os.path.join(root, 'pred_mask')
            if len(os.listdir(candidate)) > 0:
                pred_dir = candidate
                break
    if pred_dir is None:
        print('WARNING: 找不到 pred_mask 文件夹或文件夹为空!')
        print('  将只生成训练曲线 (从 history.json)')
    else:
        print(f'预测目录: {pred_dir} ({len(os.listdir(pred_dir))} 文件)')

    # 验证 GT 目录
    if not os.path.isdir(gt_dir):
        print(f'WARNING: GT 目录不存在: {gt_dir}')
        gt_dir = None

    # 验证影像目录
    if not os.path.isdir(test_img_dir):
        print(f'WARNING: 影像目录不存在: {test_img_dir}')
        test_img_dir = None

    # 1. 训练曲线
    history_path = os.path.join(result_dir, 'history.json')
    if os.path.exists(history_path):
        print('\n[1/6] 绘制训练曲线...')
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        plot_training_curves(history, save_dir)

        best_idx = int(np.argmax(history['test_miou']))
        print(f'  Best test mIoU: {history["test_miou"][best_idx]:.4f} @ epoch {history["epoch"][best_idx]}')
    else:
        print('[1/6] 未找到 history.json, 跳过')

    # 如果没有 pred_mask, 只画曲线就结束
    if pred_dir is None or gt_dir is None:
        print('\n缺少 pred_mask 或 GT, 只生成了训练曲线。')
        print(f'结果保存在: {save_dir}')
        return

    # 2. 计算指标
    print('\n[2/6] 计算测试集指标...')
    metrics = compute_metrics(pred_dir, gt_dir)

    print(f'\n{"="*60}')
    print(f'Overall Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Mean IoU:         {metrics["miou"]:.4f}')
    print(f'{"="*60}')
    print(f'{"Class":<16} {"IoU":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}')
    print('-' * 50)
    for i in range(NUM_CLASSES):
        print(f'{CLASS_NAMES[i]:<16} {metrics["iou_per_class"][i]:>8.4f} '
              f'{metrics["precision"][i]:>8.4f} {metrics["recall"][i]:>8.4f} '
              f'{metrics["f1"][i]:>8.4f}')
    print('-' * 50)

    # 2.5 TTA 指标 (如果启用了 TTA)
    if tta_pred_dir and os.path.isdir(tta_pred_dir):
        print('\n[2.5/6] 计算 TTA 指标...')
        tta_metrics = compute_metrics(tta_pred_dir, gt_dir)

        print(f'\n{"="*60}')
        print(f'TTA Results (4-fold: orig + hflip + vflip + rot180)')
        print(f'Overall Accuracy: {tta_metrics["accuracy"]:.4f}')
        print(f'Mean IoU:         {tta_metrics["miou"]:.4f}')
        print(f'{"="*60}')
        print(f'{"Class":<16} {"IoU":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}  {"vs原始":>8}')
        print('-' * 60)
        for i in range(NUM_CLASSES):
            delta = tta_metrics['iou_per_class'][i] - metrics['iou_per_class'][i]
            sign = '+' if delta >= 0 else ''
            print(f'{CLASS_NAMES[i]:<16} {tta_metrics["iou_per_class"][i]:>8.4f} '
                  f'{tta_metrics["precision"][i]:>8.4f} {tta_metrics["recall"][i]:>8.4f} '
                  f'{tta_metrics["f1"][i]:>8.4f}  {sign}{delta:>7.4f}')
        delta_miou = tta_metrics['miou'] - metrics['miou']
        sign = '+' if delta_miou >= 0 else ''
        print('-' * 60)
        print(f'{"mIoU 提升":<16} {sign}{delta_miou:.4f} ({metrics["miou"]:.4f} -> {tta_metrics["miou"]:.4f})')

        # TTA 的可视化用 tta_metrics
        vis_metrics = tta_metrics
        vis_pred_dir = tta_pred_dir
        save_dir_tta = os.path.join(result_dir, 'visualization_tta')
        os.makedirs(save_dir_tta, exist_ok=True)
    else:
        vis_metrics = metrics
        vis_pred_dir = pred_dir
        save_dir_tta = save_dir

    # 3. 混淆矩阵
    print('\n[3/6] 绘制混淆矩阵...')
    plot_confusion_matrix(vis_metrics['hist'], save_dir_tta)

    # 4. IoU 柱状图
    print('\n[4/6] 绘制 IoU 柱状图...')
    plot_iou_bar(vis_metrics, save_dir_tta)

    # 5. mIoU 分布
    print('\n[5/6] 绘制 mIoU 分布...')
    plot_miou_distribution(vis_metrics, save_dir_tta)

    # 6. 可视化
    print('\n[6/6] 绘制预测可视化...')
    plot_worst_predictions(vis_metrics, vis_pred_dir, gt_dir, test_img_dir, save_dir_tta, n=4)
    plot_top_foreground(vis_metrics, vis_pred_dir, gt_dir, test_img_dir, save_dir_tta, n=4)

    print(f'\n{"="*60}')
    print(f'全部完成! 结果保存在: {save_dir_tta}')
    print(f'共 {len(os.listdir(save_dir_tta))} 个文件')


if __name__ == '__main__':
    main()
