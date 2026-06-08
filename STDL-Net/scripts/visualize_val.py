"""
验证集可视化脚本 - 专用于对验证集 (Validation Set) 107 张预测图进行严格的指标评估与可视化
绝不触碰测试集，严格遵循学术规范！

使用方法:
    1. 修改下面 RESULT_DIR 为你的本地结果文件夹路径 (如 E:\\月球_dataset\\baseline模型结果\\result26)
    2. 直接运行本脚本: python visualize_val.py
    3. 输出图表将保存在 RESULT_DIR/visualization_val/ 下
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ==================== 路径配置 ====================
# [TODO] 只需要改这里！把路径改成你下载的结果文件夹
RESULT_DIR = r"E:\月球_dataset\baseline模型结果\result28"

# 验证集来源自 train 目录，所以 GT 和影像使用 train 目录
VAL_MASK_DIR = r"E:\月球_dataset\Research area\train\dataset_v6\mask"
VAL_IMG_DIR  = r"E:\月球_dataset\Research area\train\dataset_v6\image"

# ==================== 其他配置 ====================
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
    out[gt_fg & pr_fg] = [0, 200, 0]       # TP (Green)
    out[gt_fg & (~pr_fg)] = [255, 0, 0]    # FN (Red)
    out[(~gt_fg) & pr_fg] = [255, 165, 0]  # FP (Orange)
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
    """计算验证集混淆矩阵和各类指标"""
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    per_image = []
    names = []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f'  找到 {len(pred_files)} 张验证预测图')

    for fname in pred_files:
        pred = np.array(Image.open(os.path.join(pred_dir, fname)))
        
        # 兼容不同文件后缀 (.tif 或 .png)
        gt_path = os.path.join(gt_dir, fname.replace('.png', '.tif'))
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

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

        # 单图 mIoU (仅计算图像中实际存在或被预测出的类别，消除全0类别拉低平均值的统计偏置)
        h = np.bincount(
            gt[valid] * NUM_CLASSES + pred[valid],
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        iou = np.diag(h) / (h.sum(0) + h.sum(1) - np.diag(h) + 1e-10)
        
        # 找出该图实际存在真值或被预测出的类别
        present_classes = (h.sum(0) > 0) | (h.sum(1) > 0)
        if present_classes.sum() > 0:
            per_image_mean = float(np.mean(iou[present_classes]))
        else:
            per_image_mean = 0.0
        per_image.append(per_image_mean)
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
    axes[0, 0].plot(ep, history['val_loss'], 's-', ms=3, label='Val')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history['train_miou'], 'o-', ms=3, label='Train')
    axes[0, 1].plot(ep, history['val_miou'], 's-', ms=3, label='Val')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('mIoU Curve'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if 'train_iou_per_class' in history:
        arr = np.array(history['train_iou_per_class'])
        for c in range(min(NUM_CLASSES, arr.shape[1])):
            axes[1, 0].plot(ep, arr[:, c], 'o-', ms=2, lw=1.5,
                            color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Train Per-Class IoU'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    if 'val_iou_per_class' in history:
        arr = np.array(history['val_iou_per_class'])
        for c in range(min(NUM_CLASSES, arr.shape[1])):
            axes[1, 1].plot(ep, arr[:, c], 's-', ms=2, lw=1.5,
                            color=CLASS_COLORS_PLT[c], label=CLASS_NAMES[c])
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_title('Val Per-Class IoU'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

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
    bars = ax.bar(CLASS_NAMES, metrics['iou_per_class'], color=CLASS_COLORS_PLT, edgecolor='black', alpha=0.8)
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (mIoU={metrics["miou"]:.4f})')
    ax.set_ylim(0, 1.05)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iou_per_class.png'), dpi=150)
    plt.close()
    print(f'  保存: iou_per_class.png')


def plot_miou_distribution(metrics, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(metrics['per_image_miou'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['miou'], color='red', linestyle='dashed', linewidth=2, label=f'Mean mIoU ({metrics["miou"]:.4f})')
    ax.set_xlabel('mIoU')
    ax.set_ylabel('Number of Images')
    ax.set_title('mIoU Distribution Over Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'miou_distribution.png'), dpi=150)
    plt.close()
    print(f'  保存: miou_distribution.png')


def plot_worst_predictions(metrics, pred_dir, gt_dir, val_img_dir, save_dir, n=4):
    """可视化表现最差的 n 张图 (IoU 最小的那些，方便错误分析)"""
    per_image_miou = metrics['per_image_miou']
    names = metrics['names']

    # 排序获得 IoU 最小的图像索引
    worst_idx = np.argsort(per_image_miou)[:n]

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

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
        if val_img_dir:
            img_path = os.path.join(val_img_dir, f'{stem}.tif')
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
        axes[row, 3].set_title('Error Map (G=TP,R=FN,O=FP)')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Worst {n} Predictions on Validation Set', fontsize=16)
    fig.legend(handles=get_legend_patches(metrics['iou_per_class']),
               loc='lower center', ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'worst_predictions.png'), dpi=150)
    plt.close()
    print(f'  保存: worst_predictions.png')


def plot_top_foreground(metrics, pred_dir, gt_dir, val_img_dir, save_dir, n=4):
    """可视化前景最多（线性构造最丰富）的 n 张验证集样本"""
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
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

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
        if val_img_dir:
            img_path = os.path.join(val_img_dir, f'{stem}.tif')
            if os.path.exists(img_path):
                try:
                    import rasterio
                    with rasterio.open(img_path) as src:
                        bands = src.read()  # (C, H, W)
                    wac = bands[0].astype(np.float32)
                    wac = (wac - wac.min()) / (wac.max() - wac.min() + 1e-8)
                    if bands.shape[0] > 2:
                        dem = bands[1].astype(np.float32)
                        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
                        slope = bands[2].astype(np.float32)
                        slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)
                except:
                    pass

        # 绘图
        if wac is not None:
            axes[row, 0].imshow(wac, cmap='gray'); axes[row, 0].set_title(f'WAC - {stem}', fontsize=8)
        else:
            axes[row, 0].axis('off')

        if dem is not None:
            axes[row, 1].imshow(dem, cmap='terrain'); axes[row, 1].set_title('DEM')
        else:
            axes[row, 1].axis('off')

        if slope is not None:
            axes[row, 2].imshow(slope, cmap='magma'); axes[row, 2].set_title('Slope')
        else:
            axes[row, 2].axis('off')

        axes[row, 3].imshow(mask_to_color(gt.astype(np.uint8)));   axes[row, 3].set_title('GT')
        axes[row, 4].imshow(mask_to_color(pred));                  axes[row, 4].set_title(f'Pred (mIoU={metrics["per_image_miou"][idx]:.4f})')
        axes[row, 5].imshow(error_map(gt.astype(np.uint8), pred)); axes[row, 5].set_title('Error Map')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Validation Rich Foreground Samples Visualization', fontsize=18)
    fig.legend(handles=get_legend_patches(metrics['iou_per_class']),
               loc='lower center', ncol=5, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'predictions_top_fg.png'), dpi=150)
    plt.close()
    print(f'  保存: predictions_top_fg.png')


def main():
    result_dir = RESULT_DIR
    gt_dir = VAL_MASK_DIR
    val_img_dir = VAL_IMG_DIR

    print(f'正在分析结果目录: {result_dir}')
    
    # 自动定位由训练脚本导出的验证集掩码文件夹（形如 best_epoch_XX_miou_XXXX 或含有 pred_mask 的子文件夹）
    pred_dir = None
    for root, dirs, files in os.walk(result_dir):
        if 'pred_mask' in dirs:
            candidate = os.path.join(root, 'pred_mask')
            # 排除 tta_pred_mask
            if 'tta' not in root.lower() and len(os.listdir(candidate)) > 0:
                pred_dir = candidate
                break
                
    if pred_dir is None:
        print('❌ 错误: 找不到验证集 pred_mask 文件夹!')
        print('   请确认你下载的结果文件夹中包含形如 `best_epoch_XX_miou_XXXX\\pred_mask\\` 的文件夹。')
        return
        
    print(f'🎯 找到验证预测掩码目录: {pred_dir} (含 {len(os.listdir(pred_dir))} 张切片)')
    
    # 验证 GT 目录
    if not os.path.isdir(gt_dir):
        print(f'❌ 错误: 验证真值 GT 目录不存在: {gt_dir}')
        return

    save_dir = os.path.join(result_dir, 'visualization_val')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 训练与验证曲线
    history_path = os.path.join(result_dir, 'history.json')
    if os.path.exists(history_path):
        print('\n[1/6] 绘制训练/验证曲线...')
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        plot_training_curves(history, save_dir)
    else:
        # 递归寻找 history.json
        found_history = False
        for root, dirs, files in os.walk(result_dir):
            if 'history.json' in files:
                print('\n[1/6] 绘制训练/验证曲线...')
                with open(os.path.join(root, 'history.json'), 'r', encoding='utf-8') as f:
                    history = json.load(f)
                plot_training_curves(history, save_dir)
                found_history = True
                break
        if not found_history:
            print('[1/6] 未找到 history.json, 跳过曲线绘制')

    # 2. 计算指标
    print('\n[2/6] 计算验证集(Validation Set)指标...')
    metrics = compute_metrics(pred_dir, gt_dir)

    print(f'\n{"="*60}')
    print(f'验证集最终成绩 (Validation Set Evaluation)')
    print(f'Overall Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Mean mIoU:        {metrics["miou"]:.4f}')
    print(f'{"="*60}')
    print(f'{"Class":<16} {"IoU":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}')
    print('-' * 50)
    for i in range(NUM_CLASSES):
        print(f'{CLASS_NAMES[i]:<16} {metrics["iou_per_class"][i]:>8.4f} '
              f'{metrics["precision"][i]:>8.4f} {metrics["recall"][i]:>8.4f} '
              f'{metrics["f1"][i]:>8.4f}')
    print('-' * 50)

    # 3. 混淆矩阵
    print('\n[3/6] 绘制混淆矩阵...')
    plot_confusion_matrix(metrics['hist'], save_dir)

    # 4. IoU 柱状图
    print('\n[4/6] 绘制各类 IoU 柱状图...')
    plot_iou_bar(metrics, save_dir)

    # 5. mIoU 分布图
    print('\n[5/6] 绘制单张切片 mIoU 分布图...')
    plot_miou_distribution(metrics, save_dir)

    # 6. 绘图可视化 (Worst N 和 Top Foreground)
    print('\n[6/6] 绘制预测切片多通道可视化对照图...')
    plot_worst_predictions(metrics, pred_dir, gt_dir, val_img_dir, save_dir, n=4)
    plot_top_foreground(metrics, pred_dir, gt_dir, val_img_dir, save_dir, n=4)

    print(f'\n🎉 验证集分析完全部完成! 结果图表保存在:')
    print(f'👉 {save_dir}')
    print(f'共 6 个分析图表。此过程未对测试集（Test Set）进行任何干预，完全合乎学术规范。')


if __name__ == '__main__':
    main()
