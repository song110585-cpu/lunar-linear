import os
os.environ["WANDB_API_KEY"] = '3d0f14304695197a773e59b027afa3b3c4ca46e1'

import wandb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import models
from torch import optim
import matplotlib.pyplot as plt
import matplotlib
import time
import metrics
from MyDataset import MyDataset
import datetime
import numpy as np
# from deeplabv3plus_res18 import DeepLabV3Plus

# from swinunet import SwinUnet
from swinv2unet import SwinUnet,Swin_LCSRB_PSP_FPNPAN,Swin_LCSRB,Swin_FPNPAN,Swin_PSP,Swin_DeformablePSP,\
        Swin_LCSRB_PSP,Swin_LCSRB_FPNPAN,Swin_FPNPAN_PSP,Swin_LCSRB_DeformablePSP_FPNPAN,Swin_FPNPAN_DeformablePSP,Swin_LCSRB_DeformablePSP




def showTensor(data, title="data"):
    data = data.cpu()

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(data.detach().numpy(), cmap='gray')
    plt.title(title)

    plt.show()
    plt.close(fig)

def denormalize(tensor_image, mean=None, std=None):
    """多通道反归一化, 返回用于可视化的 uint8 图像 (单通道 HxW 或三通道 HxWx3)."""
    C = tensor_image.shape[0]
    img = tensor_image.clone().float().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(C, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(C, 1, 1)
        img = img * std_t + mean_t
    if C >= 3:
        img = img[:3]
    img_np = img.numpy()
    mn, mx = img_np.min(), img_np.max()
    if mx - mn > 1e-8:
        img_np = (img_np - mn) / (mx - mn)
    else:
        img_np = np.zeros_like(img_np)
    img_np = (img_np * 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        return img_np[0]
    return img_np.transpose(1, 2, 0)

def savePredictResult(optical, label, pred, save_path, title,
                      num_classes=5, mean=None, std=None):
    """多类别分割结果可视化. optical: (C,H,W); label, pred: (H,W) long."""
    optical_vis = denormalize(optical.cpu(), mean=mean, std=std)

    label_np = label.cpu().detach().numpy().astype(np.uint8)
    pred_np = pred.cpu().detach().numpy().astype(np.uint8)

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(optical_vis, cmap='gray' if optical_vis.ndim == 2 else None)
    plt.title('optical (first 3 bands)')
    plt.subplot(1, 3, 2)
    plt.imshow(label_np, cmap='tab10', vmin=0, vmax=max(num_classes - 1, 1))
    plt.title('label')
    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='tab10', vmin=0, vmax=max(num_classes - 1, 1))
    plt.title('predict')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, title + '.png'))
    from PIL import Image as _PILImage
    _PILImage.fromarray(label_np).save(os.path.join(save_path, title + '_y.png'))
    _PILImage.fromarray(pred_np).save(os.path.join(save_path, title + '_pre.png'))
    plt.close(fig)


def net_test(model, test_iter, loss, record_path, num_classes,
             epoch='000', save=False, vis_mean=None, vis_std=None, max_steps=0):
    """多类别分割评估: CrossEntropyLoss + 混淆矩阵 / mIoU."""
    model.eval()
    test_epoch_loss = []
    hist_total = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    with torch.no_grad(), torch.cuda.amp.autocast():
        for t_step, (optical, label, img_name) in enumerate(tqdm(test_iter, desc=f'test     Epoch {epoch}    :', unit='img')):
            if max_steps > 0 and t_step >= max_steps:
                break
            optical, label = optical.to(device), label.to(device)
            logits = model(optical)                        # (B, C, H, W)
            l = loss(logits, label)                        # CE 要求 label (B,H,W) long
            test_epoch_loss.append(l.item())

            pred = logits.argmax(dim=1)                    # (B, H, W)
            hist_total += metrics.multiclass_confusion(pred, label, num_classes).double()

            if save:
                title = 'test_{}_{}'.format(img_name[0], str(int(time.time() * 100)))
                savePredictResult(optical[0], label[0], pred[0],
                                  os.path.join(record_path, f'test_{epoch}'), title,
                                  num_classes=num_classes, mean=vis_mean, std=vis_std)

    test_epoch_loss = float(np.average(test_epoch_loss)) if test_epoch_loss else 0.0
    m = metrics.metrics_from_hist(hist_total)
    print('- ' * 30)
    print('test_loss: {:.6f}  acc: {:.4f}  mIoU: {:.4f}  mF1: {:.4f}  mPrec: {:.4f}  mRec: {:.4f}'.format(
        test_epoch_loss, m['accuracy'], m['miou'], m['mf1'], m['mprecision'], m['mrecall']))
    print('per-class IoU: ' + ', '.join(f'{v:.4f}' for v in m['iou_per_class']))
    print('per-class F1 : ' + ', '.join(f'{v:.4f}' for v in m['f1_per_class']))
    print('- ' * 30)
    return m['miou']


def train(model, train_iter, val_iter, loss, opt, num_epochs, record_path, lr_scheduler,
          num_classes=5, vis_mean=None, vis_std=None, accum_steps=1, max_steps=0):
    """多类别分割训练循环. label 形状 (B,H,W) long, 损失为 CrossEntropyLoss."""
    scaler = torch.cuda.amp.GradScaler()
    best_miou = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_epoch_loss = []
        hist_total = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        opt.zero_grad()
        for step, (optical, label, img_name) in enumerate(tqdm(train_iter, desc=f'Epoch {epoch}/{num_epochs} ', unit='img')):
            if max_steps > 0 and step >= max_steps:
                break
            optical, label = optical.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                logits = model(optical)                 # (B, C, H, W)
                l = loss(logits, label) / accum_steps   # 梯度累积需除以步数

            scaler.scale(l).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_iter):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            train_epoch_loss.append(l.item() * accum_steps)
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                hist_total += metrics.multiclass_confusion(pred, label, num_classes).double()

        train_epoch_loss = float(np.average(train_epoch_loss)) if train_epoch_loss else 0.0
        m = metrics.metrics_from_hist(hist_total)
        print('- ' * 30)
        print('train_loss: {:.6f}  acc: {:.4f}  mIoU: {:.4f}  mF1: {:.4f}  mPrec: {:.4f}  mRec: {:.4f}'.format(
            train_epoch_loss, m['accuracy'], m['miou'], m['mf1'], m['mprecision'], m['mrecall']))
        print('per-class IoU: ' + ', '.join(f'{v:.4f}' for v in m['iou_per_class']))
        print('- ' * 30)

        lr_scheduler.step()
        val_miou = net_test(model=model, test_iter=val_iter, loss=loss,
                            record_path=record_path, num_classes=num_classes,
                            epoch=str(epoch), save=True,
                            vis_mean=vis_mean, vis_std=vis_std,
                            max_steps=max_steps)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model, hp.model_save_path)
            print(f'save best model at epoch {epoch}, mIoU={best_miou:.4f}')


if __name__ == '__main__':
    # =========================
    # 超参数
    # =========================
    NUM_CLASSES  = 5       # 0=背景, 1=皱脊, 2=月溪, 3=断层, 4=地堑
    IN_CHANNELS  = 5       # WAC, DEM, Slope, TPI, 剖面曲率
    MODEL_SIZE   = 'small'  # 'tiny' 验证流程, 正式训练改为 'small' 或 'base'
    FREEZE_STAGES = 1      # 冻结 backbone 前 N 个阶段 (0=不冻结)
    USE_AUGMENT   = True
    USE_COPYPASTE = True       # CopyPaste 增强 (针对 Fault/Graben 少数类)
    COPYPASTE_P   = 0.5        # CopyPaste 触发概率

    class HyperParameter:
        def __init__(self):
            curr_time = datetime.datetime.now()
            curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
            self.name = "_result" + curr_time_str
            self.num_epochs = 60   # 快速验证用 2, 正式训练改为 30~50
            self.max_steps =0   # 每 epoch 只跑 50 步 (0=跑全部)
            self.learning_rate = 5e-5
            self.train_batchsize = 1
            self.test_batchsize = 1
            self.accum_steps = 4          # 梯度累积, 等效 batch_size = 1*4 = 4
            self.patience = 12            # Early stopping patience
            # ---- 数据路径 ----
            self.train_image_dir = r'E:\月球_dataset\Research area\train\dataset\image'
            self.train_mask_dir  = r'E:\月球_dataset\Research area\train\dataset\mask'
            self.test_image_dir  = r'E:\月球_dataset\Research area\test\dataset\image'
            self.test_mask_dir   = r'E:\月球_dataset\Research area\test\dataset\mask'
            # ---- 输出路径 ----
            self.record_path = r'E:\月球_dataset\Research area\result'
            self.model_save_path = os.path.join(self.record_path, self.name + '.pt')

    hp = HyperParameter()
    os.makedirs(hp.record_path, exist_ok=True)

    # =========================
    # 设备
    # =========================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}     GPU available:{torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        print('GPU not available')
        exit()

    # =========================
    # 模型 (5 通道输入, 5 类输出)
    # =========================
    model = Swin_LCSRB_DeformablePSP_FPNPAN(
        size=MODEL_SIZE,
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        pretrained=True,
    ).to(device)

    if FREEZE_STAGES > 0:
        model.freeze_backbone_stages(FREEZE_STAGES)

    # =========================
    # 数据集 + 数据增强 (R16: 翻转+旋转+噪声+Cutout+CopyPaste)
    # =========================
    import random

    class AugmentedDataset(torch.utils.data.Dataset):
        """强化增强: 翻转 + 旋转 + 高斯噪声 + 亮度扰动 + Cutout + CopyPaste"""
        def __init__(self, base_dataset, copypaste=False, copypaste_p=0.5):
            self.base = base_dataset
            self.copypaste = copypaste
            self.copypaste_p = copypaste_p
            if self.copypaste:
                self.rare_indices = []
                print('CopyPaste: 预索引少数类样本...')
                for i in range(len(self.base)):
                    _, mask, _ = self.base[i]
                    classes_present = set(mask.unique().tolist())
                    if 3 in classes_present or 4 in classes_present:
                        self.rare_indices.append(i)
                print(f'CopyPaste: 找到 {len(self.rare_indices)} 张含 Fault/Graben 的样本')

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, mask, name = self.base[idx]
            if self.copypaste and self.rare_indices and random.random() < self.copypaste_p:
                donor_idx = random.choice(self.rare_indices)
                donor_img, donor_mask, _ = self.base[donor_idx]
                rare_fg = (donor_mask == 3) | (donor_mask == 4)
                if rare_fg.any():
                    fg_mask = rare_fg.unsqueeze(0).expand_as(img)
                    img = torch.where(fg_mask, donor_img, img)
                    mask = torch.where(rare_fg, donor_mask, mask)
            if random.random() > 0.5:
                img = img.flip(-1); mask = mask.flip(-1)
            if random.random() > 0.5:
                img = img.flip(-2); mask = mask.flip(-2)
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, [-2, -1])
                mask = torch.rot90(mask, k, [-2, -1])
            if random.random() > 0.5:
                img = img + torch.randn_like(img) * 0.02
            if random.random() > 0.5:
                C = img.shape[0]
                img = img + (torch.rand(C, 1, 1) - 0.5) * 0.1
            if random.random() > 0.7:
                _, H, W = img.shape
                for _ in range(random.randint(1, 3)):
                    sz = random.randint(32, 64)
                    y, x = random.randint(0, H - sz), random.randint(0, W - sz)
                    img[:, y:y+sz, x:x+sz] = 0.0
            return img, mask, name

    train_data_raw = MyDataset(
        images_dir=hp.train_image_dir,
        masks_dir=hp.train_mask_dir,
    )
    if USE_AUGMENT:
        train_data = AugmentedDataset(
            train_data_raw,
            copypaste=USE_COPYPASTE,
            copypaste_p=COPYPASTE_P,
        )
        aug_str = '翻转+旋转+噪声+Cutout'
        if USE_COPYPASTE:
            aug_str += f'+CopyPaste(p={COPYPASTE_P})'
        print(f'数据增强: {aug_str}')
    else:
        train_data = train_data_raw

    train_iter = DataLoader(
        dataset=train_data, batch_size=hp.train_batchsize,
        shuffle=True, drop_last=False,
        num_workers=0, pin_memory=True,
    )

    test_data = MyDataset(
        images_dir=hp.test_image_dir,
        masks_dir=hp.test_mask_dir,
    )
    test_iter = DataLoader(
        dataset=test_data, batch_size=hp.test_batchsize,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True,
    )

    print(f'训练集: {len(train_data)} 张,  测试集: {len(test_data)} 张')

    # =========================
    # 损失函数 (R13: CE + 0.5*Dice)
    # =========================
    class_weights = torch.tensor([0.15, 1.0, 1.3, 1.8, 1.5], dtype=torch.float32).to(device)
    print('class_weights:', [f'{w:.4f}' for w in class_weights.tolist()])
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(logits, targets, smooth=1.0):
        """前景类 Dice Loss"""
        probs = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(1, logits.shape[1]):
            p = probs[:, c]
            g = (targets == c).float()
            inter = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice += (1 - (2 * inter + smooth) / (union + smooth)).mean()
        return dice / (logits.shape[1] - 1)

    def combined_loss(logits, targets):
        return ce_loss_fn(logits, targets) + 0.5 * dice_loss(logits, targets)

    loss = combined_loss
    print('Loss: CE(weights) + 0.5*Dice')

    # =========================
    # 优化器 & 调度器
    # =========================
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp.num_epochs, eta_min=1e-6)

    # =========================
    # 训练
    # =========================
    train(
        model=model,
        train_iter=train_iter,
        val_iter=test_iter,
        loss=loss,
        opt=opt,
        num_epochs=hp.num_epochs,
        record_path=hp.record_path,
        lr_scheduler=scheduler,
        num_classes=NUM_CLASSES,
        accum_steps=hp.accum_steps,
        max_steps=hp.max_steps,
    )

    # 保存最终模型 (最优模型在 train() 里已单独保存)
    torch.save(model, os.path.join(hp.record_path, hp.name + '_final.pt'))