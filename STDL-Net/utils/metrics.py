import numpy as np
import torch

# TP:422535   TN:7251807   FP:51346   FN:138632
#
# test_loss: 1.0889383726  test_acc: 0.9758430481  test_precision: 0.8916479032  test_recall: 0.7529576757  test_f1: 0.8164548890  test_iou: 0.6898384198
def confusion_matrix(pred, target):
    pred = pred.bool()
    target = target.bool()

    TP = torch.logical_and(pred, target).sum().item()
    TN = torch.logical_and(~pred, ~target).sum().item()
    FP = torch.logical_and(pred, ~target).sum().item()
    FN = torch.logical_and(~pred, target).sum().item()

    return TP, TN, FP, FN
"""这段代码定义了一个 confusion_matrix 函数,用于计算二分类任务中的混淆矩阵指标。让我们逐步分析它的功能:
pred = pred.bool(), target = target.bool():将输入的预测结果 pred 和真实标签 target 都转换为布尔张量。这是为了后续进行逻辑运算。
TP = torch.logical_and(pred, target).sum().item():计算真阳性(True Positive)的数量。torch.logical_and(pred, target) 返回一个布尔张量,表示预测正确的样本。
然后使用 .sum().item() 统计张量中为 True 的元素个数,即为真阳性的数量。
TN = torch.logical_and(~pred, ~target).sum().item():计算真阴性(True Negative)的数量。~pred 和 ~target 分别表示预测为负和真实标签为负的布尔张量。
torch.logical_and(~pred, ~target) 返回一个布尔张量,表示预测和标签都为负的样本。同样使用 .sum().item() 统计元素个数,得到真阴性的数量。
FP = torch.logical_and(pred, ~target).sum().item():计算假阳性(False Positive)的数量。
torch.logical_and(pred, ~target) 返回一个布尔张量,表示预测为正但真实标签为负的样本。统计张量中为 True 的元素个数,得到假阳性的数量。
FN = torch.logical_and(~pred, target).sum().item():计算假阴性(False Negative)的数量。
torch.logical_and(~pred, target) 返回一个布尔张量,表示预测为负但真实标签为正的样本。统计张量中为 True 的元素个数,得到假阴性的数量。
"""

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    return TP / (TP + FN)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def iou_score(TP, FP, FN):
    return TP / (TP + FP + FN)


# ============== 多类别语义分割指标 ==============
def multiclass_confusion(pred, target, num_classes, ignore_index=-1):
    """
    pred, target: LongTensor, shape (B, H, W) 或 (H, W)
    返回 (num_classes, num_classes) 的混淆矩阵, 行=真实类, 列=预测类.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    pred = pred.view(-1).long()
    target = target.view(-1).long()

    mask = (target >= 0) & (target < num_classes) & (target != ignore_index)
    pred = pred[mask]
    target = target[mask]

    idx = num_classes * target + pred
    hist = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist.cpu()


def metrics_from_hist(hist, eps=1e-10):
    """
    输入: (C, C) 混淆矩阵 (行=真实, 列=预测).
    返回: dict, 包含 per-class / mean 的 iou, precision, recall, f1 以及 overall accuracy.
    """
    hist = hist.float()
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / (hist.sum() + eps)

    return {
        'iou_per_class': iou.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'miou': iou.mean().item(),
        'mprecision': precision.mean().item(),
        'mrecall': recall.mean().item(),
        'mf1': f1.mean().item(),
        'accuracy': acc.item(),
    }

#
# TP=466535
# TN=7251807
# FP=43346
# FN=102632
# p=precision(TP,FP)
# r=recall(TP,FN)
# f1=f1_score(p,r)
# iou=iou_score(TP,FP,FN)

# print(p,r,f1,iou)



