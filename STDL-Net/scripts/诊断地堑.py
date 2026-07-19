import os
import cv2
import numpy as np

GT_DIR = r".../mask"
PRED_DIR = r".../prediction"

CLASS_ID = 4       # Graben

gt_pixels = 0
pred_pixels = 0

gt_images = 0
pred_images = 0

for name in sorted(os.listdir(GT_DIR)):

    gt = cv2.imread(os.path.join(GT_DIR, name), 0)
    pred = cv2.imread(os.path.join(PRED_DIR, name), 0)

    gt_mask = (gt == CLASS_ID)
    pred_mask = (pred == CLASS_ID)

    gt_pixels += gt_mask.sum()
    pred_pixels += pred_mask.sum()

    if gt_mask.sum() > 0:
        gt_images += 1

    if pred_mask.sum() > 0:
        pred_images += 1

print("="*50)
print("GT Graben pixels :", gt_pixels)
print("Pred Graben pixels:", pred_pixels)
print("GT images :", gt_images)
print("Pred images:", pred_images)