import pandas as pd
df = pd.read_csv(r"E:\月球_dataset\output\result45\epoch_metrics.txt")
print(df[['epoch','val_miou','val_iou_0','val_iou_1','val_iou_2','val_iou_3','val_iou_4']].tail(20))
print('\nBest epoch:')
print(df.loc[df['val_miou'].idxmax()])