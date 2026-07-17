"""val 目录下所有 test_ 前缀改为 val_"""
import os

VAL_DIR = r"E:\月球_dataset\dataset\datasetv7\val"

for sub in ["image", "mask"]:
    d = os.path.join(VAL_DIR, sub)
    for f in os.listdir(d):
        if f.startswith("test_"):
            old = os.path.join(d, f)
            new = os.path.join(d, f.replace("test_", "val_", 1))
            os.rename(old, new)
            print(f"  {f} -> {os.path.basename(new)}")

print("Done!")
