import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/song110585-cpu/lunar-linear.git"
REPO_BRANCH = "test-new-module"
REPO_DIR = Path("/kaggle/working/lunar-linear")
PROJECT_DIR = REPO_DIR / "LTL-Net"
OUTPUT_ROOT = Path("/kaggle/working")
DATA_CANDIDATES = [
    Path("/kaggle/input/datav6-overlap40/dataset_v6_random811_overlap40"),
    Path("/kaggle/input/datasets/yuanssy/datav6-overlap40/dataset_v6_random811_overlap40"),
]


required = [("rasterio", "rasterio"), ("tqdm", "tqdm")]
missing = [package for module, package in required if importlib.util.find_spec(module) is None]
try:
    smp_version = importlib.metadata.version("segmentation-models-pytorch")
except importlib.metadata.PackageNotFoundError:
    smp_version = None
if smp_version != "0.5.0":
    missing.append("segmentation-models-pytorch==0.5.0")
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import numpy as np
import torch

assert torch.cuda.is_available(), "Kaggle GPU is not enabled"
subprocess.check_call(
    ["git", "clone", "--branch", REPO_BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)]
)
commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
).strip()

existing = [path for path in DATA_CANDIDATES if path.is_dir()]
assert existing, "overlap40 dataset was not mounted"
data_root = existing[0]
expected_tiles = {"train": 1598, "val": 200, "test": 200}
for split, expected in expected_tiles.items():
    images = list((data_root / split / "image").glob("*.tif"))
    masks = list((data_root / split / "mask").glob("*.tif"))
    assert len(images) == len(masks) == expected, (split, len(images), len(masks))
stats = json.loads((data_root / "normalization_stats.json").read_text(encoding="utf-8"))
expected_mean = np.array([0.15665339073973303, 0.6052870962271574, 0.22171011101838023, 0.5087022443378417, 0.46687463729626205])
expected_std = np.array([0.07239327406001447, 0.35159567816693277, 0.23999408652260576, 0.18305312443820845, 0.18653673179588806])
assert np.allclose(stats["mean"], expected_mean, rtol=0, atol=1e-12)
assert np.allclose(stats["std"], expected_std, rtol=0, atol=1e-12)

run_name = "v6_overlap40_dsconv_resnet50_seed42_formal80_valfg"
command = [
    sys.executable,
    str(PROJECT_DIR / "scripts" / "train_module_experiment.py"),
    "--module", "dsconv",
    "--data-dir", str(data_root),
    "--output-dir", str(OUTPUT_ROOT),
    "--run-name", run_name,
    "--seed", "42",
    "--epochs", "80",
    "--batch-size", "2",
    "--accum-steps", "2",
    "--num-workers", "2",
]
print("GPU:", torch.cuda.get_device_name(0))
print("Commit:", commit)
print("Data:", data_root)
print("Command:", " ".join(command))
subprocess.check_call(command, cwd=PROJECT_DIR)

result_dir = OUTPUT_ROOT / f"result_{run_name}"
assert (result_dir / "metrics.json").is_file()
archive = shutil.make_archive(str(OUTPUT_ROOT / result_dir.name), "zip", root_dir=result_dir)
print("Result archive:", archive)
