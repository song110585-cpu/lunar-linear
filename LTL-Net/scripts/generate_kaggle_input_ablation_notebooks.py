"""Generate four Kaggle notebooks for the controlled input-channel ablation."""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"

EXPERIMENTS = {
    "full": ("I0：完整5通道控制", "v6_overlap40_deeplab_input_full_batch4_seed42.json"),
    "wac_only": ("I1：仅WAC", "v6_overlap40_deeplab_input_wac_only_batch4_seed42.json"),
    "terrain_only": ("I2：仅地形通道", "v6_overlap40_deeplab_input_terrain_only_batch4_seed42.json"),
    "wac_dem": ("I3：WAC+DEM", "v6_overlap40_deeplab_input_wac_dem_batch4_seed42.json"),
}


def source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def build_notebook(mode: str, title: str, config_name: str) -> dict:
    cells = [
        markdown(
            f"# Kaggle影像消融：{title}\n\n"
            "DeepLabV3+-ResNet50、seed42、physical batch4、80 epochs；"
            "只改变有效输入通道，不读取Test指标。\n"
        ),
        code(
            "from pathlib import Path\n"
            "import importlib.metadata, importlib.util, json, os, subprocess, sys\n\n"
            "REPO_URL = 'https://github.com/song110585-cpu/lunar-linear.git'\n"
            "REPO_BRANCH = 'test-new-module'\n"
            "REQUIRED_COMMIT = '6d33bf0'\n"
            "REPO_DIR = Path('/kaggle/working/lunar-linear')\n"
            "PROJECT_DIR = REPO_DIR / 'LTL-Net'\n"
            "OUTPUT_ROOT = Path('/kaggle/working')\n"
            f"CONFIG_NAME = '{config_name}'\n"
            f"EXPECTED_MODE = '{mode}'\n\n"
            "DATA_CANDIDATES = [\n"
            "    Path('/kaggle/input/datasets/yuanssy/datav6-overlap40/dataset_v6_random811_overlap40'),\n"
            "    Path('/kaggle/input/datasets/changyasong/datav6-overlap40/dataset_v6_random811_overlap40'),\n"
            "    Path('/kaggle/input/datasets/changyasong/v6data/dataset_v6_random811_overlap40'),\n"
            "    Path('/kaggle/input/datav6-overlap40/dataset_v6_random811_overlap40'),\n"
            "    Path('/kaggle/input/v6data/dataset_v6_random811_overlap40'),\n"
            "]\n"
        ),
        markdown("## 1. 环境与代码\n"),
        code(
            "required = [('rasterio', 'rasterio'), ('tqdm', 'tqdm')]\n"
            "missing = [package for module, package in required if importlib.util.find_spec(module) is None]\n"
            "try:\n"
            "    smp_version = importlib.metadata.version('segmentation-models-pytorch')\n"
            "except importlib.metadata.PackageNotFoundError:\n"
            "    smp_version = None\n"
            "if smp_version != '0.5.0':\n"
            "    missing.append('segmentation-models-pytorch==0.5.0')\n"
            "if missing:\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])\n\n"
            "if not REPO_DIR.exists():\n"
            "    subprocess.check_call(['git', 'clone', '--branch', REPO_BRANCH, '--single-branch', REPO_URL, str(REPO_DIR)])\n"
            "elif not (REPO_DIR / '.git').is_dir():\n"
            "    raise RuntimeError(f'目录存在但不是Git仓库: {REPO_DIR}')\n"
            "else:\n"
            "    subprocess.check_call(['git', '-C', str(REPO_DIR), 'pull', '--ff-only', 'origin', REPO_BRANCH])\n"
            "subprocess.check_call(['git', '-C', str(REPO_DIR), 'merge-base', '--is-ancestor', REQUIRED_COMMIT, 'HEAD'])\n"
            "commit = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', '--short', 'HEAD'], text=True).strip()\n"
            "print('Git commit:', commit)\n"
            "subprocess.run(['nvidia-smi'], check=False)\n"
        ),
        markdown("## 2. 自动定位并核验Kaggle数据\n"),
        code(
            "discovered = sorted({\n"
            "    path for path in Path('/kaggle/input').rglob('dataset_v6_random811_overlap40')\n"
            "    if path.is_dir()\n"
            "})\n"
            "ordered = []\n"
            "for path in [*DATA_CANDIDATES, *discovered]:\n"
            "    if path not in ordered and (path / 'dataset_protocol.json').is_file():\n"
            "        ordered.append(path)\n"
            "assert ordered, '未找到dataset_v6_random811_overlap40；请先用Add Input挂载数据集。'\n"
            "DATA_ROOT = ordered[0]\n"
            "CONFIG_PATH = PROJECT_DIR / 'configs' / CONFIG_NAME\n"
            "config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "assert config['channel_mode'] == EXPECTED_MODE, config['channel_mode']\n"
            "assert config['seed'] == 42 and config['epochs'] == 80\n"
            "assert config['batch_size'] == 4 and config['accum_steps'] == 1\n"
            "assert config['automatic_test_evaluation'] is False\n"
            "print('数据目录:', DATA_ROOT)\n"
            "print('实验:', config['run_name'])\n"
            "print('channel_mode:', config['channel_mode'])\n"
            "print('输出:', OUTPUT_ROOT / f\"result_{config['run_name']}\")\n"
        ),
        markdown("## 3. 开始训练\n"),
        code(
            "command = [\n"
            "    sys.executable, str(PROJECT_DIR / 'scripts/run_autodl_channel_ablation.py'),\n"
            "    '--project-dir', str(PROJECT_DIR),\n"
            "    '--config', str(CONFIG_PATH),\n"
            "    '--data-dir', str(DATA_ROOT),\n"
            "    '--output-dir', str(OUTPUT_ROOT),\n"
            "]\n"
            "env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'\n"
            "print(' '.join(command), flush=True)\n"
            "subprocess.check_call(command, cwd=PROJECT_DIR, env=env)\n"
        ),
        markdown("## 4. 结果检查与下载\n"),
        code(
            "result_dir = OUTPUT_ROOT / f\"result_{config['run_name']}\"\n"
            "metrics_path = result_dir / 'metrics.json'\n"
            "archive_path = Path(str(result_dir) + '.zip')\n"
            "assert metrics_path.is_file(), metrics_path\n"
            "assert archive_path.is_file(), archive_path\n"
            "metrics = json.loads(metrics_path.read_text(encoding='utf-8'))\n"
            "assert metrics['test_evaluated'] is False\n"
            "print(json.dumps(metrics, ensure_ascii=False, indent=2))\n"
            "print('下载:', archive_path)\n"
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"{mode}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for mode, (title, config_name) in EXPERIMENTS.items():
        path = NOTEBOOK_DIR / f"kaggle_v6_overlap40_deeplab_input_{mode}_seed42.ipynb"
        path.write_text(
            json.dumps(build_notebook(mode, title, config_name), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
