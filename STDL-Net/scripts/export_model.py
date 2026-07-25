"""
把 Kaggle 下载的 best_base 目录 → 转为 .pth 文件.
"""
import os, sys, argparse, shutil, tempfile, zipfile

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ['datasets', 'utils', 'models']:
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='best_base 目录路径')
    parser.add_argument('--out', '-o', default=None)
    args = parser.parse_args()

    src = args.src
    if not os.path.isdir(src):
        raise FileNotFoundError(f'不是目录: {src}')

    # 1. 打包成 PyTorch 可读的 zip (ZIP_STORED, version 必须是第一个 entry)
    print(f'Packing {src} → temp zip...')
    tmpdir = tempfile.mkdtemp(prefix='torch_legacy_')
    zip_path = os.path.join(tmpdir, 'archive.zip')
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            # version 必须第一个写入
            zf.write(os.path.join(src, 'version'), 'version')
            for fname in sorted(os.listdir(src)):
                if fname != 'version':
                    zf.write(os.path.join(src, fname), fname)

        # 2. torch.load 读取 zip
        print('Loading...')
        state = torch.load(zip_path, map_location='cpu', weights_only=False)

        # 3. 提取 state_dict
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            elif 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']

        clean = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        total = sum(v.numel() for v in clean.values())

        out = args.out or 'model_weights.pth'
        torch.save(clean, out)
        print(f'{len(clean)} tensors, {total/1e6:.1f}M params → {out}')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
