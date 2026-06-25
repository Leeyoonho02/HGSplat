"""
2D Weather Synthesis — HGSplat Exp2
snow / rain 합성 (fog 별도 생성 없음).
GPU 불필요, 로컬/Colab 모두 동작.

사용법:
  python synthesize_weather_3dca.py \
      --clean_dir  /path/to/clean/images \
      --out_root   /path/to/Dataset \
      --scene_name grass
"""

# NumPy 2.0 호환 패치
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':     [np.int8, np.int16, np.int32, np.int64],
        'uint':    [np.uint8, np.uint16, np.uint32, np.uint64],
        'float':   [np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
if not hasattr(np, 'float_'):   np.float_   = np.float64
if not hasattr(np, 'int_'):     np.int_     = np.int64
if not hasattr(np, 'complex_'): np.complex_ = np.complex128

import argparse, glob, os, random
import imgaug.augmenters as iaa
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

RAIN_ANGLE = -12   # 고정 빗줄기 방향 (음수 = 왼쪽으로 기울어진 비)
SNOW_ANGLE = -8    # 고정 눈송이 방향


class ImageAddSnow:
    def __init__(self, seed):
        self.seq = iaa.Snowflakes(
            density=(0.2, 0.3),              # imgaug 기본값
            density_uniformity=(0.8, 0.8),
            flake_size=(0.7, 0.9),
            flake_size_uniformity=(0.7, 0.9),
            angle=(SNOW_ANGLE - 1, SNOW_ANGLE + 1),  # 방향만 고정
            speed=(0.007, 0.03),
            seed=seed,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.seq(images=image[None])[0]


class ImageAddRain:
    def __init__(self, seed):
        self.seq = iaa.RainLayer(
            density=(0.10, 0.15),                # 3D_Corruptions_AD severity=3 기본값
            density_uniformity=(0.8, 1.0),
            drop_size=(0.85, 1.0),
            drop_size_uniformity=(0.8, 0.9),
            angle=(RAIN_ANGLE - 1, RAIN_ANGLE + 1),  # 방향만 고정
            speed=(0.04, 0.20),
            blur_sigma_fraction=(0.0001, 0.001),
            blur_sigma_limits=(0.5, 3.75),
            seed=seed,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.seq(images=image[None])[0]


# ─────────────────────────────────────────────────────────────────────────────

def load_paths(src_dir: str):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(src_dir, ext)))
    assert paths, f'이미지 없음: {src_dir}'
    return sorted(paths)


def run(paths, out_dir, transform):
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(paths):
        img = np.array(Image.open(path).convert('RGB'))
        out = transform(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(out).save(os.path.join(out_dir, f'{stem}.png'))
        if i % 30 == 0 or i == len(paths) - 1:
            print(f'  [{i+1:>4}/{len(paths)}] {stem}')
    print(f'[done] {len(paths)}장 → {out_dir}')


def save_sanity(clean_paths, dirs_labels, out_path, n=4):
    samples = random.sample(clean_paths, min(n, len(clean_paths)))
    rows = len(dirs_labels)
    fig, axes = plt.subplots(rows, len(samples), figsize=(5 * len(samples), 4 * rows))
    for j, path in enumerate(samples):
        stem = os.path.splitext(os.path.basename(path))[0]
        for row, (label, d) in enumerate(dirs_labels):
            files = glob.glob(os.path.join(d, f'{stem}.*')) if d else [path]
            img = np.array(Image.open(files[0]).convert('RGB'))
            ax = axes[row][j] if rows > 1 else axes[j]
            ax.imshow(img); ax.set_title(f'{label}  {stem}', fontsize=8); ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'[sanity] {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir',  required=True)
    parser.add_argument('--out_root',   required=True)
    parser.add_argument('--scene_name', default='grass')
    parser.add_argument('--sanity',     action='store_true')
    args = parser.parse_args()

    seed = 42
    snow_dir = os.path.join(args.out_root, f'{args.scene_name}_snow', 'images')
    rain_dir = os.path.join(args.out_root, f'{args.scene_name}_rain', 'images')

    print(f'[config] scene={args.scene_name}  RAIN_ANGLE={RAIN_ANGLE}°  SNOW_ANGLE={SNOW_ANGLE}°')
    paths = load_paths(args.clean_dir)

    print('\n[1/2] Snow ...')
    run(paths, snow_dir, ImageAddSnow(seed=seed))

    print('\n[2/2] Rain ...')
    run(paths, rain_dir, ImageAddRain(seed=seed))

    if args.sanity:
        out_png = os.path.join(args.out_root, f'{args.scene_name}_sanity.png')
        save_sanity(paths, [('Clean', None), ('Snow', snow_dir), ('Rain', rain_dir)], out_png)
