"""
2D Weather Synthesis — HGSplat Exp2
파이프라인:
  clean → [fixed fog overlay] → grass_fog
  grass_fog → [RandomSnow]   → grass_snow
  grass_fog → [RandomRain]   → grass_rain

사용법:
  python synthesize_weather_2d.py --clean_dir /path/to/clean/images --out_root /path/to/Dataset
  (--scene_name 기본값 grass)
"""

import argparse
import glob
import os
import random

import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── 날씨 변환 (fog는 아래 apply_fog() 함수로 별도 처리) ──────────────────────
WEATHER_TRANSFORMS = {
    'snow': A.RandomSnow(snow_point_range=(0.1, 0.3), brightness_coeff=2.5, p=1.0),
    'rain': A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1,
                         drop_color=(200, 200, 200), blur_value=3,
                         brightness_coefficient=0.9, rain_type='heavy', p=1.0),
}


def apply_fog(img: np.ndarray, coef: float = 0.18) -> np.ndarray:
    """고정 세기 안개 — RandomFog 대신 단순 white blend로 안정적 결과 보장."""
    white = np.full_like(img, 235)  # 약간 따뜻한 흰색
    return np.clip(img * (1 - coef) + white * coef, 0, 255).astype(np.uint8)


def load_images(src_dir: str):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(src_dir, ext)))
    paths = sorted(paths)
    assert paths, f'이미지 없음: {src_dir}'
    return paths


def run_stage(paths, out_dir, transform=None, seed_offset=0):
    """paths의 이미지를 transform 적용 후 out_dir에 저장. transform=None이면 fog만 적용된 원본."""
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(paths):
        np.random.seed(seed_offset + i)
        img = np.array(Image.open(path).convert('RGB'))
        if transform is not None:
            img = transform(image=img)['image']
        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(img).save(os.path.join(out_dir, f'{stem}.png'))
        if i % 30 == 0 or i == len(paths) - 1:
            print(f'  [{i+1:>4}/{len(paths)}] {stem}')
    print(f'[done] {len(paths)}장 → {out_dir}')


def save_sanity(clean_paths, fog_dir, snow_dir, rain_dir, out_path, n=4):
    samples = random.sample(clean_paths, min(n, len(clean_paths)))
    rows = [('Clean', None), ('Fog', fog_dir), ('Snow', snow_dir), ('Rain', rain_dir)]
    rows = [(l, d) for l, d in rows if d is not None or l == 'Clean']  # fog=None이면 행 생략
    fig, axes = plt.subplots(len(rows), len(samples), figsize=(5 * len(samples), 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for j, path in enumerate(samples):
        stem = os.path.splitext(os.path.basename(path))[0]
        for row, (label, d) in enumerate(rows):
            if d is None:
                img = np.array(Image.open(path).convert('RGB'))
            else:
                img = np.array(Image.open(os.path.join(d, f'{stem}.png')))
            axes[row][j].imshow(img)
            axes[row][j].set_title(f'{label}  {stem}', fontsize=9)
            axes[row][j].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'[sanity] {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir',  required=True, help='clean 원본 이미지 폴더')
    parser.add_argument('--out_root',   required=True, help='출력 루트 (Dataset 폴더)')
    parser.add_argument('--scene_name', default='grass', help='장면 이름 prefix')
    parser.add_argument('--fog_coef',   type=float, default=0.18, help='안개 블렌드 강도 (0~1)')
    parser.add_argument('--skip_fog',   action='store_true', help='fog 없이 clean에 눈/비만 적용')
    parser.add_argument('--sanity',     action='store_true')
    args = parser.parse_args()

    clean_paths = load_images(args.clean_dir)

    if args.skip_fog:
        # fog 없이 clean → snow_only / rain_only
        snow_dir = os.path.join(args.out_root, f'{args.scene_name}_snow_only', 'images')
        rain_dir = os.path.join(args.out_root, f'{args.scene_name}_rain_only', 'images')
        print(f'[config] scene={args.scene_name}, fog=SKIP')
        print(f'  clean → {snow_dir}')
        print(f'  clean → {rain_dir}')

        print('\n[Stage 1] Snow only (no fog) ...')
        run_stage(clean_paths, snow_dir, transform=WEATHER_TRANSFORMS['snow'], seed_offset=0)

        print('\n[Stage 2] Rain only (no fog) ...')
        run_stage(clean_paths, rain_dir, transform=WEATHER_TRANSFORMS['rain'], seed_offset=100)

        if args.sanity:
            out_png = os.path.join(args.out_root, f'{args.scene_name}_sanity_nofog.png')
            save_sanity(clean_paths, None, snow_dir, rain_dir, out_png)
    else:
        # fog → snow / fog → rain
        fog_dir  = os.path.join(args.out_root, f'{args.scene_name}_fog',  'images')
        snow_dir = os.path.join(args.out_root, f'{args.scene_name}_snow', 'images')
        rain_dir = os.path.join(args.out_root, f'{args.scene_name}_rain', 'images')
        print(f'[config] scene={args.scene_name}, fog_coef={args.fog_coef}')
        print(f'  clean → {fog_dir}')
        print(f'  fog   → {snow_dir}')
        print(f'  fog   → {rain_dir}')

        print('\n[Stage 1] Fog overlay ...')
        os.makedirs(fog_dir, exist_ok=True)
        for i, path in enumerate(clean_paths):
            img = np.array(Image.open(path).convert('RGB'))
            out = apply_fog(img, coef=args.fog_coef)
            stem = os.path.splitext(os.path.basename(path))[0]
            Image.fromarray(out).save(os.path.join(fog_dir, f'{stem}.png'))
            if i % 30 == 0 or i == len(clean_paths) - 1:
                print(f'  [{i+1:>4}/{len(clean_paths)}] {stem}')
        print(f'[done] {len(clean_paths)}장 → {fog_dir}')

        fog_paths = load_images(fog_dir)

        print('\n[Stage 2a] Snow on fog ...')
        run_stage(fog_paths, snow_dir, transform=WEATHER_TRANSFORMS['snow'], seed_offset=0)

        print('\n[Stage 2b] Rain on fog ...')
        run_stage(fog_paths, rain_dir, transform=WEATHER_TRANSFORMS['rain'], seed_offset=100)

        if args.sanity:
            out_png = os.path.join(args.out_root, f'{args.scene_name}_sanity_check.png')
            save_sanity(clean_paths, fog_dir, snow_dir, rain_dir, out_png)
