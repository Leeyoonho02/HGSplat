"""
generate_heatmaps_appearance.py
───────────────────────────────
Appearance 기반 snow heatmap 생성 (모델·GPU 불필요).

눈 = 작고(small) + 주변보다 밝고(local bright) + 무채색(흰색).
  - White top-hat: 큰 균일 영역(벽) 제거, 작은 밝은 구조만 남김
  - 무채색 + 밝기 게이트: 색 있는 자갈/흙/식물 제외, 흰 눈만 통과
  - 절대 스케일 → 프레임 간 일관 (per-image min-max 안 씀)

출력 (heatmap_loss.py 와 동일 contract):
  {out_dir}/{stem}.npy : raw heatmap H [0,1]  (W=exp(-alpha*H)는 학습 시 적용)
  {out_dir}/{stem}.png : 시각화 (밝을수록 눈)

사용법:
    python generate_heatmaps_appearance.py \\
        --input_dir data/snow_scene/images \\
        --out_dir   data/snow_scene/heatmaps

residual 방식(generate_heatmaps.py)이 눈 대신 텍스처를 추적하던 문제(#2)에 대한
appearance 기반 대안. 정지한 흰 장면요소(식물 크림잎·밝은 자갈)는 일부 오검출될 수 있음.
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage.morphology import white_tophat, disk
from skimage.color import rgb2hsv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def snow_heatmap(rgb01, radius, sat_thr, val_thr, spread_thr, th_thr, th_scale):
    """Returns raw snow heatmap H [0,1] (밝을수록 눈일 확률 높음)."""
    hsv = rgb2hsv(rgb01)
    val = hsv[..., 2]                       # 밝기
    sat = hsv[..., 1]                       # 채도
    spread = rgb01.max(2) - rgb01.min(2)    # 무채색일수록 0
    th = white_tophat(val, disk(radius))    # 작은 밝은 구조 (눈)
    inten = np.clip((th - th_thr) / th_scale, 0.0, 1.0)
    gate = (sat < sat_thr) & (val > val_thr) & (spread < spread_thr)
    H = (inten * gate).astype(np.float32)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="원본 이미지 폴더")
    p.add_argument("--out_dir", default=None, help="출력 폴더 (기본: input_dir/../heatmaps)")
    # appearance 하이퍼파라미터 (recall↑ 쪽으로 약간 완화한 기본값)
    p.add_argument("--radius",     type=int,   default=5,    help="white top-hat 구조요소 반경")
    p.add_argument("--sat_thr",    type=float, default=0.18, help="채도 상한 (초과 시 snow 아님)")
    p.add_argument("--val_thr",    type=float, default=0.50, help="밝기 하한")
    p.add_argument("--spread_thr", type=float, default=0.13, help="채널 스프레드 상한 (무채색)")
    p.add_argument("--th_thr",     type=float, default=0.08, help="top-hat 응답 하한")
    p.add_argument("--th_scale",   type=float, default=0.20, help="top-hat 응답 정규화 스케일")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(f for f in os.listdir(args.input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작 (appearance snow heatmap)")

    cover = []
    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        rgb = np.asarray(Image.open(
            os.path.join(args.input_dir, fname)).convert("RGB")).astype(np.float32) / 255.0
        H = snow_heatmap(rgb, args.radius, args.sat_thr, args.val_thr,
                         args.spread_thr, args.th_thr, args.th_scale)
        np.save(os.path.join(out_dir, stem + ".npy"), H)
        Image.fromarray((H * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, stem + ".png"))
        cover.append((H > 0.3).mean() * 100)

    cover = np.array(cover)
    print(f"[done] {len(img_files)}개 저장 → {out_dir}")
    print(f"[stat] 검출(snow) 비율: 평균 {cover.mean():.2f}%  "
          f"min {cover.min():.2f}%  max {cover.max():.2f}%")


if __name__ == "__main__":
    main()
