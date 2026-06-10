"""
generate_heatmaps.py
────────────────────
MWFormer 복원 결과와 원본 이미지의 residual로 weather heatmap 생성.

MWFormer 공식 코드로 복원 이미지를 먼저 생성한 뒤 이 스크립트를 실행:
  H_t    = mean(|I - restored|, dim=channel)   ← 날씨 픽셀에서 residual이 큼
  H_t    = min-max 정규화 → [0, 1]
  W_t    = exp(-alpha * H_t)                   ← photometric loss 가중치

Colab 워크플로우:
  1. MWFormer 공식 inference 실행 → restored/ 폴더에 저장
  2. python generate_heatmaps.py \\
         --input_dir   data/snow_scene/images    \\
         --restored_dir data/snow_scene/restored \\
         --out_dir     data/snow_scene/heatmaps  \\
         --alpha       5.0
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def compute_heatmap(img_t: torch.Tensor, restored_t: torch.Tensor,
                    alpha: float) -> tuple:
    """
    Parameters
    ----------
    img_t      : [1, 3, H, W] float32 [0, 1]  — 원본 (날씨 포함)
    restored_t : [1, 3, H, W] float32 [0, 1]  — MWFormer 복원 이미지
    alpha      : W_t = exp(-alpha * H_t)

    Returns
    -------
    heatmap_np : (H, W) float32 [0, 1]  — 시각화용 (밝을수록 날씨 픽셀)
    weight_map : (H, W) float32 (0, 1]  — loss 가중치용
    """
    # 해상도 불일치 시 restored를 원본 크기로 맞춤
    if img_t.shape != restored_t.shape:
        restored_t = torch.nn.functional.interpolate(
            restored_t, size=img_t.shape[2:], mode="bilinear", align_corners=False)

    residual = (img_t - restored_t).abs()          # [1, 3, H, W]
    heatmap  = residual.mean(dim=1).squeeze(0)     # [H, W]

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = torch.zeros_like(heatmap)

    heatmap_np = heatmap.cpu().numpy().astype(np.float32)
    weight_map = np.exp(-alpha * heatmap_np).astype(np.float32)
    return heatmap_np, weight_map


def find_pair(name: str, restored_dir: str) -> str:
    """stem 이름으로 복원 이미지 파일 탐색 (확장자 무관)."""
    for ext in IMG_EXTS:
        path = os.path.join(restored_dir, name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"복원 이미지를 찾을 수 없음: {restored_dir}/{name}.*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",    required=True, help="원본 이미지 폴더")
    parser.add_argument("--restored_dir", required=True, help="MWFormer 복원 이미지 폴더")
    parser.add_argument("--out_dir",      default=None,  help="출력 폴더 (기본: input_dir/../heatmaps)")
    parser.add_argument("--alpha",        type=float, default=5.0,
                        help="W_t = exp(-alpha * H_t)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.input_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(f for f in os.listdir(args.input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작 (alpha={args.alpha})")

    for fname in img_files:
        stem = os.path.splitext(fname)[0]

        img_t      = to_tensor(Image.open(
            os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0)
        restored_t = to_tensor(Image.open(
            find_pair(stem, args.restored_dir)).convert("RGB")).unsqueeze(0)

        heatmap_np, weight_map = compute_heatmap(img_t, restored_t, args.alpha)

        np.save(os.path.join(out_dir, stem + ".npy"), weight_map)

        png = Image.fromarray(
            (heatmap_np * 255).clip(0, 255).astype(np.uint8))
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
