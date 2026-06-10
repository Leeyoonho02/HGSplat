"""
generate_heatmaps.py
────────────────────
Rule-based weather heatmap 생성 (외부 모델 불필요).

눈/비 픽셀 특성:
  - 밝고 (high luminance)
  - 주변 배경과 대비가 급격함 (작은 blob)

Heatmap 계산:
  1. luminance  = 0.299R + 0.587G + 0.114B
  2. DoG        = GaussianBlur(σ_small) - GaussianBlur(σ_large)  → 작은 blob 감지
  3. H_t        = luminance_mask * relu(DoG)  → min-max 정규화 → [0, 1]
  4. W_t        = exp(-alpha * H_t)           → photometric loss 가중치

사용법:
    python generate_heatmaps.py \\
        --scene_dir  data/YOUR_SCENE/images \\
        --out_dir    data/YOUR_SCENE/heatmaps \\
        --sigma_s    1.0   \\
        --sigma_l    5.0   \\
        --bright_thr 0.6   \\
        --alpha      5.0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """1D Gaussian 커널 → 2D separable conv용."""
    radius = int(4 * sigma + 0.5)
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: [1, C, H, W] → separable Gaussian blur."""
    device = img.device
    k = gaussian_kernel(sigma, device)
    C = img.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    pad = k.shape[0] // 2
    x = F.conv2d(img, kx, padding=(0, pad), groups=C)
    x = F.conv2d(x,   ky, padding=(pad, 0), groups=C)
    return x


def compute_heatmap(img_t: torch.Tensor, sigma_s: float, sigma_l: float,
                    bright_thr: float, alpha: float,
                    device: torch.device) -> tuple:
    """
    Rule-based snow/rain heatmap.

    Parameters
    ----------
    img_t     : [1, 3, H, W] float32 [0, 1]
    sigma_s   : DoG small sigma (작은 blob 내부)
    sigma_l   : DoG large sigma (배경 추정)
    bright_thr: luminance 임계값 (0~1), 이 이상만 날씨 픽셀로 간주
    alpha     : W_t = exp(-alpha * H_t)

    Returns
    -------
    heatmap_np : (H, W) float32 [0, 1]  — 시각화용 (밝을수록 날씨 픽셀)
    weight_map : (H, W) float32 (0, 1]  — loss 가중치용
    """
    img_t = img_t.to(device)

    # 1. Luminance [1, 1, H, W]
    weights = torch.tensor([0.299, 0.587, 0.114],
                            device=device).view(1, 3, 1, 1)
    lum = (img_t * weights).sum(dim=1, keepdim=True)  # [1, 1, H, W]

    # 2. DoG: 작은 blob → 양수, 배경 구조 → 0 근처
    blur_s = gaussian_blur(lum, sigma_s)
    blur_l = gaussian_blur(lum, sigma_l)
    dog = (blur_s - blur_l).clamp(min=0)               # relu(DoG)

    # 3. 고밝기 마스크
    bright_mask = (lum >= bright_thr).float()

    # 4. 결합
    heatmap = (dog * bright_mask).squeeze()             # [H, W]

    # 5. min-max 정규화
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = torch.zeros_like(heatmap)

    heatmap_np = heatmap.cpu().numpy().astype(np.float32)
    weight_map = np.exp(-alpha * heatmap_np).astype(np.float32)
    return heatmap_np, weight_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir",  required=True,  help="입력 이미지 폴더")
    parser.add_argument("--out_dir",    default=None,   help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--sigma_s",    type=float, default=1.0,  help="DoG small sigma")
    parser.add_argument("--sigma_l",    type=float, default=5.0,  help="DoG large sigma")
    parser.add_argument("--bright_thr", type=float, default=0.6,  help="luminance 임계값 [0~1]")
    parser.add_argument("--alpha",      type=float, default=5.0,  help="W_t = exp(-alpha * H_t)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[params] sigma_s={args.sigma_s}, sigma_l={args.sigma_l}, "
          f"bright_thr={args.bright_thr}, alpha={args.alpha}")

    img_files = sorted(f for f in os.listdir(args.scene_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        stem  = os.path.splitext(fname)[0]
        img_t = to_tensor(Image.open(
            os.path.join(args.scene_dir, fname)).convert("RGB")).unsqueeze(0)

        heatmap_np, weight_map = compute_heatmap(
            img_t, args.sigma_s, args.sigma_l,
            args.bright_thr, args.alpha, device)

        np.save(os.path.join(out_dir, stem + ".npy"), weight_map)

        # 시각화: heatmap (밝을수록 날씨 픽셀)
        png = Image.fromarray(
            (heatmap_np * 255).clip(0, 255).astype(np.uint8), mode="L")
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
