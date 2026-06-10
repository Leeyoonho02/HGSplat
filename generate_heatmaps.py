"""
generate_heatmaps.py
────────────────────
Rule-based weather heatmap 생성 (외부 모델 불필요).

눈/비 픽셀 특성:
  - 밝고 (high luminance)
  - 주변 배경과 대비가 급격함 (작은 blob)

Heatmap 계산:
  1. luminance  = 0.299R + 0.587G + 0.114B
  2. H_t        = relu(lum - GaussianBlur(lum, σ_large))  → 배경보다 밝은 픽셀
  3. H_t        = min-max 정규화 → [0, 1]
  4. W_t        = exp(-alpha * H_t)

사용법:
    python generate_heatmaps.py \\
        --scene_dir  data/YOUR_SCENE/images \\
        --out_dir    data/YOUR_SCENE/heatmaps \\
        --sigma_l    15.0  \\
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


def compute_heatmap(img_t: torch.Tensor, sigma_l: float,
                    alpha: float, device: torch.device) -> tuple:
    """
    Rule-based snow/rain heatmap (로컬 대비 기반).

    눈/비 입자는 절대적으로 밝지 않아도 주변 배경보다 밝음.
    H_t = relu(lum - GaussianBlur(lum, σ_large)) 로 배경 대비 밝은 픽셀 감지.

    Parameters
    ----------
    img_t  : [1, 3, H, W] float32 [0, 1]
    sigma_l: 배경 추정용 blur sigma (클수록 넓은 영역 평균과 비교)
    alpha  : W_t = exp(-alpha * H_t)
    """
    img_t = img_t.to(device)

    # 1. Luminance [1, 1, H, W]
    lum_w = torch.tensor([0.299, 0.587, 0.114],
                          device=device).view(1, 3, 1, 1)
    lum = (img_t * lum_w).sum(dim=1, keepdim=True)

    # 2. 로컬 대비: 픽셀 - 배경 평균 → 주변보다 밝은 픽셀만 양수
    background = gaussian_blur(lum, sigma_l)
    heatmap = (lum - background).clamp(min=0).squeeze()  # [H, W]

    # 3. min-max 정규화
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
    parser.add_argument("--sigma_l",    type=float, default=15.0, help="배경 추정 blur sigma (클수록 더 넓은 배경과 비교)")
    parser.add_argument("--alpha",      type=float, default=5.0,  help="W_t = exp(-alpha * H_t)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[params] sigma_l={args.sigma_l}, alpha={args.alpha}")

    img_files = sorted(f for f in os.listdir(args.scene_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        stem  = os.path.splitext(fname)[0]
        img_t = to_tensor(Image.open(
            os.path.join(args.scene_dir, fname)).convert("RGB")).unsqueeze(0)

        heatmap_np, weight_map = compute_heatmap(
            img_t, args.sigma_l, args.alpha, device)

        np.save(os.path.join(out_dir, stem + ".npy"), weight_map)

        # 시각화: heatmap (밝을수록 날씨 픽셀)
        png = Image.fromarray(
            (heatmap_np * 255).clip(0, 255).astype(np.uint8), mode="L")
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
