"""
generate_heatmaps.py
────────────────────
MWFormer 공식 코드로 복원 이미지를 생성하고, residual로 weather heatmap을 한 번에 만듦.

  restored = MWFormer(I)
  H_t      = mean(|I - restored|, dim=channel)   ← min-max 정규화 → [0, 1]
  W_t      = exp(-alpha * H_t)                   ← photometric loss 가중치

사전 조건:
  Colab 셀에서 MWFormer 레포 클론:
    !git clone https://github.com/taco-group/MWFormer /content/MWFormer

사용법:
    python generate_heatmaps.py \\
        --mwformer_dir  /content/MWFormer                        \\
        --ckpt_style    /path/to/MWFormer_real/style_filter      \\
        --ckpt_backbone /path/to/MWFormer_real/backbone          \\
        --input_dir     data/snow_scene/images                   \\
        --out_dir       data/snow_scene/heatmaps                 \\
        --alpha         5.0
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def load_models(mwformer_dir: str, ckpt_style: str, ckpt_backbone: str,
                device: torch.device):
    """MWFormer 공식 모델 클래스를 import하고 체크포인트 로드."""
    if mwformer_dir not in sys.path:
        sys.path.insert(0, mwformer_dir)

    from model.EncDec import Network_top
    from model.style_filter64 import StyleFilter_Top

    style_filter = nn.DataParallel(StyleFilter_Top().to(device))
    style_filter.load_state_dict(
        torch.load(ckpt_style, map_location=device), strict=True)
    style_filter.eval()
    for p in style_filter.parameters():
        p.requires_grad = False

    backbone = nn.DataParallel(Network_top().to(device))
    backbone.load_state_dict(
        torch.load(ckpt_backbone, map_location=device), strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print("[model] MWFormer (StyleFilter + Network_top) 로드 완료")
    return style_filter, backbone


def compute_heatmap(style_filter, backbone, img_t: torch.Tensor,
                    alpha: float, device: torch.device) -> tuple:
    """
    Returns
    -------
    heatmap_np : (H, W) float32 [0, 1]  — 시각화 (밝을수록 날씨 픽셀)
    weight_map : (H, W) float32 (0, 1]  — loss 가중치
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        feature_vec = style_filter(img_t)
        restored    = backbone(img_t, feature_vec).clamp(0, 1)

    residual = (img_t - restored).abs()           # [1, 3, H, W]
    heatmap  = residual.mean(dim=1).squeeze(0)    # [H, W]

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
    parser.add_argument("--mwformer_dir",  default="/content/MWFormer",
                        help="taco-group/MWFormer 클론 경로")
    parser.add_argument("--ckpt_style",    required=True,
                        help="StyleFilter 체크포인트")
    parser.add_argument("--ckpt_backbone", required=True,
                        help="Network_top 체크포인트")
    parser.add_argument("--input_dir",     required=True,
                        help="원본 이미지 폴더")
    parser.add_argument("--out_dir",       default=None,
                        help="출력 폴더 (기본: input_dir/../heatmaps)")
    parser.add_argument("--alpha",         type=float, default=5.0,
                        help="W_t = exp(-alpha * H_t)")
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.input_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter, backbone = load_models(
        args.mwformer_dir, args.ckpt_style, args.ckpt_backbone, device)

    img_files = sorted(f for f in os.listdir(args.input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작 (alpha={args.alpha})")

    for fname in img_files:
        stem  = os.path.splitext(fname)[0]
        img_t = to_tensor(Image.open(
            os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0)

        heatmap_np, weight_map = compute_heatmap(
            style_filter, backbone, img_t, args.alpha, device)

        np.save(os.path.join(out_dir, stem + ".npy"), weight_map)

        png = Image.fromarray(
            (heatmap_np * 255).clip(0, 255).astype(np.uint8))
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
