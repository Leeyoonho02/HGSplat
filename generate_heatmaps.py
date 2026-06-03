"""
generate_heatmaps.py
────────────────────
MWFormer (taco-group/MWFormer) 를 이용해 입력 프레임별 weather weight map을 생성하고 .npy로 저장.

MWFormer 구조:
  StyleFilter  : 입력 이미지 → 날씨 스타일 벡터 (feature_vec)
  Network_top  : (입력 이미지, feature_vec) → 복원 이미지

Heatmap 계산:
  H_t = channel_mean( |I_t - Network_top(I_t, StyleFilter(I_t))| )  → min-max 정규화
  W_t = exp(-alpha * H_t)   ← train.py 가 로드하는 weight map

사용법:
    # MWFormer 레포를 먼저 클론
    git clone https://github.com/taco-group/MWFormer.git

    python generate_heatmaps.py \\
        --mwformer_root  ./MWFormer \\
        --ckpt_backbone  /path/to/backbone.pth \\
        --ckpt_style     /path/to/style_filter.pth \\
        --scene_dir      data/YOUR_SCENE/images \\
        --out_dir        data/YOUR_SCENE/heatmaps \\
        --alpha          5.0
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
to_tensor = transforms.ToTensor()  # [0, 1] float32


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def load_image(path: str) -> torch.Tensor:
    """이미지를 [1, C, H, W] float32 텐서로 로드."""
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def load_mwformer(mwformer_root: str, ckpt_backbone: str, ckpt_style: str, device: torch.device):
    """
    MWFormer 의 두 모델을 로드해 반환.

    Returns
    -------
    style_filter : StyleFilter_Top  (eval, frozen)
    backbone     : Network_top      (eval)
    """
    # MWFormer 레포를 sys.path 에 추가
    sys.path.insert(0, os.path.abspath(mwformer_root))
    from model.EncDec import Network_top
    from model.style_filter64 import StyleFilter_Top

    # StyleFilter
    style_filter = StyleFilter_Top().to(device)
    style_filter = nn.DataParallel(style_filter)
    style_filter.load_state_dict(torch.load(ckpt_style, map_location=device))
    style_filter.eval()
    for p in style_filter.parameters():
        p.requires_grad = False

    # Backbone (Network_top)
    backbone = Network_top().to(device)
    backbone = nn.DataParallel(backbone)
    backbone.load_state_dict(torch.load(ckpt_backbone, map_location=device))
    backbone.eval()

    print("[model] StyleFilter + Network_top 로드 완료.")
    return style_filter, backbone


def compute_weight_map(
    style_filter: nn.Module,
    backbone: nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    alpha: float,
) -> np.ndarray:
    """
    MWFormer 잔차로 weight map 계산.

      feature_vec = StyleFilter(I_t)
      restored    = Network_top(I_t, feature_vec)
      H_t         = channel_mean(|I_t - restored|)  → min-max 정규화
      W_t         = exp(-alpha * H_t)

    Returns
    -------
    weight_map : np.ndarray, shape (H, W), float32, 값 범위 (0, 1]
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        feature_vec = style_filter(img_t)           # 날씨 스타일 벡터
        restored    = backbone(img_t, feature_vec)  # [1, 3, H, W]

    residual = (img_t - restored).abs()             # [1, 3, H, W]
    heatmap  = residual.squeeze(0).mean(dim=0)      # [H, W]

    # min-max 정규화
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)

    weight = torch.exp(-alpha * heatmap)            # (0, 1]
    return weight.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mwformer_root", default="./MWFormer",
                        help="taco-group/MWFormer 클론 경로 (기본: ./MWFormer)")
    parser.add_argument("--ckpt_backbone", required=True,
                        help="Network_top 체크포인트 경로")
    parser.add_argument("--ckpt_style",    required=True,
                        help="StyleFilter_Top 체크포인트 경로")
    parser.add_argument("--scene_dir",     required=True,
                        help="입력 이미지 폴더 (images/)")
    parser.add_argument("--out_dir",       default=None,
                        help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--alpha",         type=float, default=5.0,
                        help="W_t = exp(-alpha * H_t) 감쇠 계수 (기본 5.0)")
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter, backbone = load_mwformer(
        args.mwformer_root, args.ckpt_backbone, args.ckpt_style, device
    )

    img_files = sorted(
        f for f in os.listdir(args.scene_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        img_t = load_image(os.path.join(args.scene_dir, fname))
        weight_map = compute_weight_map(style_filter, backbone, img_t, device, args.alpha)
        stem = os.path.splitext(fname)[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), weight_map)

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
