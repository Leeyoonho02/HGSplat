"""
generate_heatmaps.py
────────────────────
MWFormer 로 입력 프레임별 weather weight map 을 생성하고 .npy 로 저장.

MWFormer 구조 (mwformer/ 패키지로 내장):
  StyleFilter_Top : 입력 이미지 → 64-dim 날씨 style vector
  Network_top     : (이미지, style vector) → 복원 이미지

Heatmap 계산:
  H_t = channel_mean( |I_t - Network_top(I_t, StyleFilter_Top(I_t))| )
  W_t = exp(-alpha * H_t)   ← [0,1] 정규화 후

사용법:
    python generate_heatmaps.py \\
        --ckpt_backbone  /path/to/backbone.pth \\
        --ckpt_style     /path/to/style_filter.pth \\
        --scene_dir      data/YOUR_SCENE/images \\
        --out_dir        data/YOUR_SCENE/heatmaps \\
        --alpha          5.0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from mwformer import Network_top, StyleFilter_Top

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def load_models(ckpt_backbone: str, ckpt_style: str,
                device: torch.device) -> tuple[nn.Module, nn.Module]:
    style_filter = nn.DataParallel(StyleFilter_Top().to(device))
    style_filter.load_state_dict(torch.load(ckpt_style, map_location=device))
    style_filter.eval()
    for p in style_filter.parameters():
        p.requires_grad = False

    backbone = nn.DataParallel(Network_top().to(device))
    backbone.load_state_dict(torch.load(ckpt_backbone, map_location=device))
    backbone.eval()

    print("[model] StyleFilter_Top + Network_top 로드 완료")
    return style_filter, backbone


def compute_weight_map(style_filter: nn.Module, backbone: nn.Module,
                       img_t: torch.Tensor, device: torch.device,
                       alpha: float) -> np.ndarray:
    """
    W_t = exp(-alpha * H_t),  H_t ∈ [0,1]
    반환: np.ndarray shape (H, W), float32, 값 범위 (0, 1]
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        feat     = style_filter(img_t)
        restored = backbone(img_t, feat)

    heatmap = (img_t - restored).abs().squeeze(0).mean(dim=0)  # [H, W]

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)

    return torch.exp(-alpha * heatmap).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_backbone", required=True,
                        help="Network_top 체크포인트 (.pth)")
    parser.add_argument("--ckpt_style",    required=True,
                        help="StyleFilter_Top 체크포인트 (.pth)")
    parser.add_argument("--scene_dir",     required=True,
                        help="입력 이미지 폴더")
    parser.add_argument("--out_dir",       default=None,
                        help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--alpha",         type=float, default=5.0,
                        help="감쇠 계수 (기본 5.0, train.py --heatmap_alpha 와 맞출 것)")
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter, backbone = load_models(args.ckpt_backbone, args.ckpt_style, device)

    img_files = sorted(f for f in os.listdir(args.scene_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        img_t = load_image(os.path.join(args.scene_dir, fname))
        w_map = compute_weight_map(style_filter, backbone, img_t, device, args.alpha)
        np.save(os.path.join(out_dir, os.path.splitext(fname)[0] + ".npy"), w_map)

    print(f"[done] {len(img_files)}개 weight map 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
