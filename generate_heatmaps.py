"""
generate_heatmaps.py
────────────────────
MWFormer (StyleFilter_Top + Network_top) 복원 residual로 픽셀별 weather heatmap 생성.

Heatmap 계산:
  Y      = Network_top(I, StyleFilter_Top(I))   ← MWFormer 복원 이미지
  H_t    = mean(|I - Y|, dim=channel)           ← residual (날씨 픽셀에서 큼)
  H_t    = min-max 정규화 → [0, 1]
  W_t    = exp(-alpha * H_t)                    ← photometric loss 가중치

사용법:
    python generate_heatmaps.py \\
        --ckpt_style    /path/to/style_filter   \\
        --ckpt_backbone /path/to/backbone       \\
        --scene_dir     data/YOUR_SCENE/images  \\
        --out_dir       data/YOUR_SCENE/heatmaps \\
        --alpha         5.0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from mwformer import StyleFilter_Top
from mwformer.backbone import Network_top

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)  # [1, 3, H, W]


def load_models(ckpt_style: str, ckpt_backbone: str,
                device: torch.device) -> tuple:
    # StyleFilter
    style_filter = nn.DataParallel(StyleFilter_Top().to(device))
    style_filter.load_state_dict(torch.load(ckpt_style, map_location=device))
    style_filter.eval()
    for p in style_filter.parameters():
        p.requires_grad = False

    # Network_top (복원 backbone)
    backbone = nn.DataParallel(Network_top().to(device))
    ckpt = torch.load(ckpt_backbone, map_location=device)
    result = backbone.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f"[warn] missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[warn] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print("[model] StyleFilter_Top + Network_top 로드 완료")
    return style_filter, backbone


def compute_heatmap(style_filter: nn.Module, backbone: nn.Module,
                    img_t: torch.Tensor, device: torch.device,
                    alpha: float) -> np.ndarray:
    """
    MWFormer 복원 residual → heatmap H_t → weight map W_t.

    Returns
    -------
    heatmap_vis : np.ndarray (H, W) float32, [0, 1]  — 시각화용 (밝을수록 날씨 픽셀)
    weight_map  : np.ndarray (H, W) float32, (0, 1]  — loss 가중치용
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        feature_vec = style_filter(img_t)                     # weather type vector
        restored    = backbone(img_t, feature_vec)            # [1, 3, H, W]
        restored    = restored.clamp(0, 1)

        residual    = (img_t - restored).abs()                # [1, 3, H, W]
        heatmap     = residual.mean(dim=1).squeeze(0)         # [H, W]

        # min-max 정규화
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
    parser.add_argument("--ckpt_style",    required=True,
                        help="StyleFilter_Top 체크포인트")
    parser.add_argument("--ckpt_backbone", required=True,
                        help="Network_top 체크포인트")
    parser.add_argument("--scene_dir",     required=True,
                        help="입력 이미지 폴더")
    parser.add_argument("--out_dir",       default=None,
                        help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--alpha",         type=float, default=5.0,
                        help="W_t = exp(-alpha * H_t) 감쇠 계수")
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter, backbone = load_models(args.ckpt_style, args.ckpt_backbone, device)

    img_files = sorted(f for f in os.listdir(args.scene_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_t = load_image(os.path.join(args.scene_dir, fname))

        heatmap_np, weight_map = compute_heatmap(
            style_filter, backbone, img_t, device, args.alpha)

        # weight map 저장 (.npy) — train.py에서 사용
        np.save(os.path.join(out_dir, stem + ".npy"), weight_map)

        # heatmap 시각화 (.png) — 밝을수록 날씨 아티팩트
        png = Image.fromarray((heatmap_np * 255).clip(0, 255).astype(np.uint8), mode="L")
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
