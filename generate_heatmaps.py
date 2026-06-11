"""
generate_heatmaps.py
────────────────────
MWFormer 공식 코드로 복원 이미지를 생성하고, residual로 weather heatmap을 한 번에 만듦.

  restored = MWFormer(I)
  H_t      = mean(|I - restored|, dim=channel)   ← raw residual heatmap [0,1] (npy 저장)
  (loss 가중치 W_t = exp(-alpha * H_t) 는 학습 시점에 heatmap_loss.py 에서 계산)

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
                    device: torch.device) -> tuple:
    """
    Returns
    -------
    heatmap_raw : (H, W) float32 [0, 1]  — raw residual heatmap (npy 저장용).
        두 [0,1] 이미지의 abs diff 이므로 별도 정규화 없이 이미 [0,1] 범위이며,
        프레임 간 비교가 가능하다. loss 가중치 W=exp(-alpha*H) 는 학습 시점에 계산.
    heatmap_vis : (H, W) float32 [0, 1]  — per-image min-max 정규화 (PNG 시각화 전용).
    """
    img_t = img_t.to(device)
    _, _, H, W = img_t.shape
    # MWFormer 디코더는 입력 크기가 stride 배수여야 함 — 32 단위로 패딩
    STRIDE = 32
    pad_h = (STRIDE - H % STRIDE) % STRIDE
    pad_w = (STRIDE - W % STRIDE) % STRIDE
    if pad_h > 0 or pad_w > 0:
        img_padded = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        img_padded = img_t
    with torch.no_grad():
        feature_vec = style_filter(img_padded)
        restored    = backbone(img_padded, feature_vec)[..., :H, :W].clamp(0, 1)

    residual = (img_t - restored).abs()           # [1, 3, H, W]
    heatmap  = residual.mean(dim=1).squeeze(0)    # [H, W], 이미 [0,1]

    heatmap_raw = heatmap.cpu().numpy().astype(np.float32)

    # 시각화 전용: per-image min-max 스트레치 (npy 에는 반영하지 않음)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap_vis = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_vis = torch.zeros_like(heatmap)
    heatmap_vis = heatmap_vis.cpu().numpy().astype(np.float32)

    return heatmap_raw, heatmap_vis


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
                        help="[deprecated] 더 이상 사용되지 않음. "
                             "npy 에는 raw heatmap H 를 저장하고, "
                             "W=exp(-alpha*H) 는 학습 시 --heatmap_alpha 로 적용된다.")
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
    print(f"[info] {len(img_files)}개 이미지 처리 시작 "
          f"(npy=raw heatmap H, W=exp(-alpha*H)는 학습 시 적용)")

    for fname in img_files:
        stem  = os.path.splitext(fname)[0]
        img_t = to_tensor(Image.open(
            os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0)

        heatmap_raw, heatmap_vis = compute_heatmap(
            style_filter, backbone, img_t, device)

        np.save(os.path.join(out_dir, stem + ".npy"), heatmap_raw)

        png = Image.fromarray(
            (heatmap_vis * 255).clip(0, 255).astype(np.uint8))
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
