"""
generate_heatmaps.py
────────────────────
StyleFilter_Top 인코더의 공간 feature map으로 픽셀별 weather weight map 생성.
Network_top(복원 backbone) 체크포인트 불필요 — StyleFilter 체크포인트 하나만 사용.

Heatmap 계산:
  enc[0]: [1, 64,  H/4, W/4]  — stage 1 feature (채널 L2 norm)
  enc[1]: [1, 128, H/8, W/8]  — stage 2 feature (채널 L2 norm)
  H_t = upsample_avg(norm(enc[0]), norm(enc[1]))  → min-max 정규화 → [0, 1]
  W_t = exp(-alpha * H_t)

사용법:
    python generate_heatmaps.py \\
        --ckpt_style  /path/to/style_filter.pth \\
        --scene_dir   data/YOUR_SCENE/images \\
        --out_dir     data/YOUR_SCENE/heatmaps \\
        --alpha       5.0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from mwformer import StyleFilter_Top

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
to_tensor = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)  # [1, 3, H, W]


def load_style_filter(ckpt_style: str, device: torch.device) -> nn.Module:
    style_filter = nn.DataParallel(StyleFilter_Top().to(device))
    style_filter.load_state_dict(torch.load(ckpt_style, map_location=device))
    style_filter.eval()
    for p in style_filter.parameters():
        p.requires_grad = False
    print("[model] StyleFilter_Top 로드 완료")
    return style_filter


def compute_weight_map(style_filter: nn.Module, img_t: torch.Tensor,
                       device: torch.device, alpha: float) -> np.ndarray:
    """
    StyleFilter 인코더 feature map → weight map W_t.

    Returns
    -------
    np.ndarray shape (H, W), float32, 값 범위 (0, 1]
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        # DataParallel 내부 모듈의 encode_spatial 호출
        heatmap = style_filter.module.encode_spatial(img_t)  # [H, W], [0,1]

    weight = torch.exp(-alpha * heatmap)  # (0, 1]
    return weight.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_style", required=True,
                        help="StyleFilter_Top 체크포인트 (.pth) — 하나만 필요")
    parser.add_argument("--scene_dir",  required=True,
                        help="입력 이미지 폴더")
    parser.add_argument("--out_dir",    default=None,
                        help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--alpha",      type=float, default=5.0,
                        help="W_t = exp(-alpha * H_t) 감쇠 계수 (기본 5.0)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.scene_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter = load_style_filter(args.ckpt_style, device)

    img_files = sorted(f for f in os.listdir(args.scene_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작")

    for fname in img_files:
        stem   = os.path.splitext(fname)[0]
        img_t  = load_image(os.path.join(args.scene_dir, fname))
        w_map  = compute_weight_map(style_filter, img_t, device, args.alpha)
        np.save(os.path.join(out_dir, stem + ".npy"), w_map)

        # PNG 시각화: weight map (0~1] → grayscale (0=날씨 픽셀, 255=정상)
        png = Image.fromarray((w_map * 255).clip(0, 255).astype(np.uint8), mode="L")
        png.save(os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 weight map 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
