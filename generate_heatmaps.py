"""
generate_heatmaps.py
────────────────────
MWFormer(WeatherEdit)를 이용해 입력 프레임별 weather heatmap을 생성하고 .npy로 저장.

실행 환경: yoonho_weatheredit (PyTorch 1.12 + CUDA 11.6)
실행 위치: WeatherEdit/General_Scene/

사용법:
    python generate_heatmaps.py \
        --scene_dir /path/to/scene/images \
        --ckpt     /path/to/mwformer.pth \
        --out_dir  /path/to/scene/heatmaps \
        --alpha    5.0
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# WeatherEdit 루트를 sys.path에 추가
WEATHEREDIT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WEATHEREDIT_ROOT)

# MWFormer 모델 import (WeatherEdit 레포 구조 기준)
try:
    from basicsr.models.archs.mwformer_arch import MWFormer  # WeatherEdit 내부 경로
except ImportError:
    raise ImportError(
        "MWFormer를 찾을 수 없습니다. "
        "WeatherEdit/General_Scene/ 디렉토리에서 실행하세요."
    )


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

to_tensor = transforms.ToTensor()   # [0,1] float32 tensor


def load_image(path: str) -> torch.Tensor:
    """이미지를 [1, C, H, W] float32 텐서로 로드."""
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0)   # [1, 3, H, W]


def compute_heatmap(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    alpha: float,
) -> np.ndarray:
    """
    MWFormer 잔차(residual)로 heatmap을 계산.

    H_t = mean_channel( |I_t - MWFormer(I_t)| )   (정규화 후)
    W_t = exp(-alpha * H_t)

    Returns
    -------
    weight_map : np.ndarray, shape (H, W), dtype float32
        값 범위 (0, 1] — 날씨 영향이 클수록 0에 가까움.
    """
    img_t = img_t.to(device)
    with torch.no_grad():
        restored = model(img_t)          # [1, 3, H, W]

    # 잔차 → 채널 평균 → [H, W]
    residual = (img_t - restored).abs()  # [1, 3, H, W]
    heatmap = residual.squeeze(0).mean(dim=0)  # [H, W]

    # min-max 정규화 → [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        heatmap = (heatmap - h_min) / (h_max - h_min)

    # weight map
    weight = torch.exp(-alpha * heatmap)  # [H, W], (0, 1]

    return weight.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True,
                        help="입력 이미지 폴더 (images/)")
    parser.add_argument("--ckpt", required=True,
                        help="MWFormer 체크포인트 경로 (.pth)")
    parser.add_argument("--out_dir", default=None,
                        help="출력 폴더 (기본: scene_dir/../heatmaps)")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="weight map 감쇠 계수 (기본 5.0)")
    parser.add_argument("--device", default="cuda",
                        help="cuda 또는 cpu")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(scene_dir), "heatmaps"
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── 모델 로드 ──
    print(f"[model] Loading MWFormer from {args.ckpt}")
    model = MWFormer()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # 체크포인트 포맷에 따라 key 조정
    state = ckpt.get("params", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print("[model] Ready.")

    # ── 이미지 목록 ──
    img_files = sorted(
        f for f in os.listdir(scene_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    print(f"[info] {len(img_files)} images found in {scene_dir}")

    # ── 처리 ──
    for fname in img_files:
        img_path = os.path.join(scene_dir, fname)
        img_t = load_image(img_path)

        weight_map = compute_heatmap(model, img_t, device, args.alpha)

        stem = os.path.splitext(fname)[0]
        npy_path = os.path.join(out_dir, f"{stem}.npy")
        np.save(npy_path, weight_map)

    print(f"[done] Heatmaps saved to {out_dir}")


if __name__ == "__main__":
    main()
