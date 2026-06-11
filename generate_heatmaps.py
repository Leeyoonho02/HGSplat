"""
generate_heatmaps.py
────────────────────
MWFormer 공식 코드(taco-group/MWFormer)로 입력을 복원하고, residual 로 weather heatmap 생성.

  restored = MWFormer(I)
  H_t      = mean(|I - restored|, dim=channel)   ← raw residual heatmap [0,1] (npy 저장)
  (loss 가중치 W_t = exp(-alpha * H_t) 는 학습 시점에 heatmap_loss.py 에서 계산)

★ 작동하는 MWFormer_Colab_v4_real.ipynb 와 동일한 전처리/로딩을 사용한다.
  - 입력 정규화: ToTensor + Normalize(0.5,0.5,0.5) → [-1, 1]   (이게 빠지면 복원이 망가짐)
  - 리사이즈   : 최대변 1024 제한 후 16배수, LANCZOS
  - 체크포인트 : MWFormer-real (backbone, style_filter), module. prefix strip 후 strict=True
  - residual   : 정규화된 입력을 모델에 넣되, 잔차는 [0,1] 원본 vs restored[0,1] 로 계산

사전 조건 (Colab):
    !git clone https://github.com/taco-group/MWFormer /content/MWFormer

사용법:
    python generate_heatmaps.py \\
        --mwformer_dir  /content/MWFormer                   \\
        --ckpt_style    /path/to/MWFormer-real/style_filter \\
        --ckpt_backbone /path/to/MWFormer-real/backbone     \\
        --input_dir     data/snow_scene/images              \\
        --out_dir       data/snow_scene/heatmaps
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 노트북과 동일: 모델 입력은 [-1,1] 정규화, 잔차 비교용 원본은 [0,1]
to_norm = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
to_01   = ToTensor()


def load_clean_state_dict(model, ckpt_path, device):
    """노트북 load_clean_state_dict 와 동일: module. prefix strip 후 strict=True."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model.to(device)


def load_models(mwformer_dir, ckpt_style, ckpt_backbone, device):
    """MWFormer 공식 모델 클래스 import + MWFormer-real 체크포인트 로드 (bare 모델)."""
    if mwformer_dir not in sys.path:
        sys.path.insert(0, mwformer_dir)
    from model.EncDec import Network_top
    from model.style_filter64 import StyleFilter_Top

    net = load_clean_state_dict(Network_top(), ckpt_backbone, device).eval()
    style_filter = load_clean_state_dict(StyleFilter_Top(), ckpt_style, device).eval()
    for p in net.parameters():          p.requires_grad = False
    for p in style_filter.parameters(): p.requires_grad = False
    print("[model] MWFormer-real (Network_top + StyleFilter_Top) 로드 완료")
    return style_filter, net


def resize_like_notebook(img: Image.Image) -> Image.Image:
    """노트북 WeatherDataset._resize: 최대변 1024 제한 후 16배수, LANCZOS."""
    wd, ht = img.size
    if ht > wd and ht > 1024:
        wd = int(np.ceil(wd * 1024 / ht)); ht = 1024
    elif ht <= wd and wd > 1024:
        ht = int(np.ceil(ht * 1024 / wd)); wd = 1024
    wd = int(16 * np.ceil(wd / 16.0))
    ht = int(16 * np.ceil(ht / 16.0))
    return img.resize((wd, ht), Image.Resampling.LANCZOS)


@torch.no_grad()
def compute_heatmap(style_filter, net, img_pil: Image.Image, device):
    """
    Returns
    -------
    heatmap_raw : (H, W) float32 [0,1]  raw residual (npy 저장용)
    heatmap_vis : (H, W) float32 [0,1]  per-image min-max (PNG 시각화 전용)
    """
    img_r = resize_like_notebook(img_pil)
    inp   = to_norm(img_r).unsqueeze(0).to(device)   # [-1,1]
    orig  = to_01(img_r).unsqueeze(0).to(device)     # [0,1] (잔차 비교용)

    feature_vec = style_filter(inp)
    restored    = net(inp, feature_vec).clamp(0, 1)  # [0,1]

    residual = (orig - restored).abs().mean(dim=1).squeeze(0)   # [H,W] in [0,1]
    heatmap_raw = residual.cpu().numpy().astype(np.float32)

    h_min, h_max = heatmap_raw.min(), heatmap_raw.max()
    heatmap_vis = ((heatmap_raw - h_min) / (h_max - h_min)
                   if h_max - h_min > 1e-6 else np.zeros_like(heatmap_raw)).astype(np.float32)
    return heatmap_raw, heatmap_vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mwformer_dir",  default="/content/MWFormer",
                   help="taco-group/MWFormer 클론 경로")
    p.add_argument("--ckpt_style",    required=True, help="MWFormer-real style_filter 체크포인트")
    p.add_argument("--ckpt_backbone", required=True, help="MWFormer-real backbone 체크포인트")
    p.add_argument("--input_dir",     required=True, help="원본 이미지 폴더")
    p.add_argument("--out_dir",       default=None,  help="출력 폴더 (기본: input_dir/../heatmaps)")
    p.add_argument("--device",        default="cuda")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    style_filter, net = load_models(args.mwformer_dir, args.ckpt_style,
                                    args.ckpt_backbone, device)

    img_files = sorted(f for f in os.listdir(args.input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작 "
          f"(npy=raw H, W=exp(-alpha*H)는 학습 시 적용)")

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_pil = Image.open(os.path.join(args.input_dir, fname)).convert("RGB")
        heatmap_raw, heatmap_vis = compute_heatmap(style_filter, net, img_pil, device)

        np.save(os.path.join(out_dir, stem + ".npy"), heatmap_raw)
        Image.fromarray((heatmap_vis * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
