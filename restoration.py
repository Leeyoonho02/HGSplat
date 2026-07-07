"""
restoration.py
──────────────
[HGSplat v2 파이프라인 Step 1] LongSplat 학습에 앞서 복원 모델(MWFormer)을 실행.

입력 디렉토리(눈/비 낀 원본 images/)를 받아 두 가지를 생성한다:

  1. images_cleaned/  : MWFormer 복원 이미지 (원본 해상도로 역리사이즈, 파일명 동일)
                        → LongSplat 학습 GT 로 사용 (train.py --images images_cleaned)
  2. heatmap_init/    : H = mean(|I - restored|) raw residual (.npy) + 시각화(.png)
                        → v1 가중치 W=exp(-alpha*H) 용 (train.py --heatmap_dir .../heatmap_init)

전처리/로딩은 generate_heatmaps.py 와 동일 (MWFormer_Colab_v4_real.ipynb 기준):
  - 입력 정규화: ToTensor + Normalize(0.5,0.5,0.5) → [-1, 1]   (빠지면 복원이 망가짐)
  - 리사이즈   : 최대변 1024 제한 후 16배수, LANCZOS
  - 체크포인트 : MWFormer-real, module. prefix strip 후 strict=True
  - residual   : [0,1] 원본 vs restored[0,1] 로 계산
  - thresh 기본 0.0  ([v7] 교훈: 눈 잔차 ~0.01 이라 0.1 floor 는 유해)

기본값은 레포에 내장된 mwformer/ 패키지와 mwformer/weights/ 체크포인트를 사용하므로
인자 없이 --input_dir 만 주면 동작한다.

사용법:
    python restoration.py --input_dir data/grass_snow/images
    # → data/grass_snow/images_cleaned/, data/grass_snow/heatmap_init/ 생성
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 모델 입력은 [-1,1] 정규화, 잔차 비교용 원본은 [0,1]
to_norm = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
to_01 = ToTensor()


def load_clean_state_dict(model, ckpt_path, device):
    """module. prefix strip 후 strict=True 로드 (generate_heatmaps.py 와 동일)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model.to(device)


def load_models(ckpt_style, ckpt_backbone, device):
    """내장 mwformer 패키지에서 모델 로드."""
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    from mwformer import Network_top, StyleFilter_Top

    net = load_clean_state_dict(Network_top(), ckpt_backbone, device).eval()
    style_filter = load_clean_state_dict(StyleFilter_Top(), ckpt_style, device).eval()
    for p in net.parameters():
        p.requires_grad = False
    for p in style_filter.parameters():
        p.requires_grad = False
    print("[model] MWFormer-real (Network_top + StyleFilter_Top) 로드 완료")
    return style_filter, net


def resize_like_notebook(img: Image.Image) -> Image.Image:
    """최대변 1024 제한 후 16배수, LANCZOS (generate_heatmaps.py 와 동일)."""
    wd, ht = img.size
    if ht > wd and ht > 1024:
        wd = int(np.ceil(wd * 1024 / ht)); ht = 1024
    elif ht <= wd and wd > 1024:
        ht = int(np.ceil(ht * 1024 / wd)); wd = 1024
    wd = int(16 * np.ceil(wd / 16.0))
    ht = int(16 * np.ceil(ht / 16.0))
    return img.resize((wd, ht), Image.Resampling.LANCZOS)


@torch.no_grad()
def restore_one(style_filter, net, img_pil: Image.Image, device, thresh: float = 0.0):
    """
    Returns
    -------
    restored_pil : PIL.Image  복원 이미지, **원본 해상도**로 역리사이즈됨
    heatmap_raw  : (H, W) float32 [0,1]  raw residual (MWFormer 처리 해상도, npy 저장용)
    heatmap_vis  : (H, W) float32 [0,1]  per-image min-max (PNG 시각화 전용)
    """
    orig_size = img_pil.size  # (W0, H0)
    img_r = resize_like_notebook(img_pil)
    inp = to_norm(img_r).unsqueeze(0).to(device)   # [-1,1]
    orig = to_01(img_r).unsqueeze(0).to(device)    # [0,1] (잔차 비교용)

    feature_vec = style_filter(inp)
    restored = net(inp, feature_vec).clamp(0, 1)   # [1,3,H,W] in [0,1]

    residual = (orig - restored).abs().mean(dim=1).squeeze(0)  # [H,W] in [0,1]
    if thresh > 0:
        residual[residual < thresh] = 0.0
    heatmap_raw = residual.cpu().numpy().astype(np.float32)

    h_min, h_max = heatmap_raw.min(), heatmap_raw.max()
    heatmap_vis = ((heatmap_raw - h_min) / (h_max - h_min)
                   if h_max - h_min > 1e-6 else np.zeros_like(heatmap_raw)).astype(np.float32)

    restored_np = (restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round()
    restored_pil = Image.fromarray(restored_np.clip(0, 255).astype(np.uint8))
    if restored_pil.size != orig_size:
        # 학습 GT 로 쓰기 위해 원본 해상도로 역리사이즈 (images/ 와 drop-in 호환)
        restored_pil = restored_pil.resize(orig_size, Image.Resampling.LANCZOS)

    return restored_pil, heatmap_raw, heatmap_vis


def main():
    p = argparse.ArgumentParser(description="HGSplat v2 Step 1: MWFormer 복원 + heatmap_init 생성")
    p.add_argument("--input_dir", required=True, help="원본 이미지 폴더 (예: data/grass_snow/images)")
    p.add_argument("--cleaned_dir", default=None,
                   help="복원 이미지 출력 폴더 (기본: input_dir/../images_cleaned)")
    p.add_argument("--heatmap_dir", default=None,
                   help="heatmap 출력 폴더 (기본: input_dir/../heatmap_init)")
    p.add_argument("--ckpt_style", default=os.path.join(CODE_DIR, "mwformer", "weights", "style_filter"),
                   help="MWFormer-real style_filter 체크포인트 (기본: 내장 weights)")
    p.add_argument("--ckpt_backbone", default=os.path.join(CODE_DIR, "mwformer", "weights", "backbone"),
                   help="MWFormer-real backbone 체크포인트 (기본: 내장 weights)")
    p.add_argument("--heatmap_thresh", type=float, default=0.0,
                   help="residual < thresh 픽셀을 0 으로 floor (기본 0.0 — [v7] 교훈)")
    p.add_argument("--force", action="store_true", help="기존 출력이 있어도 재생성")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    scene_dir = os.path.dirname(input_dir)
    cleaned_dir = os.path.abspath(args.cleaned_dir or os.path.join(scene_dir, "images_cleaned"))
    heatmap_dir = os.path.abspath(args.heatmap_dir or os.path.join(scene_dir, "heatmap_init"))

    img_files = sorted(f for f in os.listdir(input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not img_files:
        sys.exit(f"[error] 입력 폴더에 이미지가 없습니다: {input_dir}")

    # 파일 단위 스킵(재실행/중단 복구 안전). --force 면 전부 재생성.
    def _done(fname):
        stem = os.path.splitext(fname)[0]
        return (os.path.exists(os.path.join(cleaned_dir, fname))
                and os.path.exists(os.path.join(heatmap_dir, stem + ".npy")))

    todo = img_files if args.force else [f for f in img_files if not _done(f)]
    print(f"[info] 입력 {len(img_files)}장 / 처리 대상 {len(todo)}장  "
          f"(cleaned={cleaned_dir}, heatmap={heatmap_dir}, thresh={args.heatmap_thresh})")
    if not todo:
        print("[skip] 모든 출력이 이미 존재합니다. --force 로 재생성할 수 있습니다.")
        return

    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    style_filter, net = load_models(args.ckpt_style, args.ckpt_backbone, device)

    for i, fname in enumerate(todo, 1):
        stem = os.path.splitext(fname)[0]
        img_pil = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        restored_pil, heatmap_raw, heatmap_vis = restore_one(
            style_filter, net, img_pil, device, thresh=args.heatmap_thresh)

        restored_pil.save(os.path.join(cleaned_dir, fname))
        np.save(os.path.join(heatmap_dir, stem + ".npy"), heatmap_raw)
        Image.fromarray((heatmap_vis * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(heatmap_dir, stem + ".png"))

        if i % 10 == 0 or i == len(todo):
            print(f"  [{i:>4}/{len(todo)}] {fname}  "
                  f"H max={heatmap_raw.max():.4f} mean={heatmap_raw.mean():.4f}")

    print(f"[done] {len(todo)}장 처리 완료\n"
          f"  images_cleaned → {cleaned_dir}\n"
          f"  heatmap_init   → {heatmap_dir}")


if __name__ == "__main__":
    main()
