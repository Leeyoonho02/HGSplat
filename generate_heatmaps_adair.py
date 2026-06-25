"""
generate_heatmaps_adair.py
──────────────────────────
AdaIR (ICLR'25, c-yn/AdaIR) 로 입력을 복원하고 residual 로 weather heatmap 생성.
generate_heatmaps.py (MWFormer 판) 의 AdaIR 교체본 — Exp1 (히트맵 생성용 복원 모델 교체).

  restored = AdaIR(I)
  H_t      = mean(|I - restored|, dim=channel)   ← raw residual heatmap [0,1] (npy 저장)
  (loss 가중치 W_t = exp(-alpha * H_t) 는 학습 시점에 heatmap_loss.py 에서 계산)

★ MWFormer 판과의 핵심 차이 — 입력 정규화 도메인 ([Method_Develop_Log v6] 버그 교훈):
  - MWFormer : Normalize(0.5,0.5,0.5) → [-1, 1]
  - AdaIR    : ToTensor 만, **[0, 1]** (Restormer/PromptIR 계열, mean/std 없음).
    여기서 Normalize 를 넣으면 복원이 망가짐 — 절대 추가 금지.

★ AdaIR 는 desnow 태스크/가중치가 없음. 눈은 derain 과 시각적으로 가장 유사하므로
  all-in-one (mode 6: denoise+derain+dehaze+deblur+enhance) 체크포인트를 snow proxy 로 사용.

체크포인트:
  AdaIR 는 PyTorch-Lightning 으로 저장됨 → state_dict 키가 `net.` prefix.
  여기선 bare `AdaIR(decoder=True)` 에 prefix strip 후 로드 (lightning 의존성 불필요).

사전 조건 (Colab):
    !git clone https://github.com/c-yn/AdaIR /content/AdaIR
    # mode 6 (5-task all-in-one) 체크포인트를 repo README 의 Google Drive 에서 받아 둔다.

사용법:
    python generate_heatmaps_adair.py \\
        --adair_dir  /content/AdaIR                 \\
        --ckpt       /path/to/adair_allinone5.ckpt  \\
        --input_dir  data/snow_scene/images         \\
        --out_dir    data/snow_scene/heatmaps
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# AdaIR 입력은 [0,1] (ToTensor 만). residual 비교용 원본도 동일 → 변환 하나면 충분.
to_01 = ToTensor()


def load_adair(adair_dir, ckpt_path, device):
    """bare AdaIR(decoder=True) 에 Lightning 체크포인트를 prefix strip 후 로드."""
    if adair_dir not in sys.path:
        sys.path.insert(0, adair_dir)
    from net.model import AdaIR

    model = AdaIR(decoder=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Lightning: AdaIRModel.net.* → bare AdaIR.* (그 외 loss/ema 키는 무시)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    if not cleaned:  # 이미 bare 로 저장된 경우 대비
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[model] AdaIR (decoder=True, all-in-one) 로드 완료")
    return model


def resize_pad16(img: Image.Image) -> Image.Image:
    """최대변 1024 제한 후 16배수로 맞춤 (Restormer 계열 UNet 다운샘플 호환), LANCZOS."""
    wd, ht = img.size
    if ht > wd and ht > 1024:
        wd = int(np.ceil(wd * 1024 / ht)); ht = 1024
    elif ht <= wd and wd > 1024:
        ht = int(np.ceil(ht * 1024 / wd)); wd = 1024
    wd = int(16 * np.ceil(wd / 16.0))
    ht = int(16 * np.ceil(ht / 16.0))
    return img.resize((wd, ht), Image.Resampling.LANCZOS)


@torch.no_grad()
def compute_heatmap(model, img_pil: Image.Image, device, thresh=0.0):
    """
    Returns
    -------
    heatmap_raw : (H, W) float32 [0,1]  raw residual (npy 저장용)
    heatmap_vis : (H, W) float32 [0,1]  per-image min-max (PNG 시각화 전용)

    thresh : residual < thresh 인 애매한 픽셀(텍스처 차이·노이즈)은 0 으로 floor.
    """
    img_r = resize_pad16(img_pil)
    inp   = to_01(img_r).unsqueeze(0).to(device)     # [0,1]

    restored = model(inp).clamp(0, 1)                # [0,1]

    residual = (inp - restored).abs().mean(dim=1).squeeze(0)   # [H,W] in [0,1]
    residual[residual < thresh] = 0.0
    heatmap_raw = residual.cpu().numpy().astype(np.float32)

    h_min, h_max = heatmap_raw.min(), heatmap_raw.max()
    heatmap_vis = ((heatmap_raw - h_min) / (h_max - h_min)
                   if h_max - h_min > 1e-6 else np.zeros_like(heatmap_raw)).astype(np.float32)
    return heatmap_raw, heatmap_vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adair_dir", default="/content/AdaIR", help="c-yn/AdaIR 클론 경로")
    p.add_argument("--ckpt",      required=True, help="AdaIR all-in-one(mode6) Lightning 체크포인트")
    p.add_argument("--input_dir", required=True, help="원본 이미지 폴더")
    p.add_argument("--out_dir",   default=None,  help="출력 폴더 (기본: input_dir/../heatmaps)")
    p.add_argument("--heatmap_thresh", type=float, default=0.0,
                   help="residual < thresh 인 픽셀을 0 으로 floor. 기본 0 (floor 안 함) — "
                        "눈 residual 이 ~0.01 로 작아 [v7] 에서 thresh 0.1 은 유해로 판명. "
                        "스케일은 학습 시 heatmap_norm=frame 이 담당.")
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input_dir), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = load_adair(args.adair_dir, args.ckpt, device)

    img_files = sorted(f for f in os.listdir(args.input_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"[info] {len(img_files)}개 이미지 처리 시작 "
          f"(npy=raw H, W=exp(-alpha*H)는 학습 시 적용)")

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_pil = Image.open(os.path.join(args.input_dir, fname)).convert("RGB")
        heatmap_raw, heatmap_vis = compute_heatmap(model, img_pil, device,
                                                   thresh=args.heatmap_thresh)

        np.save(os.path.join(out_dir, stem + ".npy"), heatmap_raw)
        Image.fromarray((heatmap_vis * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, stem + ".png"))

    print(f"[done] {len(img_files)}개 heatmap 저장 완료 → {out_dir}")


if __name__ == "__main__":
    main()
