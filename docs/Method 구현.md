IWAIT'26 / Method 구현

# Weather-aware Heatmap Loss — 구현 로드맵

---

## 전체 파이프라인

```
[로컬 맥]  코드 작성 → GitHub push
     │
     ▼
[Google Colab L4]
  ├─ Step 0. LongSplat + 이 레포 clone
  ├─ Step 1. generate_heatmaps.py   (MWFormer 전처리 → heatmaps/*.npy)
  └─ Step 2. train.py (수정)        (heatmap_loss.py 로드 → weighted loss)
```

**환경 분리 이유:** MWFormer(WeatherEdit)는 PyTorch 1.12, LongSplat은 PyTorch 2.x.  
오프라인 전처리로 `.npy`만 공유 → 환경 충돌 없음.

---

## 파일 구조

```
6_IWAIT'26/
├── generate_heatmaps.py     ← Step 1: MWFormer 전처리 스크립트
├── utils/
│   └── heatmap_loss.py      ← Step 2: LongSplat train.py에 import
└── Method 구현.md
```

---

## Colab 세팅 순서

### 0. 환경 설치
```bash
# LongSplat clone
!git clone --recursive https://github.com/NVlabs/LongSplat.git
%cd LongSplat

# 이 레포 clone (utils/ 등 복사)
!git clone https://github.com/YOUR_ID/IWAIT26-HeatmapLoss.git iwait26
!cp -r iwait26/utils ./utils
!cp iwait26/generate_heatmaps.py .

# WeatherEdit (MWFormer용) — Step 1에서만 필요
!git clone https://github.com/Jumponthemoon/WeatherEdit.git
```

### 1. Heatmap 생성 (Step 1)
```bash
# WeatherEdit 환경에서 실행
!python generate_heatmaps.py \
    --scene_dir data/YOUR_SCENE/images \
    --ckpt      WeatherEdit/General_Scene/pretrained/mwformer.pth \
    --out_dir   data/YOUR_SCENE/heatmaps \
    --alpha     5.0
```

→ `data/YOUR_SCENE/heatmaps/frame_00001.npy` ... 생성

### 2. LongSplat train.py 수정 (Step 2)

`train.py` 상단 import 추가:
```python
from utils.heatmap_loss import HeatmapWeightedLoss
```

학습 루프 초기화 부분에 추가:
```python
heatmap_loss_fn = HeatmapWeightedLoss(
    heatmap_dir=os.path.join(args.source_path, "heatmaps"),
    device=torch.device("cuda"),
    enabled=True,   # False로 바꾸면 Baseline 재현
)
```

기존 photometric loss 부분을 교체:
```python
# ── 기존 ──
# Ll1 = l1_loss(image, gt_image)

# ── 수정 ──
image_name = os.path.splitext(os.path.basename(viewpoint_cam.image_path))[0]
Ll1 = heatmap_loss_fn(image, gt_image, image_name)
```

---

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--alpha` | 5.0 | heatmap → weight 감쇠 강도 (클수록 날씨 픽셀 더 강하게 무시) |
| `enabled` | True | False면 일반 L1 → Baseline 재현 가능 |

---

## 실험 계획

| 실험 | α | enabled | 목적 |
|------|---|---------|------|
| Baseline | — | False | 비교 기준 |
| Ours-weak | 3.0 | True | α 탐색 |
| **Ours** | **5.0** | **True** | **메인** |
| Ours-strong | 10.0 | True | α 탐색 |

**데이터셋:** WeatherGS snow 씬  
**지표:** PSNR / SSIM / LPIPS

---

## TODO

- [ ] GitHub 레포 생성 및 push
- [ ] Colab에서 MWFormer 체크포인트 확인
- [ ] `generate_heatmaps.py` 실행 테스트
- [ ] `train.py` 수정 및 Baseline 실험
- [ ] α=5 메인 실험
- [ ] 결과 정리 및 비교
