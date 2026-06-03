# HGSplat: Weather-Aware Heatmap Guided 3D Gaussian Splatting for Robust Reconstruction in Adverse Weather

**이윤호 (Leeyoonho02) / KNUVI**

LongSplat 기반의 Weather-aware Heatmap Loss 구현 레포지토리.  
악천후(눈·비·안개) 입력 영상에서 날씨 아티팩트 픽셀의 photometric loss 기여도를 줄여 3D 재구성 강건성을 높이는 것이 목표.

---

## 핵심 아이디어

MWFormer로 각 프레임의 날씨 영향도 heatmap $H_t$ 를 사전 생성하고,  
LongSplat 학습 시 weight map $W_t = \exp(-\alpha H_t)$ 를 photometric loss에 곱한다.

$$\mathcal{L}_{photo} = \frac{\sum_p W_t(p) \cdot |I_t(p) - \hat{I}_t(p)|}{\sum_p W_t(p)}$$

날씨 픽셀은 낮은 가중치를 받아 모델이 배경 구조에 집중하게 된다.

---

## 베이스 코드

- **LongSplat (ICCV 2025)** — Chin-Yang Lin et al., NVIDIA  
  원본 라이선스: [LICENSE.md](LICENSE.md), [LICENSE_inria.md](LICENSE_inria.md)
- **MWFormer (IEEE TIP 2024)** — taco-group  
  `mwformer/` 패키지로 내장 (별도 클론 불필요)

---

## 파일 구조

```
HGSplat/
├── train.py                   ← [수정] HeatmapWeightedLoss 적용
├── arguments/__init__.py      ← [수정] --heatmap_alpha 파라미터 추가
├── generate_heatmaps.py       ← [신규] MWFormer 기반 heatmap 전처리 스크립트
├── mwformer/                  ← [신규] MWFormer 내장 패키지
│   ├── __init__.py
│   ├── backbone.py            ←   Network_top (복원 backbone)
│   ├── style_filter.py        ←   StyleFilter_Top (날씨 style vector 추출)
│   └── base_networks.py       ←   공용 레이어
├── utils/heatmap_loss.py      ← [신규] HeatmapWeightedLoss 클래스
├── IWAIT26_Colab.ipynb        ← Colab 실험 노트북
└── docs/Method 구현.md        ← 구현 상세 로그
```

수정 상세 → [`docs/Method 구현.md`](docs/Method%20구현.md)

---

## 설치

```bash
git clone --recursive https://github.com/Leeyoonho02/HGSplat.git
cd HGSplat

conda create -n hgsplat python=3.10.13 cmake=3.14.0 -y
conda activate hgsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim
```

---

## 데이터셋 디렉토리 구조

```
data/
└── YOUR_SCENE/
    ├── images/               ← 입력 프레임 (필수)
    │   ├── frame_00001.png
    │   └── ...
    └── heatmaps/             ← Weight map (선택)
        ├── frame_00001.npy   │  존재하면 → Ours (weighted loss) 자동 활성화
        └── ...               │  없으면   → Baseline (일반 L1)
```

- `images/`와 `heatmaps/`의 파일명은 **확장자 제외하고 동일**해야 함
- `heatmaps/*.npy`: shape `[H, W]`, dtype `float32`, 값 범위 `(0, 1]`

---

## 사용법

### 1. Heatmap 사전 생성

MWFormer `StyleFilter` 체크포인트 하나만 필요 (`style_filter.pth`).  
복원 backbone(Network_top) 없이 **인코더 feature map**에서 직접 공간 heatmap 생성.

```bash
python generate_heatmaps.py \
    --ckpt_style  /path/to/style_filter.pth \
    --scene_dir   data/YOUR_SCENE/images \
    --out_dir     data/YOUR_SCENE/heatmaps \
    --alpha       5.0
```

### 2. 학습

```bash
# Ours: heatmaps/ 폴더 존재 시 자동 활성화
python train.py -s data/YOUR_SCENE -m output/ours --heatmap_alpha 5.0

# Baseline: heatmaps/ 폴더 없이 실행
python train.py -s data/YOUR_SCENE -m output/baseline
```

---

## 실험 계획

| 실험 | α | 설명 |
|------|---|------|
| Baseline | — | 원본 LongSplat L1 |
| Ours-weak | 3.0 | α 탐색 |
| **Ours** | **5.0** | **메인** |
| Ours-strong | 10.0 | α 탐색 |

데이터셋: WeatherGS snow / 지표: PSNR · SSIM · LPIPS
