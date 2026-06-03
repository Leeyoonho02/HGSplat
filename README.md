# IWAIT'26 — Weather-aware Heatmap Loss for LongSplat

**이윤호 (Leeyoonho02) / KNUVI**

LongSplat 기반의 Weather-aware Heatmap Loss 구현 레포지토리.  
악천후(눈·비·안개) 입력 영상에서 날씨 아티팩트 픽셀의 photometric loss 기여도를 줄여 3D 재구성 강건성을 높이는 것이 목표.

---

## 핵심 아이디어

MWFormer(WeatherEdit)로 각 프레임의 날씨 영향도 heatmap $H_t$ 를 사전 생성하고,  
LongSplat 학습 시 weight map $W_t = \exp(-\alpha H_t)$ 를 photometric loss에 곱한다.

$$\mathcal{L}_{photo} = \frac{\sum_p W_t(p) \cdot |I_t(p) - \hat{I}_t(p)|}{\sum_p W_t(p)}$$

날씨 픽셀은 낮은 가중치를 받아 모델이 배경 구조에 집중하게 된다.

---

## 베이스 코드

[LongSplat (ICCV 2025)](https://github.com/NVlabs/LongSplat) — Chin-Yang Lin et al., NVIDIA  
원본 라이선스: [LICENSE.md](LICENSE.md), [LICENSE_inria.md](LICENSE_inria.md)

---

## 수정 파일

| 파일 | 내용 |
|------|------|
| `train.py` | HeatmapWeightedLoss 초기화 + 4개 루프 Ll1 교체 |
| `arguments/__init__.py` | `--heatmap_alpha` 파라미터 추가 |
| `utils/heatmap_loss.py` | HeatmapWeightedLoss 클래스 (신규) |
| `generate_heatmaps.py` | MWFormer 기반 heatmap 전처리 스크립트 (신규) |

수정 상세 → [`docs/Method 구현.md`](docs/Method%20구현.md)

---

## 설치

```bash
git clone --recursive https://github.com/Leeyoonho02/HGSplat.git
cd IWAIT_26

conda create -n iwait26 python=3.10.13 cmake=3.14.0 -y
conda activate iwait26
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
    │   ├── frame_00002.png
    │   └── ...
    └── heatmaps/             ← Heatmap weight map (선택)
        ├── frame_00001.npy   │  존재하면 → Ours (weighted loss) 자동 활성화
        ├── frame_00002.npy   │  없으면   → Baseline (일반 L1)
        └── ...
```

- `images/` 파일명과 `heatmaps/` 파일명은 **확장자를 제외하고 동일**해야 함
- `heatmaps/*.npy` 는 `generate_heatmaps.py` 로 생성 (shape: `[H, W]`, dtype: `float32`, 값 범위: `(0, 1]`)

---

## 사용법

### 1. Heatmap 사전 생성 (WeatherEdit/MWFormer 환경)

```bash
python generate_heatmaps.py \
    --scene_dir data/YOUR_SCENE/images \
    --ckpt      /path/to/mwformer.pth \
    --out_dir   data/YOUR_SCENE/heatmaps \
    --alpha     5.0
```

### 2. 학습

```bash
# Ours: data/YOUR_SCENE/heatmaps/ 존재 시 자동 활성화
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
