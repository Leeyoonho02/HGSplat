# HGSplat: Weather-Aware Heatmap Guided 3D Gaussian Splatting for Robust Reconstruction in Adverse Weather

**이윤호 (Leeyoonho02) / KNUVI** · 목표 학회: IWAIT '26

LongSplat 기반의 Weather-aware Heatmap Loss 구현 레포지토리.
악천후(눈·비·안개) 입력 영상에서 날씨 아티팩트 픽셀의 photometric loss 기여도를 줄여 3D 재구성 강건성을 높이는 것이 목표.

---

## 핵심 아이디어

악천후 픽셀은 *view-inconsistent*(프레임마다 위치가 다름)하므로 3D로 일관되게 재구성될 수 없다. 그러나 LongSplat의 photometric L1 loss는 입력에 박힌 눈송이까지 "정답"으로 맞추려 해 아티팩트를 재구성에 포함시킨다.

→ **MWFormer 복원 잔차**로 각 프레임의 weather heatmap $H_t$ 를 사전 생성하고, weight map $W_t = \exp(-\alpha H_t)$ 를 photometric loss에 곱해 날씨 픽셀을 다운웨이트한다.

$$\mathcal{L}_{photo} = \frac{\sum_p W_t(p) \cdot |I_t(p) - \hat{I}_t(p)|}{\sum_p W_t(p)}$$

- $H_t(p)\approx 0$ (배경) → $W_t\approx 1$ → loss 정상 반영 (디테일 학습)
- $H_t(p)$ 큼 (날씨) → $W_t\approx 0$ → loss 무시 (눈송이 학습 안 함)

> 상세 안내서: [`HGSplat_Method_Guide.md`](../HGSplat_Method_Guide.md) (레포 외부, 프로젝트 루트)

---

## Heatmap 생성 원리 (MWFormer Residual)

```
restored = MWFormer(I)                      # 날씨 제거 복원본
H_t      = mean_channel( |I − restored| )   # 복원 전후 차이 = 날씨 위치
H_t[H_t < thresh] = 0                       # 애매한 저잔차 픽셀 floor (기본 0.1)
```

- **raw H 를 `.npy`로 저장** → weight 변환 $W=\exp(-\alpha H)$ 는 학습 시점에 수행. heatmap 재생성 없이 `--heatmap_alpha`만 바꿔 α ablation 가능.
- 입력은 **[-1,1] 정규화**(Normalize 0.5/0.5) 후 MWFormer에 투입 (학습 도메인과 일치, 정규화 누락 시 복원이 망가짐).

---

## 베이스 코드

- **LongSplat (ICCV 2025)** — Chin-Yang Lin et al., NVIDIA
  원본 라이선스: [LICENSE.md](LICENSE.md), [LICENSE_inria.md](LICENSE_inria.md)
- **MWFormer (IEEE TIP 2024)** — taco-group/MWFormer
  `generate_heatmaps.py`가 공식 repo를 클론해 `model.EncDec`(Network_top) + `model.style_filter64`(StyleFilter_Top)를 사용. 체크포인트는 **MWFormer-real**.

---

## 파일 구조

```
HGSplat/
├── generate_heatmaps.py       ← [신규] MWFormer 복원 잔차로 heatmap 생성 (공식 repo 사용)
├── utils/heatmap_loss.py      ← [신규] HeatmapWeightedLoss (raw H 로드 → W=exp(-αH))
├── train.py                   ← [수정] LongSplat 4개 루프에 weighted L1 적용
│                                  · heatmaps/ 자동 감지로 Baseline/Ours 전환
│                                  · 출력 폴더에 _YYMMDD_HHMMSS 자동 부착
│                                  · terminal_log.txt 로 터미널 출력 전체 저장
├── arguments/__init__.py      ← [수정] --heatmap_alpha 파라미터 추가 (기본 5.0)
├── mwformer/                  ← (레거시) 내장 패키지. 현재 generate_heatmaps.py는 미사용
└── IWAIT26_Colab.ipynb        ← Colab 실험 노트북
```

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
    └── heatmaps/             ← raw heatmap (선택)
        ├── frame_00001.npy   │  존재하면 → Ours (weighted loss) 자동 활성화
        └── ...               │  없으면   → Baseline (일반 L1)
```

- `images/`와 `heatmaps/`의 파일명은 **확장자 제외하고 동일**해야 함
- `heatmaps/*.npy`: shape `[H, W]`, dtype `float32`, raw residual heatmap (값 범위 `[0, 1]`, threshold 미만은 0)

---

## 사용법

### 1. Heatmap 사전 생성 (MWFormer, GPU)

```bash
git clone https://github.com/taco-group/MWFormer /content/MWFormer

python generate_heatmaps.py \
    --mwformer_dir   /content/MWFormer \
    --ckpt_style     /path/to/MWFormer-real/style_filter \
    --ckpt_backbone  /path/to/MWFormer-real/backbone \
    --input_dir      data/YOUR_SCENE/images \
    --out_dir        data/YOUR_SCENE/heatmaps \
    --heatmap_thresh 0.1
```

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--heatmap_thresh` | 0.1 | `residual < thresh` 인 애매한 픽셀을 0으로 floor |

### 2. 학습

```bash
# Ours: heatmaps/ 폴더 존재 시 자동 활성화
python train.py -s data/YOUR_SCENE -m output/ours --mode custom --resolution 2 --heatmap_alpha 5.0

# Baseline: heatmaps/ 폴더를 잠시 숨기고 실행
mv data/YOUR_SCENE/heatmaps data/YOUR_SCENE/heatmaps_bak
python train.py -s data/YOUR_SCENE -m output/baseline --mode custom --resolution 2
mv data/YOUR_SCENE/heatmaps_bak data/YOUR_SCENE/heatmaps
```

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--heatmap_alpha` | 5.0 | $W=\exp(-\alpha H)$ 감쇠 계수 (Ours 전용) |

> - **Baseline/Ours는 `--heatmap_alpha`가 아니라 `source_path/heatmaps/` 폴더 존재 여부로 갈린다.**
> - 출력 폴더는 `_YYMMDD_HHMMSS`가 자동 부착되어 실제로는 `output/ours_260614_HHMMSS/`가 된다. `render.py`/`metrics.py`의 `-m`은 생성된 실제 폴더명을 사용할 것.
> - 결과 폴더에 `outputs.log`(logger) + `terminal_log.txt`(터미널 출력 전체)가 함께 저장된다.

---

## 실험 계획

| 실험 | α | 설명 |
|------|---|------|
| Baseline | — | 원본 LongSplat L1 |
| Ours-weak | 3.0 | α 탐색 |
| **Ours** | **5.0** | **메인** |
| Ours-strong | 10.0 | α 탐색 |

데이터셋: WeatherGS snow / 지표: PSNR · SSIM · LPIPS
