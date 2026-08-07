# GSplat: Weather-Aware Heatmap-Weighted 3D Gaussian Splatting from Unposed Videos

LongSplat 기반의 **unposed 3D Gaussian Splatting(3DGS)** 연구 코드입니다. 이 프로젝트는 날씨 제거 모델을 새로 제안하는 것이 아니라, **불완전한 2D 날씨 복원이 선행된 영상으로부터 3D 장면을 재구성할 때 그 복원 결과를 얼마나 신뢰할지 조절하는 방법**을 다룹니다.

현재 주 실험 대상은 눈이 포함된 casual video이며, MWFormer가 처리 가능한 비 입력으로의 확장은 후속 과제입니다.

> 연구 프레이밍: _2D restoration의 한계에 대응하는 unposed 3DGS loss 개선_

---

## 어떤 환경의 문제인가?

폭설·폭우 같은 악천후에서 드론이나 휴대 장비로 촬영한 영상은 다음 두 조건을 동시에 가질 수 있습니다.

- **악천후 오염:** 눈·비 같은 일시적 artifact가 관측을 가리고, frame마다 다른 위치에 나타납니다.
- **unposed 입력:** 카메라 pose가 사전에 주어지지 않습니다. LongSplat은 영상에서 pose와 3D Gaussian scene을 함께 최적화합니다.

예를 들어 재난 현장에서 고정 CCTV 없이 얻은 드론 영상으로 산악·계곡의 geometry를 빠르게 파악해야 하는 상황을 생각할 수 있습니다. 이때 날씨 artifact는 appearance뿐 아니라 pose 추정과 geometry 최적화에도 잘못된 photometric signal을 줄 수 있습니다.

## 왜 단순 2D 복원만으로는 충분하지 않은가?

MWFormer 같은 2D restoration 모델로 눈/비를 줄일 수 있지만, 복원 출력은 완전한 clean ground truth가 아닙니다.

- artifact가 일부 남을 수 있습니다.
- 실제 texture나 경계를 artifact로 오인해 바꿀 수 있습니다.
- 이렇게 바뀐 이미지를 3DGS의 photometric target으로 그대로 신뢰하면, 잘못된 색·texture·correspondence가 reconstruction과 pose 최적화에 전파될 수 있습니다.

따라서 HGSplat의 목표는 **복원본을 새 정답으로 강제하는 것**이 아니라, 복원 모델이 크게 개입했거나 reconstruction 중 불안정하게 남는 pixel의 supervision을 부드럽게 낮추어 LongSplat을 더 견고하게 만드는 것입니다.

---

## 입력 → 방법 → 출력

| 단계                      | 입력                                       | HGSplat 처리                                                                          | 산출물                                          |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1. 사전 복원              | weather-corrupted unposed frames `images/` | 고정된 MWFormer로 frame별 복원                                                        | `images_cleaned/`                               |
| 2. 초기 불확실성          | 원본 $I^{weather}$, 복원본 $I^{clean}$     | 두 이미지 차이로 복원 모델이 크게 바꾼 영역을 $H^{init}$로 표현                       | `heatmap_init/*.npy`                            |
| 3. Unposed reconstruction | cleaned frames, $H^{init}$                 | LongSplat이 pose와 Gaussian scene을 공동 최적화. pixel별 weighted photometric L1 적용 | 학습된 Gaussian representation 및 camera pose   |
| 4. 동적 신뢰도 보정       | Refinement 중 render residual              | view별 residual EMA와 분산 gate로 $H^{dyn}$을 만들고 $H^{init}$과 점진적으로 결합     | 최종 render, novel-view images, PSNR/SSIM/LPIPS |

최종 출력물은 날씨가 덜 포함된 3D Gaussian scene, 해당 scene의 novel-view render, 그리고 unposed 입력에서 추정된 camera pose입니다. 현재 코드는 `render.py`, `metrics.py`를 통해 render와 image-quality metric을 생성합니다. **Pose 품질 평가는 별도 후속 검증 항목**입니다.

## 제안 방법: loss만 바꾼다

HGSplat은 LongSplat의 pose 추정, Gaussian representation, Init/Local/Global/Refinement stage를 새로 설계하지 않습니다. **기존 LongSplat의 pixel-wise photometric L1 항만** heatmap으로 재가중합니다. DSSIM, depth, 2D correspondence, geometry regularization 항은 유지합니다.

### 1. 초기 복원 불확실성

원본과 복원본의 channel-average absolute difference로 초기 heatmap을 만듭니다.

$$
\widetilde H_i^{init}(p)=\frac{1}{3}\sum_{c=1}^{3}
\left|I_{i,c}^{weather}(p)-I_{i,c}^{clean}(p)\right|.
$$

이는 눈의 정확한 segmentation mask가 아니라, **복원 모델이 변경한 영역의 불확실성 proxy**입니다. 이 신호는 Init, Local, Global stage에서 사용됩니다.

### 2. Refinement의 동적 residual 신호

Refinement에서는 현재 render와 cleaned target의 residual을 view별 EMA로 추적합니다. v3에서는 평균 residual에 표준편차 gate를 곱합니다.

$$
H_{i,k}^{dyn}=\mathcal{N}(m_{i,k})\;g(\sigma_{i,k}),
$$

여기서 $m$과 $\sigma$는 각각 channel-average render residual의 1차·2차 모멘트 EMA에서 얻은 평균과 표준편차입니다. 초기 prior와 동적 신호는 ramp로 결합합니다.

$$
H_{i,k}=(1-\lambda_k)H_i^{init}+\lambda_k H_{i,k}^{dyn}.
$$

### 3. Heatmap-weighted photometric L1

최종 heatmap이 큰 pixel일수록 photometric supervision을 낮춥니다.

$$
W_{i,k}(p)=\exp(-\alpha H_{i,k}(p)),
$$

$$
\mathcal{L}_{wL1}=
\frac{\sum_{p,c}W_{i,k}(p)\left|\hat I_{i,k,c}(p)-I_{i,c}^{clean}(p)\right|}
{3\sum_p W_{i,k}(p)+\varepsilon}.
$$

큰 heatmap 값은 해당 pixel을 완전히 지우는 binary mask가 아니라, 그 pixel의 gradient 기여도를 낮추는 **soft confidence weight**입니다.

> 주의: $H^{dyn}$은 동일 3D point를 view 간 직접 대응시키는 신호가 아닙니다. 이는 shared multi-view optimization 과정에서 얻는 **간접적인 residual-stability cue**이며, explicit cross-view correspondence로 주장하지 않습니다.

---

## 실험 구분

| Run    | LongSplat 입력      | Heatmap                   | 확인하려는 것                        |
| ------ | ------------------- | ------------------------- | ------------------------------------ |
| `og`   | 원본 weather frames | 없음                      | 원본 LongSplat baseline              |
| `og_c` | `images_cleaned`    | 없음                      | 2D 복원 입력 자체의 기여             |
| `v1`   | `images_cleaned`    | $H^{init}$                | 초기 복원 불확실성의 효과            |
| `v2`   | `images_cleaned`    | $H^{init}$ + residual EMA | 동적 residual 평균의 효과            |
| `v3`   | `images_cleaned`    | v2 + variance gate        | 불안정한 residual의 과도한 억제 완화 |

현재 기본 설정은 v3, `alpha=30`, percentile 99, heatmap floor 0.05, EMA beta 0.9, residual-std floor 0.02, ramp 10,000입니다. 최신 정량 결과와 한계는 [`../docs/v3_experiment_result.md`](../docs/v3_experiment_result.md), 방법의 전체 정의는 [`../docs/v3_method.md`](../docs/v3_method.md)를 참고하세요.

---

## 코드 구조

```text
code/
├── restoration.py          # MWFormer 복원 → images_cleaned/ + heatmap_init/
├── run_hgsplat.py          # restoration → train → render → metrics 실행기
├── train.py                # LongSplat; weighted L1이 적용된 학습 루프
├── utils/heatmap_loss.py   # H_init, residual EMA, variance gate 구현
├── render.py               # 학습 모델 render
├── metrics.py              # PSNR / SSIM / LPIPS 계산
└── submodules/             # LongSplat 의존 모듈
```

## 설치

LongSplat 및 CUDA 환경 요구사항은 원본 프로젝트 설정을 따릅니다. 아래는 예시입니다.

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

MWFormer는 내장 레거시 모듈이 아니라 [공식 MWFormer repository](https://github.com/taco-group/MWFormer)와 호환 체크포인트를 사용해야 합니다.

## 데이터와 실행

### 입력 데이터 구조

```text
data/
└── YOUR_SCENE/
    └── images/                 # weather-corrupted unposed input frames
```

### 1. 사전 복원 및 초기 heatmap 생성

```bash
python restoration.py \
  --input_dir data/YOUR_SCENE/images \
  --mwformer_dir /path/to/MWFormer \
  --ckpt_style /path/to/MWFormer-real/style_filter \
  --ckpt_backbone /path/to/MWFormer-real/backbone
```

이 단계는 다음을 생성합니다.

```text
data/YOUR_SCENE/
├── images/                     # 원본 입력
├── images_cleaned/             # LongSplat 입력 및 photometric target
└── heatmap_init/               # raw residual .npy와 시각화 .png
```

### 2. v3 학습·render·metric

`run_hgsplat.py`는 `restoration → train → render → metrics`를 순서대로 수행합니다. 위에서 복원을 마쳤다면 `--skip_restore`를 사용하세요.

```bash
python run_hgsplat.py \
  -s data/YOUR_SCENE \
  -m outputs/YOUR_SCENE_v3 \
  --eval --mode custom \
  --skip_restore \
  --heatmap_alpha 30 \
  --heatmap_norm frame --heatmap_pct 99 --heatmap_floor 0.05 \
  --heatmap_mv --heatmap_mv_beta 0.9 --heatmap_mv_ramp 10000 \
  --heatmap_mv_var --heatmap_mv_std_floor 0.02
```

`run_hgsplat.py`가 timestamp를 한 번만 부여하고 학습·render·metric에 동일한 model path를 전달합니다. 기존 output을 다시 render/metric할 때는 `--skip_train`을 사용하고, 이미 존재하는 정확한 output path를 `-m`에 넣으세요.

### 대조군 실행

```bash
# og_c: cleaned input만 사용, heatmap loss는 명시적으로 끔
python run_hgsplat.py -s data/YOUR_SCENE -m outputs/YOUR_SCENE_ogc \
  --eval --mode custom --skip_restore --no_heatmap

# v1: H_init만 사용
python run_hgsplat.py -s data/YOUR_SCENE -m outputs/YOUR_SCENE_v1 \
  --eval --mode custom --skip_restore --heatmap_alpha 30
```

## 범위와 한계

- HGSplat은 2D restoration model이나 weather segmentation model을 제안하지 않습니다.
- $H^{init}$은 실제 artifact mask가 아니라 restoration residual이므로 실제 texture 변화도 포함할 수 있습니다.
- 높은 residual variance는 날씨 외에도 pose 오류, occlusion, reflection, 얇은 구조에서 생길 수 있습니다.
- 현재 image-quality 평가는 snow-free reference를 우선 사용합니다. unposed 입력의 pose 변화가 결과에 영향을 줄 수 있어 pose 평가와 반복 실행 평균±표준편차 검증이 필요합니다.

## 기반 연구

- LongSplat (ICCV 2025), Chin-Yang Lin et al.
- MWFormer (IEEE TIP 2024), taco-group/MWFormer

LongSplat의 원본 라이선스는 [LICENSE.md](LICENSE.md) 및 [LICENSE_inria.md](LICENSE_inria.md)를 참고하세요.
