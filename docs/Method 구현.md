IWAIT'26 / Method 구현

# Weather-aware Heatmap Loss — 구현 로드맵

---

## 전체 파이프라인

```
[로컬 맥]  코드 작성 → GitHub push
     │
     ▼
[Google Colab L4]
  ├─ Step 0. LongSplat(IWAIT_26 레포) clone
  ├─ Step 1. generate_heatmaps.py   (MWFormer 전처리 → heatmaps/*.npy)
  └─ Step 2. train.py (수정)        (heatmap_loss.py 로드 → weighted loss)
```

**환경 분리 이유:** MWFormer(WeatherEdit)는 PyTorch 1.12, LongSplat은 PyTorch 2.x.
오프라인 전처리로 `.npy`만 공유 → 환경 충돌 없음.

---

## 파일 구조

```
IWAIT_26/                          ← 이 레포 (LongSplat 기반)
├── train.py                       ← [수정] weighted loss 적용
├── arguments/__init__.py          ← [수정] --heatmap_dir, --heatmap_alpha 추가
├── generate_heatmaps.py           ← [신규] MWFormer 전처리 스크립트
├── utils/
│   ├── heatmap_loss.py            ← [신규] HeatmapWeightedLoss 클래스
│   └── ... (LongSplat 기존 파일)
├── submodules/                    ← diff-gs, simple-knn, fused-ssim, mast3r
└── docs/
    ├── Method 구현.md             ← 이 파일
    ├── Extention_Idea_List.md
    └── 아이디어_논의_1.md
```

---

## Colab 세팅 순서

### 0. 클론 및 환경 설치
```bash
!git clone --recursive https://github.com/Leeyoonho02/IWAIT_26.git
%cd IWAIT_26
# 이후 requirements.txt 및 submodule pip install
```

### 1. Heatmap 생성 (WeatherEdit 환경에서)
```bash
python generate_heatmaps.py \
    --scene_dir data/YOUR_SCENE/images \
    --ckpt      /path/to/mwformer.pth \
    --out_dir   data/YOUR_SCENE/heatmaps \
    --alpha     5.0
```
→ `data/YOUR_SCENE/heatmaps/프레임이름.npy` 생성

### 2. 학습 실행
```bash
# Ours (heatmap loss 적용)
python train.py \
    -s data/YOUR_SCENE \
    -m output/ours_alpha5 \
    --heatmap_dir data/YOUR_SCENE/heatmaps \
    --heatmap_alpha 5.0

# Baseline (heatmap_dir 미지정 → 일반 L1)
python train.py \
    -s data/YOUR_SCENE \
    -m output/baseline
```

---

## 코드 수정 상세 로그

### 1. `arguments/__init__.py` — 파라미터 추가

**위치:** `OptimizationParams.__init__()` 마지막 `super().__init__()` 직전

**변경 내용:**
```python
# [IWAIT'26] Weather-aware Heatmap Loss 파라미터
# heatmap_dir: generate_heatmaps.py 로 생성한 .npy 폴더 경로
#              빈 문자열이면 heatmap loss 비활성화 → Baseline 재현 가능
# heatmap_alpha: W_t = exp(-alpha * H_t) 의 감쇠 계수
self.heatmap_dir = ""
self.heatmap_alpha = 5.0
```

**이유:** `ParamGroup`의 자동 파싱 구조를 활용해 CLI 인자(`--heatmap_dir`, `--heatmap_alpha`)로
자동 등록. `heatmap_dir=""` 기본값으로 Baseline/Ours를 동일 코드에서 전환 가능.

---

### 2. `train.py` — import 추가

**위치:** line 24 (기존 `from utils.loss_utils import ...` 바로 다음 줄)

```python
from utils.heatmap_loss import HeatmapWeightedLoss  # [IWAIT'26] Weather-aware Heatmap Loss
```

---

### 3. `train.py` — `HeatmapWeightedLoss` 초기화

**위치:** `training()` 함수 진입 직후, `Scene` 생성 이후

```python
# [IWAIT'26] HeatmapWeightedLoss 초기화
# opt.heatmap_dir 가 비어 있으면 enabled=False → 기존 l1_loss 와 동일하게 동작 (Baseline)
# opt.heatmap_dir 가 지정되면 해당 폴더의 .npy 를 로드해 weighted loss 적용
heatmap_loss_fn = HeatmapWeightedLoss(
    heatmap_dir=opt.heatmap_dir,
    device=torch.device("cuda"),
    enabled=bool(opt.heatmap_dir),
    alpha=opt.heatmap_alpha,
)
```

**이유:** 학습 전체에서 단 한 번만 초기화. `.npy`는 `_cache` dict로 lazy-load되어
메모리 효율 유지 (처음 접근 시에만 디스크 읽기).

---

### 4. `train.py` — `Ll1` 교체 (4개 루프)

**기존 코드:**
```python
Ll1 = l1_loss(image, gt_image)
```

**교체된 코드 (4곳 모두 동일한 패턴):**
```python
# [IWAIT'26] {루프명} 루프: weather-aware weighted L1 loss
# heatmap_dir 미지정 시 내부적으로 일반 l1_loss 와 동일하게 동작
Ll1 = heatmap_loss_fn(image, gt_image, viewpoint_cam.image_name)
```

| 루프 | 역할 |
|------|------|
| Init Optimization | 초기 N 프레임으로 Gaussian 초기화 |
| Local Optimization | 슬라이딩 윈도우 내 로컬 정밀화 |
| Global Optimization | 전체 프레임 대상 글로벌 최적화 |
| Refinement | 최종 해상도 후처리 |

**미수정 루프 — Pose Estimation:**
```python
Ll1 = l1_loss(image[:,occ_mask], gt_image[:,occ_mask])
```
occlusion mask로 픽셀을 1D로 슬라이싱하기 때문에 heatmap의 2D 공간 구조와 충돌.
Pose 추정은 카메라 파라미터 최적화가 목적이므로 날씨 가중치 적용 효과가 제한적.
→ **추후 별도 설계 필요 (현재 원본 유지)**

---

### 5. `utils/heatmap_loss.py` — 신규 파일

**핵심 동작:**
1. `image_name`(확장자 없는 파일명)으로 `{heatmap_dir}/{image_name}.npy` 로드
2. `_cache` dict로 캐싱 (같은 프레임 반복 로드 방지)
3. weight map 해상도 ≠ 렌더 이미지 해상도일 경우 `F.interpolate` 자동 정렬
4. weighted L1: `(diff * W).sum() / (W.sum() * C + 1e-8)`
5. `enabled=False` 또는 `.npy` 미존재 시 → 일반 평균 L1 반환 (Baseline 동작)

---

### 6. `generate_heatmaps.py` — 신규 파일

WeatherEdit(MWFormer) 환경에서 별도 실행.

**처리 순서:**
1. MWFormer 체크포인트 로드 (freeze, eval)
2. 각 프레임에 대해: `residual = |I_t - MWFormer(I_t)|`
3. 채널 평균 → min-max 정규화 → `W_t = exp(-alpha * H_t)`
4. `{out_dir}/{stem}.npy` 로 저장

---

## 하이퍼파라미터

| 파라미터 | 기본값 | CLI | 설명 |
|---------|--------|-----|------|
| `heatmap_dir` | `""` | `--heatmap_dir` | 비어있으면 Baseline (일반 L1) |
| `heatmap_alpha` | `5.0` | `--heatmap_alpha` | W_t = exp(-α·H_t) 감쇠 강도 |

---

## 실험 계획

| 실험 | α | 설명 |
|------|---|------|
| Baseline | — | `--heatmap_dir` 미지정, 원본 L1 |
| Ours-weak | 3.0 | α 탐색 |
| **Ours** | **5.0** | **메인 실험** |
| Ours-strong | 10.0 | α 탐색 |

**데이터셋:** WeatherGS snow 씬
**지표:** PSNR / SSIM / LPIPS

---

## TODO

- [x] GitHub 레포 생성 및 LongSplat 통합 push
- [x] `arguments/__init__.py` — `heatmap_dir`, `heatmap_alpha` 추가
- [x] `train.py` — import, 초기화, 4개 루프 `Ll1` 교체
- [x] `utils/heatmap_loss.py` 작성
- [x] `generate_heatmaps.py` 작성
- [ ] Colab에서 MWFormer 체크포인트 경로 확인 및 heatmap 생성 테스트
- [ ] Baseline 실험 실행
- [ ] Ours (α=5) 실험 실행
- [ ] 결과 정리 및 비교표 작성
