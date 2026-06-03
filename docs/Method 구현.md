HGSplat / Method 구현 로그

# Weather-aware Heatmap Loss — 구현 로드맵

---

## 전체 파이프라인

```
[로컬 맥]  코드 작성 → GitHub push
     │
     ▼
[Google Colab L4]
  ├─ Step 0. HGSplat 레포 clone (MWFormer 내장)
  ├─ Step 1. generate_heatmaps.py   → heatmaps/*.npy 생성
  └─ Step 2. train.py (수정)        → weighted photometric loss 적용
```

---

## 파일 구조

```
HGSplat/
├── train.py                   ← [수정] HeatmapWeightedLoss 적용
├── arguments/__init__.py      ← [수정] --heatmap_alpha 파라미터 추가
├── generate_heatmaps.py       ← [신규] MWFormer 기반 heatmap 전처리 스크립트
├── mwformer/                  ← [신규] MWFormer 내장 패키지
│   ├── __init__.py            ←   Network_top, StyleFilter_Top export
│   ├── backbone.py            ←   복원 backbone (EncDec.py 기반)
│   ├── style_filter.py        ←   날씨 style vector 추출 (style_filter64.py 기반)
│   └── base_networks.py       ←   공용 레이어 + strip_prefix_if_present
├── utils/heatmap_loss.py      ← [신규] HeatmapWeightedLoss 클래스
└── IWAIT26_Colab.ipynb        ← Colab 실험 노트북
```

---

## Colab 세팅 순서

### 0. 클론 및 환경 설치
```bash
git clone --recursive https://github.com/Leeyoonho02/HGSplat.git
cd HGSplat
pip install -r requirements.txt
pip install submodules/simple-knn submodules/diff-gaussian-rasterization submodules/fused-ssim
```

### 1. Heatmap 생성
```bash
python generate_heatmaps.py \
    --ckpt_backbone /path/to/backbone.pth \
    --ckpt_style    /path/to/style_filter.pth \
    --scene_dir     data/YOUR_SCENE/images \
    --out_dir       data/YOUR_SCENE/heatmaps \
    --alpha         5.0
```

### 2. 학습
```bash
# Ours: heatmaps/ 폴더 존재 시 자동 활성화
python train.py -s data/YOUR_SCENE -m output/ours --heatmap_alpha 5.0

# Baseline
python train.py -s data/YOUR_SCENE -m output/baseline
```

---

## 코드 수정 상세 로그

### [v1] `arguments/__init__.py` — 파라미터 추가

**위치:** `OptimizationParams.__init__()` 마지막 `super().__init__()` 직전

```python
# [HGSplat] Weather-aware Heatmap Loss 파라미터
# heatmap_dir 는 source_path/heatmaps/ 로 자동 결정 (별도 인자 없음)
# heatmap_alpha: W_t = exp(-alpha * H_t) 의 감쇠 계수
self.heatmap_alpha = 5.0
```

**이유:** `ParamGroup` 자동 파싱으로 `--heatmap_alpha` CLI 인자 등록.
`heatmap_dir`는 `source_path/heatmaps/` 존재 여부로 자동 결정하므로 별도 인자 불필요.

---

### [v1] `train.py` — import 추가

**위치:** line 24, `from utils.loss_utils import ...` 다음 줄

```python
from utils.heatmap_loss import HeatmapWeightedLoss  # [HGSplat]
```

---

### [v1] `train.py` — `HeatmapWeightedLoss` 초기화

**위치:** `training()` 함수 진입 직후, `Scene` 생성 이후

```python
# [HGSplat] source_path/heatmaps/ 존재 시 자동 활성화, 없으면 일반 L1 (Baseline)
heatmap_dir = os.path.join(dataset.source_path, "heatmaps")
heatmap_loss_fn = HeatmapWeightedLoss(
    heatmap_dir=heatmap_dir,
    device=torch.device("cuda"),
    enabled=os.path.isdir(heatmap_dir),
    alpha=opt.heatmap_alpha,
)
```

**이유:** 폴더 존재 여부만으로 Baseline/Ours 자동 전환. `.npy`는 lazy-load로 메모리 효율 유지.

---

### [v1] `train.py` — `Ll1` 교체 (4개 루프)

**기존:**
```python
Ll1 = l1_loss(image, gt_image)
```

**교체 (4곳 동일 패턴):**
```python
# [HGSplat] {루프명}: weather-aware weighted L1
Ll1 = heatmap_loss_fn(image, gt_image, viewpoint_cam.image_name)
```

| 루프 | 역할 |
|------|------|
| Init Optimization | 초기 N 프레임 Gaussian 초기화 |
| Local Optimization | 슬라이딩 윈도우 로컬 정밀화 |
| Global Optimization | 전체 프레임 글로벌 최적화 |
| Refinement | 최종 해상도 후처리 |

**미수정:** Pose Estimation 루프 — `l1_loss(image[:,occ_mask], ...)` 형태로 픽셀이 1D 슬라이싱되어 heatmap 2D 구조와 충돌. 추후 별도 설계 필요.

---

### [v1] `utils/heatmap_loss.py` — 신규

**핵심 동작:**
1. `image_name` → `{heatmap_dir}/{image_name}.npy` lazy-load + `_cache` 캐싱
2. weight map 해상도 불일치 시 `F.interpolate` 자동 정렬
3. weighted L1: `(diff * W).sum() / (W.sum() * C + 1e-8)`
4. `.npy` 없거나 `enabled=False` → 일반 L1 반환

---

### [v1] `generate_heatmaps.py` — 신규 (초기: WeatherEdit 기반)

초기 버전은 WeatherEdit의 `basicsr.models.archs.mwformer_arch.MWFormer` import.  
→ **[v2]에서 MWFormer 공식 레포 기반으로 교체.**

---

### [v2] `mwformer/` 패키지 — MWFormer 내장

**배경:** `generate_heatmaps.py`가 외부 MWFormer 클론(`taco-group/MWFormer`)에 의존하던 것을
레포 내부 패키지로 이동. Colab에서 별도 클론 불필요.

**원본:** [taco-group/MWFormer](https://github.com/taco-group/MWFormer) (IEEE TIP 2024)

| 파일 | 원본 | 변경 내용 |
|------|------|---------|
| `mwformer/backbone.py` | `model/EncDec.py` | import 경로 수정, 불필요 코드 제거 |
| `mwformer/style_filter.py` | `model/style_filter64.py` | import 경로 수정 |
| `mwformer/base_networks.py` | `model/base_networks.py` | `strip_prefix_if_present` 유틸 추가 |
| `mwformer/__init__.py` | 없음 | `Network_top`, `StyleFilter_Top` export |

**MWFormer 인퍼런스 흐름:**
```python
feature_vec = StyleFilter_Top(I_t)          # 64-dim 날씨 style vector
restored    = Network_top(I_t, feature_vec) # 복원 이미지
H_t = channel_mean(|I_t - restored|)        # → min-max 정규화
W_t = exp(-alpha * H_t)                     # weight map
```

---

### [v2] `generate_heatmaps.py` — MWFormer 내장 패키지 사용으로 교체

**변경:**
- `--mwformer_root`, `--ckpt` → `--ckpt_backbone`, `--ckpt_style` (두 체크포인트 분리)
- `from mwformer import Network_top, StyleFilter_Top` 직접 import

---

### [v2] `IWAIT26_Colab.ipynb` — MWFormer 클론 셀 제거

Section 3에서 `git clone taco-group/MWFormer` 셀 삭제.  
`generate_heatmaps.py` 인자도 `--ckpt_backbone` / `--ckpt_style` 로 업데이트.

---

## 하이퍼파라미터

| 파라미터 | 기본값 | CLI | 설명 |
|---------|--------|-----|------|
| `heatmap_dir` | `source_path/heatmaps/` | (자동) | 폴더 존재 여부로 활성화 결정 |
| `heatmap_alpha` | `5.0` | `--heatmap_alpha` | W_t 감쇠 강도 |

---

## 실험 계획

| 실험 | α | 설명 |
|------|---|------|
| Baseline | — | `heatmaps/` 없이 실행, 원본 L1 |
| Ours-weak | 3.0 | α 탐색 |
| **Ours** | **5.0** | **메인** |
| Ours-strong | 10.0 | α 탐색 |

데이터셋: WeatherGS snow / 지표: PSNR · SSIM · LPIPS

---

## TODO

- [x] GitHub 레포 생성 및 LongSplat 통합 push
- [x] `arguments/__init__.py` — `heatmap_alpha` 추가
- [x] `train.py` — import, 초기화, 4개 루프 `Ll1` 교체
- [x] `utils/heatmap_loss.py` 작성
- [x] `generate_heatmaps.py` 작성 (v1: WeatherEdit → v2: MWFormer 공식)
- [x] `mwformer/` 패키지 내장 (backbone, style_filter, base_networks)
- [x] Colab 노트북 작성 및 MWFormer 클론 셀 제거
- [x] README 업데이트 (HGSplat 제목, mwformer 패키지 반영)
- [ ] Colab에서 체크포인트 확인 및 heatmap 생성 테스트
- [ ] Baseline 실험 실행
- [ ] Ours (α=5) 실험 실행
- [ ] 결과 정리 및 비교표 작성
