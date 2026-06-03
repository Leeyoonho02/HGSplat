HGSplat / Method 구현 로그

# Weather-aware Heatmap Loss — 구현 로드맵

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

### [v3] `mwformer/style_filter.py` — StyleEncoder 버그 수정 + `encode_spatial()` 추가

**버그:** `StyleEncoder.forward()`에서 stage 1 feature `x1` ([B, 64, H/4, W/4])를
`patch_embed2` 호출이 덮어씌워 `return [x2, x1]`이 둘 다 128-dim을 반환하고 있었음.
→ Gram matrix 크기가 `StyleFilter_conv1(2080)` 기대값과 불일치 → StyleFilter 자체가 오작동.

**수정:** `x1_s1 = x1` 으로 stage 1 feature를 저장한 뒤 `return [x1_s1, x1]` 로 교체.

**`encode_spatial()` 추가:** 체크포인트 1개(StyleFilter)만으로 공간 heatmap 생성.

```python
def encode_spatial(self, x):
    enc_out = self.encoder(x)
    # enc_out[0]: [1, 64,  H/4, W/4]  stage 1
    # enc_out[1]: [1, 128, H/8, W/8]  stage 2

    h1 = enc_out[0].norm(dim=1, keepdim=True)   # 채널 L2 norm
    h2 = enc_out[1].norm(dim=1, keepdim=True)

    h1 = F.interpolate(h1, size=(H, W), mode='bilinear', align_corners=False)
    h2 = F.interpolate(h2, size=(H, W), mode='bilinear', align_corners=False)
    heatmap = ((h1 + h2) / 2).squeeze()         # [H, W]

    # min-max 정규화 → [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return heatmap
```

**근거:** StyleFilter 인코더는 날씨 유형을 구분하도록 학습되었으므로,
인코더 feature의 채널 활성화 강도가 날씨 아티팩트 위치와 상관관계를 가진다고 가정.
Network_top(복원)의 잔차를 쓰는 방식보다 빠르고 체크포인트도 1개만 필요.

---

### [v3] `generate_heatmaps.py` — StyleFilter 단독 사용으로 교체

**변경:**
- `--ckpt_backbone`, `--ckpt_style` → `--ckpt_style` 하나로 단순화
- `Network_top` import 및 로드 제거
- `style_filter.module.encode_spatial(img_t)` 호출로 heatmap 생성

---

### [v3] README, Colab 노트북 업데이트

- Drive 구조에서 `backbone.pth` 제거, `style_filter.pth` 하나만 표기
- Colab Section 3: `CKPT_BACKBONE` 변수 제거, `--ckpt_style` 단일 인자로 교체

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
- [x] `generate_heatmaps.py` 작성 (v1: WeatherEdit → v2: MWFormer 공식 → v3: StyleFilter 단독)
- [x] `mwformer/` 패키지 내장 (backbone, style_filter, base_networks)
- [x] `mwformer/style_filter.py` 버그 수정 + `encode_spatial()` 추가
- [x] Colab 노트북 작성 및 체크포인트 단일화
- [x] README 업데이트 (HGSplat 제목, 체크포인트 1개로 단순화)
- [ ] Colab에서 체크포인트 확인 및 heatmap 생성 테스트
- [ ] Baseline 실험 실행
- [ ] Ours (α=5) 실험 실행
- [ ] 결과 정리 및 비교표 작성
