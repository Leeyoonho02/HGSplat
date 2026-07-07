"""
utils/heatmap_loss.py
─────────────────────
Weather-aware weighted photometric loss.

LongSplat의 train.py에서 import해서 사용.

v1: 사전 계산된 heatmap(.npy, H_single)만으로 W=exp(-alpha*H) 재가중.
v2: Refinement 루프에서 렌더-잔차 EMA(H_multi)를 H_single 과 블렌딩.
    H = (1-λ)·H_single + λ·H_multi,  λ = min(refine_iter / mv_ramp, 1)
    (근거: 눈 등 날씨는 view-inconsistent 라 학습이 진행돼도 렌더-잔차가
     사라지지 않고, 실제 텍스처는 다수 뷰 합의로 수렴하며 잔차가 사라짐.
     자세한 설계는 docs/method_v2.md 참고.)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F


class HeatmapWeightedLoss:
    """
    Pre-computed heatmap(.npy)을 로드하고 weighted L1 loss를 계산.

    Parameters
    ----------
    heatmap_dir : str
        restoration.py / generate_heatmaps.py 가 출력한 .npy 폴더 경로.
    device : torch.device
    enabled : bool
        False이면 일반 L1 loss로 fallback (Baseline 재현용).
    alpha : float
        W_t = exp(-alpha * H_t) 의 감쇠 계수.
        .npy 에는 raw heatmap H_t 가 저장되어 있으며, weight map W_t 는
        호출 시점에 이 alpha 로 계산된다. 따라서 --heatmap_alpha 만 바꾸면
        heatmap 재생성 없이 alpha ablation 이 가능하다.
    norm : str
        H 재스케일 방식. "none"(원본 H 사용) 또는 "frame"(프레임별 robust 정규화).
        "frame": H' = clip(H / max(percentile(H, pct), floor), 0, 1).
        진단 1에서 눈 H 가 ~0.01 로 너무 작아 alpha=5 로도 거의 안 깎이는 문제,
        그리고 객체 프레임의 절대 H 가 낮아 덜 다운웨이트되는 문제를 동시에 교정.
        floor 게이트로 눈이 거의 없는 프레임에서 노이즈가 증폭되는 것을 방지.
    pct : float
        norm="frame" 시 분모로 쓰는 백분위수 (기본 99).
    floor : float
        프레임 percentile 이 이 값 미만이면 정규화하지 않음(노이즈 증폭 방지, 기본 0.05).
    mv : bool
        [v2] True 면 refine_iter 가 주어진 호출에서 렌더-잔차 EMA(H_multi) 블렌딩 활성화.
    mv_beta : float
        [v2] 뷰별 렌더-잔차 EMA decay 계수. ema = beta*ema + (1-beta)*|render-gt|.
    mv_ramp : int
        [v2] λ 램프 길이(refine_iter 기준). refine_iter >= mv_ramp 이면 λ=1.
    """

    def __init__(self, heatmap_dir: str, device: torch.device, enabled: bool = True, alpha: float = 5.0,
                 norm: str = "frame", pct: float = 99.0, floor: float = 0.05,
                 log_interval: int = 100,
                 mv: bool = False, mv_beta: float = 0.9, mv_ramp: int = 10000):
        self.heatmap_dir = heatmap_dir
        self.device = device
        self.enabled = enabled
        self.alpha = alpha
        self.norm = norm
        self.pct = pct
        self.floor = floor
        self.log_interval = log_interval
        self.mv = mv
        self.mv_beta = mv_beta
        self.mv_ramp = mv_ramp
        # v1: 정규화된 H_single 캐시 (주의: exp 적용 전 H 를 캐싱한다.
        #     v2 블렌딩이 exp 이전 단계에서 일어나야 하기 때문)
        self._cache: dict[str, torch.Tensor] = {}                 # 원본(npy) 해상도 H
        self._cache_resized: dict[tuple, torch.Tensor] = {}       # (image_name, h, w) → H
        # v2: 뷰별 렌더-잔차 EMA 버퍼 (렌더 해상도 [H, W])
        self._ema: dict[str, torch.Tensor] = {}
        self._call_count = 0

        if enabled and not os.path.isdir(heatmap_dir):
            raise FileNotFoundError(
                f"Heatmap 폴더를 찾을 수 없습니다: {heatmap_dir}\n"
                "restoration.py (또는 generate_heatmaps.py) 를 먼저 실행하세요."
            )

    # ──────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────

    def _frame_norm(self, h: torch.Tensor) -> torch.Tensor:
        """[v7] 프레임별 robust 정규화. norm="none" 이면 그대로 반환."""
        if self.norm == "frame":
            # floor 게이트: percentile 이 floor 미만(눈 거의 없는 프레임)이면 정규화 안 함.
            p = torch.quantile(h, self.pct / 100.0)
            scale = torch.clamp(p, min=self.floor)
            h = torch.clamp(h / scale, 0.0, 1.0)
        return h

    def _load_h(self, image_name: str) -> torch.Tensor | None:
        """image_name(확장자 제외)에 대응하는 정규화된 H_single 을 반환."""
        if not self.enabled:
            return None

        if image_name in self._cache:
            return self._cache[image_name]

        npy_path = os.path.join(self.heatmap_dir, f"{image_name}.npy")
        if not os.path.exists(npy_path):
            return None

        h = torch.from_numpy(np.load(npy_path)).to(self.device)  # [H, W] raw heatmap
        h = self._frame_norm(h)
        self._cache[image_name] = h
        return h

    def _resize_h(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """H [H, W] → target [C, H', W'] 해상도에 맞게 리사이즈."""
        _, th, tw = target.shape
        return F.interpolate(
            h.unsqueeze(0).unsqueeze(0),   # [1, 1, H, W]
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # [H', W']

    def _update_h_multi(self, diff: torch.Tensor, image_name: str) -> torch.Tensor:
        """[v2] 뷰별 렌더-잔차 EMA 갱신 후 정규화된 H_multi 반환.

        diff : [C, H, W] = |render - gt| (grad 차단해서 사용)
        """
        r = diff.detach().mean(dim=0)  # [H, W]
        ema = self._ema.get(image_name)
        if ema is None or ema.shape != r.shape:
            # cold start (또는 해상도 변경 시 리셋)
            self._ema[image_name] = r.clone()
        else:
            ema.mul_(self.mv_beta).add_(r, alpha=1.0 - self.mv_beta)
        return self._frame_norm(self._ema[image_name])

    # ──────────────────────────────────
    # 퍼블릭 API
    # ──────────────────────────────────

    def __call__(
        self,
        render: torch.Tensor,
        gt: torch.Tensor,
        image_name: str,
        refine_iter: int | None = None,
    ) -> torch.Tensor:
        """
        Weighted L1 photometric loss.

        Parameters
        ----------
        render : torch.Tensor  [C, H, W]  렌더링 결과
        gt     : torch.Tensor  [C, H, W]  GT 이미지
        image_name : str  파일 stem (확장자 없이, e.g. "frame_00001")
        refine_iter : int | None
            [v2] Refinement 루프 진입 후 경과 iteration (1부터).
            None 이면 v1 경로(H_single 만). mv=False 면 값이 있어도 무시.

        Returns
        -------
        loss : scalar tensor
        """
        self._call_count += 1
        diff = (render - gt).abs()  # [C, H, W]

        h_single = self._load_h(image_name)
        if h_single is None:
            if self._call_count % self.log_interval == 1:
                plain = diff.mean().item()
                print(f"[HeatmapLoss iter={self._call_count}] mode=BASELINE  L1={plain:.6f}")
            return diff.mean()

        _, th, tw = diff.shape
        if h_single.shape != (th, tw):
            cache_key = (image_name, th, tw)
            if cache_key not in self._cache_resized:
                self._cache_resized[cache_key] = self._resize_h(h_single, diff)
            h_single = self._cache_resized[cache_key]

        lam = None
        if self.mv and refine_iter is not None:
            h_multi = self._update_h_multi(diff, image_name)
            lam = 1.0 if self.mv_ramp <= 0 else min(refine_iter / self.mv_ramp, 1.0)
            h = (1.0 - lam) * h_single + lam * h_multi
        else:
            h = h_single

        weight = torch.exp(-self.alpha * h)  # [H, W]

        # weight: [H, W] → [1, H, W] broadcast
        wmap = weight.unsqueeze(0)
        weighted_loss = (diff * wmap).sum() / (wmap.sum() * diff.shape[0] + 1e-8)

        if self._call_count % self.log_interval == 1:
            plain = diff.mean().item()
            mode = "OURS-v2" if lam is not None else "OURS"
            extra = f"  lambda={lam:.3f}" if lam is not None else ""
            print(f"[HeatmapLoss iter={self._call_count}] mode={mode}  "
                  f"L1_weighted={weighted_loss.item():.6f}  L1_plain={plain:.6f}  "
                  f"weight_mean={weight.mean().item():.4f}  weight_min={weight.min().item():.4f}{extra}")

        return weighted_loss
