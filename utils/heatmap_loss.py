"""
utils/heatmap_loss.py
─────────────────────
Weather-aware weighted photometric loss.

LongSplat의 train.py에서 import해서 사용.
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
        generate_heatmaps.py가 출력한 .npy 폴더 경로.
    device : torch.device
    enabled : bool
        False이면 일반 L1 loss로 fallback (Baseline 재현용).
    alpha : float
        W_t = exp(-alpha * H_t) 의 감쇠 계수.
        .npy 에는 raw heatmap H_t 가 저장되어 있으며, weight map W_t 는
        로드 시점에 이 alpha 로 계산된다. 따라서 --heatmap_alpha 만 바꾸면
        heatmap 재생성 없이 alpha ablation 이 가능하다.
    """

    def __init__(self, heatmap_dir: str, device: torch.device, enabled: bool = True, alpha: float = 5.0,
                 log_interval: int = 100):
        self.heatmap_dir = heatmap_dir
        self.device = device
        self.enabled = enabled
        self.alpha = alpha
        self.log_interval = log_interval
        self._cache: dict[str, torch.Tensor] = {}        # 원본 해상도
        self._cache_resized: dict[tuple, torch.Tensor] = {}  # (image_name, h, w) 키
        self._call_count = 0

        if enabled and not os.path.isdir(heatmap_dir):
            raise FileNotFoundError(
                f"Heatmap 폴더를 찾을 수 없습니다: {heatmap_dir}\n"
                "generate_heatmaps.py 를 먼저 실행하세요."
            )

    # ──────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────

    def _load_weight(self, image_name: str) -> torch.Tensor | None:
        """image_name(확장자 제외)에 대응하는 weight map W=exp(-alpha*H)를 반환."""
        if not self.enabled:
            return None

        if image_name in self._cache:
            return self._cache[image_name]

        npy_path = os.path.join(self.heatmap_dir, f"{image_name}.npy")
        if not os.path.exists(npy_path):
            return None

        h = torch.from_numpy(np.load(npy_path)).to(self.device)  # [H, W] raw heatmap
        w = torch.exp(-self.alpha * h)                           # [H, W] weight map
        self._cache[image_name] = w
        return w

    def _resize_weight(
        self, weight: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """weight [H, W] → target [C, H', W'] 해상도에 맞게 리사이즈."""
        _, h, w = target.shape
        return F.interpolate(
            weight.unsqueeze(0).unsqueeze(0),   # [1, 1, H, W]
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # [H', W']

    # ──────────────────────────────────
    # 퍼블릭 API
    # ──────────────────────────────────

    def __call__(
        self,
        render: torch.Tensor,
        gt: torch.Tensor,
        image_name: str,
    ) -> torch.Tensor:
        """
        Weighted L1 photometric loss.

        Parameters
        ----------
        render : torch.Tensor  [C, H, W]  렌더링 결과
        gt     : torch.Tensor  [C, H, W]  GT 이미지
        image_name : str  파일 stem (확장자 없이, e.g. "frame_00001")

        Returns
        -------
        loss : scalar tensor
        """
        self._call_count += 1
        diff = (render - gt).abs()  # [C, H, W]

        weight = self._load_weight(image_name)
        if weight is None:
            if self._call_count % self.log_interval == 1:
                plain = diff.mean().item()
                print(f"[HeatmapLoss iter={self._call_count}] mode=BASELINE  L1={plain:.6f}")
            return diff.mean()

        _, h, w = diff.shape
        if weight.shape != (h, w):
            cache_key = (image_name, h, w)
            if cache_key not in self._cache_resized:
                self._cache_resized[cache_key] = self._resize_weight(weight, diff)
            weight = self._cache_resized[cache_key]

        # weight: [H, W] → [1, H, W] broadcast
        wmap = weight.unsqueeze(0)
        weighted_loss = (diff * wmap).sum() / (wmap.sum() * diff.shape[0] + 1e-8)

        if self._call_count % self.log_interval == 1:
            plain = diff.mean().item()
            print(f"[HeatmapLoss iter={self._call_count}] mode=OURS  "
                  f"L1_weighted={weighted_loss.item():.6f}  L1_plain={plain:.6f}  "
                  f"weight_mean={weight.mean().item():.4f}  weight_min={weight.min().item():.4f}")

        return weighted_loss
