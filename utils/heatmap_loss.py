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
    """

    def __init__(self, heatmap_dir: str, device: torch.device, enabled: bool = True):
        self.heatmap_dir = heatmap_dir
        self.device = device
        self.enabled = enabled
        self._cache: dict[str, torch.Tensor] = {}

        if enabled and not os.path.isdir(heatmap_dir):
            raise FileNotFoundError(
                f"Heatmap 폴더를 찾을 수 없습니다: {heatmap_dir}\n"
                "generate_heatmaps.py 를 먼저 실행하세요."
            )

    # ──────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────

    def _load_weight(self, image_name: str) -> torch.Tensor | None:
        """image_name(확장자 제외)에 대응하는 weight map을 반환."""
        if not self.enabled:
            return None

        if image_name in self._cache:
            return self._cache[image_name]

        npy_path = os.path.join(self.heatmap_dir, f"{image_name}.npy")
        if not os.path.exists(npy_path):
            return None

        w = torch.from_numpy(np.load(npy_path)).to(self.device)  # [H, W]
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
        diff = (render - gt).abs()  # [C, H, W]

        weight = self._load_weight(image_name)
        if weight is None:
            # heatmap 없으면 일반 L1
            return diff.mean()

        if weight.shape != diff.shape[-2:]:
            weight = self._resize_weight(weight, diff)

        # weight: [H, W] → [1, H, W] broadcast
        w = weight.unsqueeze(0)
        return (diff * w).sum() / (w.sum() * diff.shape[0] + 1e-8)
