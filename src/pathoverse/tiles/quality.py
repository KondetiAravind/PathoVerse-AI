from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileQCResult:
    passed: bool
    reason: str
    mean_intensity: float
    std_intensity: float
    tissue_ratio: float

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "mean_intensity": round(
                self.mean_intensity,
                4,
            ),
            "std_intensity": round(
                self.std_intensity,
                4,
            ),
            "tissue_ratio": round(
                self.tissue_ratio,
                6,
            ),
        }


class TileQualityChecker:
    """
    Lightweight quality-control checks for pathology tiles.
    """

    def __init__(
        self,
        *,
        min_tissue_ratio: float = 0.50,
        min_std: float = 8.0,
        min_mean: float = 15.0,
        max_mean: float = 245.0,
    ):
        if not 0.0 <= min_tissue_ratio <= 1.0:
            raise ValueError(
                "min_tissue_ratio must be between 0 and 1."
            )

        if min_std < 0:
            raise ValueError(
                "min_std must be non-negative."
            )

        if not 0 <= min_mean <= 255:
            raise ValueError(
                "min_mean must be between 0 and 255."
            )

        if not 0 <= max_mean <= 255:
            raise ValueError(
                "max_mean must be between 0 and 255."
            )

        if min_mean >= max_mean:
            raise ValueError(
                "min_mean must be smaller than max_mean."
            )

        self.min_tissue_ratio = min_tissue_ratio
        self.min_std = min_std
        self.min_mean = min_mean
        self.max_mean = max_mean

    def check(
        self,
        image: Image.Image | np.ndarray,
        tissue_ratio: float,
    ) -> TileQCResult:

        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                "Tile must have shape (H, W, 3)."
            )

        if not 0.0 <= tissue_ratio <= 1.0:
            raise ValueError(
                "tissue_ratio must be between 0 and 1."
            )

        gray = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )

        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())

        if tissue_ratio < self.min_tissue_ratio:
            return TileQCResult(
                passed=False,
                reason="low_tissue",
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                tissue_ratio=tissue_ratio,
            )

        if std_intensity < self.min_std:
            return TileQCResult(
                passed=False,
                reason="low_variance",
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                tissue_ratio=tissue_ratio,
            )

        if mean_intensity < self.min_mean:
            return TileQCResult(
                passed=False,
                reason="too_dark",
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                tissue_ratio=tissue_ratio,
            )

        if mean_intensity > self.max_mean:
            return TileQCResult(
                passed=False,
                reason="too_bright",
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                tissue_ratio=tissue_ratio,
            )

        return TileQCResult(
            passed=True,
            reason="valid",
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            tissue_ratio=tissue_ratio,
        )