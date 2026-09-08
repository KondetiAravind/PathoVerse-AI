from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from skimage.transform import resize

from .reader import WSIReader


@dataclass(frozen=True)
class TissueMaskResult:
    mask: np.ndarray
    width: int
    height: int
    tissue_ratio: float
    thumbnail_width: int
    thumbnail_height: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.fromarray(
            (self.mask.astype(np.uint8) * 255),
            mode="L",
        )

        image.save(path)


class TissueDetector:
    """
    Lightweight classical tissue detector for brightfield pathology WSIs.

    The detector works on a thumbnail rather than the full-resolution WSI,
    making it inexpensive even for gigapixel slides.
    """

    def __init__(
        self,
        *,
        thumbnail_width: int = 2048,
        min_object_size: int = 64,
        morphology_radius: int = 3,
    ):
        self.thumbnail_width = thumbnail_width
        self.min_object_size = min_object_size
        self.morphology_radius = morphology_radius

    def detect(self, reader: WSIReader) -> TissueMaskResult:
        thumbnail = reader.read_thumbnail(
            self.thumbnail_width,
            self.thumbnail_width,
        )

        image = np.asarray(thumbnail).astype(np.float32) / 255.0

        hsv = rgb2hsv(image)

        saturation = hsv[..., 1]
        value = hsv[..., 2]

        # White background generally has low saturation and high value.
        # Tissue tends to have stronger chromatic content.
        non_white = value < 0.95

        saturation_values = saturation[non_white]

        if saturation_values.size == 0:
            mask = np.zeros(
                image.shape[:2],
                dtype=bool,
            )
        else:
            threshold = threshold_otsu(saturation_values)

            mask = (
                (saturation > threshold * 0.5)
                | (value < 0.85)
            )

            mask &= non_white

        structure = disk(self.morphology_radius)

        mask = binary_closing(
            mask,
            structure,
        )

        mask = binary_opening(
            mask,
            structure,
        )

        mask = remove_small_objects(
            mask,
            min_size=self.min_object_size,
        )

        tissue_ratio = float(mask.mean())

        return TissueMaskResult(
            mask=mask,
            width=image.shape[1],
            height=image.shape[0],
            tissue_ratio=tissue_ratio,
            thumbnail_width=image.shape[1],
            thumbnail_height=image.shape[0],
        )


def resize_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Resize a binary tissue mask to another resolution.
    """

    if width <= 0 or height <= 0:
        raise ValueError(
            "width and height must be positive."
        )

    resized = resize(
        mask.astype(np.float32),
        (height, width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return resized >= 0.5