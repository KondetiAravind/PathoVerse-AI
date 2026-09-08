from __future__ import annotations

from typing import Tuple

import numpy as np


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into uint8 representation.
    """

    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)

    if image.max() <= 1.0:
        image = image * 255.0

    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 [0, 1].
    """

    image = ensure_uint8(image)

    return image.astype(np.float32) / 255.0


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """
    Convert CHW tensor/image layout to HWC.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, received shape {image.shape}"
        )

    return np.transpose(image, (1, 2, 0))


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Convert HWC image layout to CHW.
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image, received shape {image.shape}"
        )

    return np.transpose(image, (2, 0, 1))


def resize_shape(
    height: int,
    width: int,
    target_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Return target spatial dimensions.
    """

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    target_height, target_width = target_size

    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target dimensions must be positive.")

    return target_height, target_width