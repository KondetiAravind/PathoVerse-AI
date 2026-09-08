from pathlib import Path

import numpy as np

from pathoverse.datasets.base import DatasetInfo
from pathoverse.datasets.medmnist import PathMNISTDataset
from pathoverse.datasets.patchcamelyon import (
    PatchCamelyonDataset,
)
from pathoverse.datasets.preprocessing import (
    chw_to_hwc,
    ensure_uint8,
    hwc_to_chw,
    normalize_image,
)
from pathoverse.datasets.registry import DATASET_REGISTRY


def test_dataset_registry():
    datasets = DATASET_REGISTRY.list()

    assert "pathmnist" in datasets
    assert "patchcamelyon" in datasets


def test_pathmnist_info(tmp_path: Path):
    dataset = PathMNISTDataset(tmp_path)

    info = dataset.info()

    assert isinstance(info, DatasetInfo)
    assert info.name == "pathmnist"
    assert info.num_classes == 9
    assert info.image_shape == (28, 28, 3)


def test_patchcamelyon_info(tmp_path: Path):
    dataset = PatchCamelyonDataset(tmp_path)

    info = dataset.info()

    assert info.name == "patchcamelyon"
    assert info.num_classes == 2
    assert info.image_shape == (96, 96, 3)


def test_uint8_conversion():
    image = np.array(
        [[[0.0, 0.5, 1.0]]],
        dtype=np.float32,
    )

    result = ensure_uint8(image)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalization():
    image = np.array(
        [[[0, 128, 255]]],
        dtype=np.uint8,
    )

    result = normalize_image(image)

    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 1


def test_layout_conversion():
    image = np.random.randint(
        0,
        255,
        size=(3, 28, 28),
        dtype=np.uint8,
    )

    hwc = chw_to_hwc(image)

    assert hwc.shape == (28, 28, 3)

    chw = hwc_to_chw(hwc)

    assert chw.shape == image.shape
    assert np.array_equal(chw, image)