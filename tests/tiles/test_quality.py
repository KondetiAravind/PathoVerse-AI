import numpy as np

from pathoverse.tiles.quality import (
    TileQualityChecker,
)


def test_valid_tile():

    image = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    image[:112] = 80
    image[112:] = 180

    checker = TileQualityChecker(
        min_tissue_ratio=0.5,
        min_std=8.0,
    )

    result = checker.check(
        image,
        tissue_ratio=0.8,
    )

    assert result.passed
    assert result.reason == "valid"


def test_low_tissue():

    image = np.full(
        (224, 224, 3),
        120,
        dtype=np.uint8,
    )

    checker = TileQualityChecker()

    result = checker.check(
        image,
        tissue_ratio=0.1,
    )

    assert not result.passed
    assert result.reason == "low_tissue"


def test_low_variance():

    image = np.full(
        (224, 224, 3),
        120,
        dtype=np.uint8,
    )

    checker = TileQualityChecker(
        min_tissue_ratio=0.1,
    )

    result = checker.check(
        image,
        tissue_ratio=0.8,
    )

    assert not result.passed
    assert result.reason == "low_variance"