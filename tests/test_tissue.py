import numpy as np

from pathoverse.wsi.tissue import (
    TissueDetector,
    TissueMaskResult,
    resize_mask,
)


def test_resize_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    resized = resize_mask(
        mask,
        20,
        20,
    )

    assert resized.shape == (20, 20)
    assert resized.dtype == bool
    assert resized.mean() > 0


def test_tissue_mask_save(tmp_path):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    result = TissueMaskResult(
        mask=mask,
        width=20,
        height=20,
        tissue_ratio=float(mask.mean()),
        thumbnail_width=20,
        thumbnail_height=20,
    )

    output = tmp_path / "mask.png"

    result.save(output)

    assert output.exists()