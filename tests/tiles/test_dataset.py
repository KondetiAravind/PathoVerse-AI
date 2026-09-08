import json

import numpy as np
import torch
from PIL import Image

from pathoverse.tiles.dataset import (
    PathoVerseTileDataset,
)


def test_dataset(tmp_path):

    image_path = (
        tmp_path / "tile.png"
    )

    image = Image.fromarray(
        np.full(
            (224, 224, 3),
            120,
            dtype=np.uint8,
        )
    )

    image.save(image_path)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    manifest = {
        "tiles": [
            {
                "tile_id": 0,
                "slide_id": "slide",
                "x": 0,
                "y": 0,
                "level": 0,
                "tissue_ratio": 0.8,
                "image_path": str(
                    image_path
                ),
            }
        ]
    }

    manifest_path.write_text(
        json.dumps(manifest)
    )

    dataset = PathoVerseTileDataset(
        manifest_path
    )

    assert len(dataset) == 1

    item = dataset[0]

    assert item["image"].shape == (
    3,
    224,
    224,
    )

    assert item["image"].dtype == torch.float32

    assert float(item["image"].min()) >= 0.0
    assert float(item["image"].max()) <= 1.0
