from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pathoverse.models.adapters.conch import (
    ConchAdapter,
)
from pathoverse.models.adapters.gigapath import (
    GigaPathFlashAdapter,
)
from pathoverse.models.adapters.timm_vision import (
    TimmVisionAdapter,
)


# ============================================================
# PATHS
# ============================================================

TILE_MANIFEST = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

OUTPUT_DIR = Path(
    "data/processed/wsi/embeddings"
)

METADATA_DIR = Path(
    "data/metadata/wsi/embeddings"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {

    "vit_b_16": {
        "factory": lambda device: TimmVisionAdapter(
            model_id="vit_base_patch16_224",
            device=device,
            batch_size=32,
        ),
    },

    "gigapath_flash": {
        "factory": lambda device: GigaPathFlashAdapter(
            device=device,
            batch_size=32,
        ),
    },

    "conch": {
        "factory": lambda device: ConchAdapter(
            device=device,
            batch_size=16,
        ),
    },

}


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract foundation-model embeddings "
            "for all tiles belonging to a WSI."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIGS.keys(),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load tile manifest
    # ========================================================

    if not TILE_MANIFEST.exists():

        raise FileNotFoundError(
            f"Tile manifest not found: "
            f"{TILE_MANIFEST}"
        )

    with open(
        TILE_MANIFEST,
        "r",
    ) as f:

        manifest = json.load(f)

    tiles = manifest[
        "tiles"
    ]

    if not tiles:

        raise RuntimeError(
            "Tile manifest contains no tiles."
        )

    slide_ids = {
        tile["slide_id"]
        for tile in tiles
    }

    if len(slide_ids) != 1:

        raise RuntimeError(
            "This script expects exactly "
            "one WSI in the tile manifest."
        )

    slide_id = next(
        iter(slide_ids)
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 72)

    print(
        "PATHOVERSE — WSI FOUNDATION EMBEDDINGS"
    )

    print("=" * 72)

    print(
        f"Slide  : {slide_id}"
    )

    print(
        f"Model  : {args.model}"
    )

    print(
        f"Tiles  : {len(tiles)}"
    )

    print(
        f"Device : {args.device}"
    )

    print("=" * 72)

    # ========================================================
    # Load adapter
    # ========================================================

    print()
    print(
        "Loading foundation model..."
    )

    load_start = time.perf_counter()

    adapter = MODEL_CONFIGS[
        args.model
    ]["factory"](
        args.device
    )

    adapter.ensure_loaded()

    load_time = (
        time.perf_counter()
        - load_start
    )

    info = adapter.info()

    print(
        f"Model name       : {info.name}"
    )

    print(
        f"Embedding dim    : {info.embedding_dim}"
    )

    print(
        f"Input size       : {info.input_size}"
    )

    print(
        f"Load time        : {load_time:.2f} sec"
    )

    # ========================================================
    # Extraction
    # ========================================================

    embeddings = []

    batch_size = adapter.batch_size

    extraction_start = (
        time.perf_counter()
    )

    for start in range(
        0,
        len(tiles),
        batch_size,
    ):

        batch_tiles = tiles[
            start:start + batch_size
        ]

        images = []

        for tile in batch_tiles:

            image_path = Path(
                tile["image_path"]
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Tile image not found: "
                    f"{image_path}"
                )

            from PIL import Image

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            images.append(
                image
            )

        with torch.inference_mode():

            batch_embeddings = (
                adapter.encode(
                    images
                )
            )

        batch_embeddings = (
            batch_embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        embeddings.append(
            batch_embeddings
        )

        processed = min(
            start + batch_size,
            len(tiles),
        )

        elapsed = (
            time.perf_counter()
            - extraction_start
        )

        throughput = (
            processed / elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"\rProcessed "
            f"{processed}/{len(tiles)} "
            f"| {throughput:.2f} tiles/sec",
            end="",
            flush=True,
        )

    print()

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    extraction_time = (
        time.perf_counter()
        - extraction_start
    )

    # ========================================================
    # Validation
    # ========================================================

    if embeddings.shape[0] != len(tiles):

        raise RuntimeError(
            "Embedding count does not "
            "match tile count."
        )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Embeddings must be 2-dimensional."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "Embeddings contain NaN or Inf."
        )

    # ========================================================
    # Output
    # ========================================================

    embedding_path = (
        OUTPUT_DIR
        / (
            f"{slide_id}_"
            f"{args.model}_embeddings.npy"
        )
    )

    metadata_path = (
        METADATA_DIR
        / (
            f"{slide_id}_"
            f"{args.model}_embeddings.json"
        )
    )

    np.save(
        embedding_path,
        embeddings,
    )

    metadata = {

        "schema_version": "1.0",

        "dataset": "WSI",

        "slide_id": slide_id,

        "model": {
            "name": info.name,
            "model_id": info.model_id,
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "input_size": int(
                info.input_size
            ),
            "modality": info.modality,
        },

        "tiles": [
            {
                "tile_id": int(
                    tile["tile_id"]
                ),
                "x": int(
                    tile["x"]
                ),
                "y": int(
                    tile["y"]
                ),
                "width": int(
                    tile["width"]
                ),
                "height": int(
                    tile["height"]
                ),
                "level": int(
                    tile["level"]
                ),
                "tissue_ratio": float(
                    tile["tissue_ratio"]
                ),
                "image_path": tile[
                    "image_path"
                ],
            }
            for tile in tiles
        ],

        "statistics": {
            "tile_count": int(
                len(tiles)
            ),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "extraction_time_sec": float(
                extraction_time
            ),
            "throughput_tiles_per_sec": float(
                len(tiles)
                / extraction_time
                if extraction_time > 0
                else 0.0
            ),
        },

        "artifacts": {
            "embedding_file": str(
                embedding_path
            ),
            "tile_manifest": str(
                TILE_MANIFEST
            ),
        },

    }

    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 72)

    print(
        "WSI FOUNDATION EMBEDDING EXTRACTION COMPLETE"
    )

    print("=" * 72)

    print(
        f"Slide           : {slide_id}"
    )

    print(
        f"Model           : {info.name}"
    )

    print(
        f"Tiles           : {len(tiles)}"
    )

    print(
        f"Embedding shape : {embeddings.shape}"
    )

    print(
        f"Time            : {extraction_time:.2f} sec"
    )

    print(
        f"Throughput      : "
        f"{len(tiles) / extraction_time:.2f} tiles/sec"
    )

    print()
    print(
        "Saved:",
        embedding_path,
    )

    print(
        "Saved:",
        metadata_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()