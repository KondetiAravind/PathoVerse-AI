from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token
from PIL import Image

from pathoverse.models.adapters.conch import ConchAdapter


# ============================================================
# PATHS
# ============================================================

TILES_MANIFEST = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

OUTPUT_EMBEDDINGS = Path(
    "data/processed/embeddings/"
    "CMU-1-Small-Region_conch.npy"
)

OUTPUT_METADATA = Path(
    "data/metadata/embeddings/"
    "CMU-1-Small-Region_conch.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4

CANONICAL_MODEL_ID = "conch"

ARTIFACT_MODEL_ID = (
    "conch_ViT-B-16"
)

SLIDE_ID = (
    "CMU-1-Small-Region"
)

EMBEDDING_DIMENSION = 512


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "PATHOVERSE — CONCH TILE EMBEDDING EXTRACTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for CONCH extraction."
        )

    print(
        "PyTorch       :",
        torch.__version__,
    )

    print(
        "GPU           :",
        torch.cuda.get_device_name(0),
    )

    print(
        "Canonical ID  :",
        CANONICAL_MODEL_ID,
    )

    print(
        "Artifact ID   :",
        ARTIFACT_MODEL_ID,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not TILES_MANIFEST.exists():
        raise FileNotFoundError(
            f"Tiles manifest not found: "
            f"{TILES_MANIFEST}"
        )

    with TILES_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    tiles = manifest.get(
        "tiles",
        [],
    )

    if not tiles:
        raise ValueError(
            "No tiles found in manifest."
        )

    print(
        "Input tiles   :",
        len(tiles),
    )

    print(
        "Batch size    :",
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    token = get_token()

    if not token:
        raise RuntimeError(
            "Hugging Face authentication token "
            "not found."
        )

    # --------------------------------------------------------
    # Adapter
    # --------------------------------------------------------

    adapter = ConchAdapter(
        device="cuda",
        batch_size=BATCH_SIZE,
        hf_auth_token=token,
    )

    print()
    print(
        "Loading CONCH..."
    )

    load_start = time.perf_counter()

    adapter.load()

    load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        "Load time     :",
        f"{load_time:.2f} sec",
    )

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    embeddings = []

    valid_tiles = []

    print()
    print(
        "Extracting embeddings..."
    )

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(tiles),
        BATCH_SIZE,
    ):

        batch_tiles = tiles[
            batch_start :
            batch_start + BATCH_SIZE
        ]

        images = []

        for tile in batch_tiles:

            image_path = Path(
                tile["image_path"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    "Tile image not found: "
                    f"{image_path}"
                )

            image = (
                Image.open(
                    image_path
                )
                .convert("RGB")
            )

            images.append(
                image
            )

        batch_embeddings = (
            adapter.encode(
                images
            )
        )

        embeddings.append(
            batch_embeddings
            .detach()
            .cpu()
            .numpy()
        )

        valid_tiles.extend(
            batch_tiles
        )

        processed = min(
            batch_start
            + BATCH_SIZE,
            len(tiles),
        )

        print(
            f"  Processed "
            f"{processed}/{len(tiles)}"
        )

    total_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Build matrix
    # --------------------------------------------------------

    embedding_matrix = (
        np.concatenate(
            embeddings,
            axis=0,
        )
        .astype(
            np.float32
        )
    )

    expected_shape = (
        len(valid_tiles),
        EMBEDDING_DIMENSION,
    )

    if embedding_matrix.shape != (
        expected_shape
    ):

        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embedding_matrix.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.isfinite(
        embedding_matrix
    ).all():

        raise RuntimeError(
            "Embedding matrix contains "
            "NaN or Inf values."
        )

    norms = np.linalg.norm(
        embedding_matrix,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):

        raise RuntimeError(
            "CONCH embeddings are not normalized."
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    OUTPUT_EMBEDDINGS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        OUTPUT_EMBEDDINGS,
        embedding_matrix,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "schema_version": "1.0",

        "model": {
            "name": "CONCH",

            # Canonical PathoVerse identifier.
            "canonical_model_id": (
                CANONICAL_MODEL_ID
            ),

            # Original/upstream artifact identifier.
            "model_id": (
                ARTIFACT_MODEL_ID
            ),

            "embedding_dim": (
                EMBEDDING_DIMENSION
            ),

            "input_size": 448,

            "modality": (
                "vision-language"
            ),
        },

        "source": {
            "manifest": str(
                TILES_MANIFEST
            ),
            "slide_id": SLIDE_ID,
        },

        "statistics": {
            "tile_count": len(
                valid_tiles
            ),

            "embedding_shape": list(
                embedding_matrix.shape
            ),

            "embedding_dtype": str(
                embedding_matrix.dtype
            ),

            "normalized": True,

            "load_time_sec": (
                load_time
            ),

            "extraction_time_sec": (
                total_time
            ),

            "latency_ms_per_tile": (
                total_time
                / len(valid_tiles)
                * 1000
            ),

            "throughput_tiles_per_sec": (
                len(valid_tiles)
                / total_time
            ),
        },

        "tiles": [
            {
                "tile_id": tile[
                    "tile_id"
                ],

                "slide_id": tile[
                    "slide_id"
                ],

                "x": tile["x"],
                "y": tile["y"],
                "width": tile[
                    "width"
                ],
                "height": tile[
                    "height"
                ],

                "level": tile[
                    "level"
                ],

                "tissue_ratio": tile[
                    "tissue_ratio"
                ],

                "image_path": tile[
                    "image_path"
                ],
            }
            for tile in valid_tiles
        ],
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CONCH EMBEDDING EXTRACTION COMPLETE"
    )
    print("=" * 70)

    print(
        "Tiles         :",
        len(valid_tiles),
    )

    print(
        "Embedding     :",
        embedding_matrix.shape,
    )

    print(
        "Dtype         :",
        embedding_matrix.dtype,
    )

    print(
        "Mean          :",
        f"{embedding_matrix.mean():.6f}",
    )

    print(
        "Std           :",
        f"{embedding_matrix.std():.6f}",
    )

    print(
        "Min           :",
        f"{embedding_matrix.min():.6f}",
    )

    print(
        "Max           :",
        f"{embedding_matrix.max():.6f}",
    )

    print(
        "Latency       :",
        f"{total_time / len(valid_tiles) * 1000:.2f}"
        " ms/tile",
    )

    print(
        "Throughput    :",
        f"{len(valid_tiles) / total_time:.2f}"
        " tiles/sec",
    )

    print()
    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_EMBEDDINGS,
    )

    print(
        " ",
        OUTPUT_METADATA,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()