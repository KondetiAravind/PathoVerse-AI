from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pathoverse.models.adapters import GigaPathFlashAdapter


def load_manifest(path: Path) -> tuple[list[dict], dict]:
    """
    Load the canonical PathoVerse extracted-tile manifest.

    Manifest structure:

    {
        "schema_version": "...",
        "statistics": {...},
        "tiles": [...],
        "rejected_tiles": [...]
    }
    """

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected tile manifest to be a JSON object."
        )

    if "tiles" not in data:
        raise ValueError(
            "Tile manifest does not contain a 'tiles' field."
        )

    tiles = data["tiles"]

    if not isinstance(tiles, list):
        raise ValueError(
            "Manifest 'tiles' field must be a list."
        )

    return tiles, data


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract GigaPath-Flash embeddings "
            "from PathoVerse pathology tiles."
        )
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/metadata/tiles/"
            "extracted_tiles.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output embedding .npy file",
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help="Output metadata JSON file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    tiles, manifest = load_manifest(
        manifest_path
    )

    if not tiles:
        raise ValueError(
            "Tile manifest contains zero tiles."
        )

    # --------------------------------------------------------------
    # Validate tile structure
    # --------------------------------------------------------------

    required_fields = {
        "tile_id",
        "slide_id",
        "x",
        "y",
        "width",
        "height",
        "level",
        "tissue_ratio",
        "image_path",
    }

    missing = (
        required_fields
        - set(tiles[0].keys())
    )

    if missing:
        raise ValueError(
            "Tile record is missing fields: "
            f"{sorted(missing)}"
        )

    slide_ids = {
        tile["slide_id"]
        for tile in tiles
    }

    if len(slide_ids) != 1:
        raise ValueError(
            "Expected one slide per extraction run, "
            f"found: {sorted(slide_ids)}"
        )

    slide_id = tiles[0]["slide_id"]

    if args.output is None:
        args.output = (
            "data/processed/embeddings/"
            f"{slide_id}_gigapath-flash.npy"
        )

    if args.metadata is None:
        args.metadata = (
            "data/metadata/embeddings/"
            f"{slide_id}_gigapath-flash.json"
        )

    output_path = Path(args.output)
    metadata_path = Path(args.metadata)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print("=" * 75)
    print(
        "PATHOVERSE — GIGAPATH-FLASH "
        "EMBEDDING EXTRACTION"
    )
    print("=" * 75)

    print(
        f"\nManifest       : "
        f"{manifest_path}"
    )

    print(
        f"Schema version : "
        f"{manifest.get('schema_version', 'unknown')}"
    )

    print(
        f"Slide          : "
        f"{slide_id}"
    )

    print(
        f"Tile count     : "
        f"{len(tiles)}"
    )

    print(
        f"Batch size     : "
        f"{args.batch_size}"
    )

    print(
        f"Output         : "
        f"{output_path}"
    )

    print(
        f"Metadata       : "
        f"{metadata_path}"
    )

    print("\nDevice:")

    if torch.cuda.is_available():

        print("CUDA           : True")

        print(
            "GPU            :",
            torch.cuda.get_device_name(0),
        )

    else:

        print(
            "CUDA           : False"
        )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print(
        "\nLoading GigaPath-Flash..."
    )

    adapter = GigaPathFlashAdapter(
        device="cuda",
        batch_size=args.batch_size,
    )

    adapter.load()

    info = adapter.info()

    print("\nModel:")
    print(
        "Name           :",
        info.name,
    )

    print(
        "Embedding dim  :",
        info.embedding_dim,
    )

    print(
        "Input size     :",
        info.input_size,
    )

    # --------------------------------------------------------------
    # Extraction
    # --------------------------------------------------------------

    all_embeddings = []
    output_metadata = []

    total_start = time.perf_counter()

    for start in tqdm(
        range(
            0,
            len(tiles),
            args.batch_size,
        ),
        desc="Extracting embeddings",
    ):

        batch_records = tiles[
            start:
            start + args.batch_size
        ]

        images = []

        for record in batch_records:

            image_path = Path(
                record["image_path"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Tile not found: "
                    f"{image_path}"
                )

            with Image.open(
                image_path
            ) as image:

                image = image.convert(
                    "RGB"
                )

                images.append(image)

        # ----------------------------------------------------------
        # Preprocess
        # ----------------------------------------------------------

        batch = adapter.preprocess(
            images
        )

        # ----------------------------------------------------------
        # Encode
        # ----------------------------------------------------------

        embeddings = adapter.encode(
            batch
        )

        embeddings_np = (
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_embeddings.append(
            embeddings_np
        )

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        for record in batch_records:

            output_metadata.append(
                {
                    "tile_id": record["tile_id"],
                    "slide_id": record["slide_id"],
                    "x": record["x"],
                    "y": record["y"],
                    "width": record["width"],
                    "height": record["height"],
                    "level": record["level"],
                    "tissue_ratio": record[
                        "tissue_ratio"
                    ],
                    "image_path": record[
                        "image_path"
                    ],
                    "model": info.name,
                    "model_id": info.model_id,
                    "embedding_dim": (
                        info.embedding_dim
                    ),
                }
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------

    embeddings_array = np.concatenate(
        all_embeddings,
        axis=0,
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    expected_shape = (
        len(tiles),
        info.embedding_dim,
    )

    if embeddings_array.shape != expected_shape:

        raise RuntimeError(
            "Unexpected embedding matrix shape: "
            f"{embeddings_array.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(
        embeddings_array
    ).all():

        raise RuntimeError(
            "Embedding matrix contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------------

    np.save(
        output_path,
        embeddings_array,
    )

    # --------------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------------

    metadata = {
        "schema_version": "1.0",
        "slide_id": slide_id,
        "model": info.name,
        "model_id": info.model_id,
        "embedding_dim": info.embedding_dim,
        "tile_count": len(tiles),
        "embedding_shape": list(
            embeddings_array.shape
        ),
        "extraction_time_seconds": (
            total_time
        ),
        "tiles_per_second": (
            len(tiles) / total_time
            if total_time > 0
            else 0.0
        ),
        "source_manifest": str(
            manifest_path
        ),
        "tiles": output_metadata,
    }

    with metadata_path.open("w") as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nEmbedding shape : "
        f"{embeddings_array.shape}"
    )

    print(
        f"Embedding dtype : "
        f"{embeddings_array.dtype}"
    )

    print(
        f"Finite          : "
        f"{np.isfinite(embeddings_array).all()}"
    )

    print(
        f"Mean            : "
        f"{embeddings_array.mean():.6f}"
    )

    print(
        f"Std             : "
        f"{embeddings_array.std():.6f}"
    )

    print(
        f"Min             : "
        f"{embeddings_array.min():.6f}"
    )

    print(
        f"Max             : "
        f"{embeddings_array.max():.6f}"
    )

    print(
        f"\nTotal time      : "
        f"{total_time:.4f} sec"
    )

    print(
        f"Throughput      : "
        f"{len(tiles) / total_time:.2f} "
        f"tiles/sec"
    )

    print(
        f"\nEmbeddings saved : "
        f"{output_path}"
    )

    print(
        f"Metadata saved   : "
        f"{metadata_path}"
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "✓ GIGAPATH-FLASH EXTRACTION SUCCESSFUL"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()