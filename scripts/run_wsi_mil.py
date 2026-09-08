from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pathoverse.mil.model import AttentionMIL
from pathoverse.mil.inference import (
    run_mil_inference,
)


# ============================================================
# PATHS
# ============================================================

EMBEDDING_DIR = Path(
    "data/processed/wsi/embeddings"
)

METADATA_DIR = Path(
    "data/metadata/wsi/embeddings"
)

RESULTS_DIR = Path(
    "results/mil"
)

MODEL_DIR = Path(
    "models/mil"
)


MODEL_DIMS = {

    "vit_b_16": 768,

    "gigapath_flash": 384,

    "conch": 512,

}


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run attention MIL on a WSI "
            "foundation embedding bag."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_DIMS.keys(),
    )

    parser.add_argument(
        "--slide-id",
        default="CMU-1-Small-Region",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Paths
    # ========================================================

    embedding_path = (
        EMBEDDING_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_embeddings.npy"
        )
    )

    metadata_path = (
        METADATA_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_embeddings.json"
        )
    )

    if not embedding_path.exists():

        raise FileNotFoundError(
            f"Embedding file not found: "
            f"{embedding_path}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{metadata_path}"
        )

    # ========================================================
    # Load
    # ========================================================

    embeddings = np.load(
        embedding_path
    )

    with open(
        metadata_path,
        "r",
    ) as f:

        metadata = json.load(
            f
        )

    embedding_dim = MODEL_DIMS[
        args.model
    ]

    if embeddings.shape[1] != embedding_dim:

        raise RuntimeError(
            f"Expected embedding dimension "
            f"{embedding_dim}, got "
            f"{embeddings.shape[1]}."
        )

    # ========================================================
    # Create MIL model
    # ========================================================

    print()
    print("=" * 72)

    print(
        "PATHOVERSE — WSI ATTENTION MIL"
    )

    print("=" * 72)

    print(
        f"Slide       : {args.slide_id}"
    )

    print(
        f"Foundation  : {args.model}"
    )

    print(
        f"Embeddings  : {embeddings.shape}"
    )

    print(
        f"Device      : {args.device}"
    )

    print("=" * 72)

    model = AttentionMIL(
        input_dim=embedding_dim,
        hidden_dim=256,
        attention_dim=128,
        num_classes=2,
        dropout=0.0,
    )

    # --------------------------------------------------------
    # Prototype note:
    #
    # There is currently no slide-level labeled training set.
    # Therefore this model is initialized deterministically
    # and used to demonstrate attention-based aggregation
    # and explainability.
    # --------------------------------------------------------

    model.eval()

    # ========================================================
    # Inference
    # ========================================================

    result = run_mil_inference(
        model=model,
        embeddings=embeddings,
        device=args.device,
        top_k=args.top_k,
    )

    # ========================================================
    # Tile metadata
    # ========================================================

    tile_metadata = metadata[
        "tiles"
    ]

    top_tiles = []

    for tile_index in result.top_tile_indices:

        tile = tile_metadata[
            tile_index
        ]

        top_tiles.append(
            {
                "rank": len(top_tiles) + 1,
                "tile_index": int(
                    tile_index
                ),
                "tile_id": int(
                    tile["tile_id"]
                ),
                "attention": float(
                    result.attention_weights[
                        tile_index
                    ]
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
        )

    # ========================================================
    # All tile attention
    # ========================================================

    all_tiles = []

    for index, tile in enumerate(
        tile_metadata
    ):

        all_tiles.append(
            {
                "tile_index": int(
                    index
                ),
                "tile_id": int(
                    tile["tile_id"]
                ),
                "attention": float(
                    result.attention_weights[
                        index
                    ]
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
        )

    # ========================================================
    # Save result
    # ========================================================

    output = {

        "schema_version": "1.0",

        "task": "wsi_attention_mil_prototype",

        "slide_id": args.slide_id,

        "foundation_model": {
            "name": args.model,
            "embedding_dimension": embedding_dim,
        },

        "mil": {
            "architecture": "Gated Attention MIL",
            "hidden_dimension": 256,
            "attention_dimension": 128,
            "num_classes": 2,
            "trained": False,
            "prototype": True,
            "seed": args.seed,
        },

        "prediction": {
            "class": int(
                result.prediction
            ),
            "probability": float(
                result.probability
            ),
        },

        "slide_embedding": {
            "dimension": int(
                len(
                    result.slide_embedding
                )
            ),
            "path": None,
        },

        "attention": {
            "tile_count": int(
                len(
                    result.attention_weights
                )
            ),
            "sum": float(
                np.sum(
                    result.attention_weights
                )
            ),
            "top_k": int(
                args.top_k
            ),
            "top_tiles": top_tiles,
            "all_tiles": all_tiles,
        },

    }

    output_path = (
        RESULTS_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_mil.json"
        )
    )

    slide_embedding_path = (
        RESULTS_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_slide_embedding.npy"
        )
    )

    np.save(
        slide_embedding_path,
        result.slide_embedding,
    )

    output[
        "slide_embedding"
    ][
        "path"
    ] = str(
        slide_embedding_path
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # Display
    # ========================================================

    print()
    print(
        "Slide prediction :",
        result.prediction,
    )

    print(
        "Probability       :",
        f"{result.probability:.4f}",
    )

    print(
        "Attention sum     :",
        f"{np.sum(result.attention_weights):.6f}",
    )

    print()
    print(
        "TOP ATTENTION TILES"
    )

    print(
        "-" * 72
    )

    for tile in top_tiles:

        print(
            f"Rank {tile['rank']:02d} "
            f"| Tile {tile['tile_id']:02d} "
            f"| Attention "
            f"{tile['attention']:.6f} "
            f"| "
            f"({tile['x']}, {tile['y']})"
        )

    print()
    print(
        "Saved:",
        output_path,
    )

    print(
        "Saved:",
        slide_embedding_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()