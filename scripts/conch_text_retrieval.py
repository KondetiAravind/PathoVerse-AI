from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token

from pathoverse.models.adapters.conch import ConchAdapter


EMBEDDINGS_PATH = Path(
    "data/processed/embeddings/"
    "CMU-1-Small-Region_conch.npy"
)

METADATA_PATH = Path(
    "data/metadata/embeddings/"
    "CMU-1-Small-Region_conch.json"
)

TOP_K = 5

QUERIES = [
    "tumor tissue",
    "normal tissue",
    "lymphocyte-rich tissue",
    "necrotic tissue",
    "epithelial tissue",
]


def main() -> None:
    print("=" * 70)
    print("PATHOVERSE — CONCH TEXT → IMAGE RETRIEVAL")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("PyTorch       :", torch.__version__)
    print("GPU           :", torch.cuda.get_device_name(0))

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    image_embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype(np.float32)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    tiles = metadata["tiles"]

    print()
    print("Image embeddings :", image_embeddings.shape)
    print("Tile metadata    :", len(tiles))

    if image_embeddings.ndim != 2:
        raise RuntimeError(
            "Expected a 2-D embedding matrix."
        )

    if image_embeddings.shape[0] != len(tiles):
        raise RuntimeError(
            "Embedding count does not match tile metadata."
        )

    if image_embeddings.shape[1] != 512:
        raise RuntimeError(
            f"Expected 512-D embeddings, "
            f"got {image_embeddings.shape[1]}"
        )

    # CONCH image embeddings are already normalized.
    image_norms = np.linalg.norm(
        image_embeddings,
        axis=1,
    )

    if not np.allclose(
        image_norms,
        1.0,
        atol=1e-4,
    ):
        print(
            "Warning: image embeddings are not perfectly "
            "normalized. Normalizing them now."
        )

        image_embeddings = (
            image_embeddings
            / np.maximum(
                image_norms[:, None],
                1e-12,
            )
        )

    token = get_token()

    adapter = ConchAdapter(
        device="cuda",
        batch_size=4,
        hf_auth_token=token,
    )

    print()
    print("Loading CONCH...")
    adapter.load()

    print()
    print("Running text queries...")

    for query in QUERIES:

        text_embedding = adapter.encode_text(
            [query]
        )

        text_vector = (
            text_embedding[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Both image and text embeddings are normalized.
        scores = image_embeddings @ text_vector

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        print()
        print("=" * 70)
        print(f'QUERY: "{query}"')
        print("=" * 70)

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):
            tile = tiles[int(index)]
            score = float(scores[index])

            print(
                f"Rank {rank} | "
                f"Tile {tile['tile_id']} | "
                f"Score {score:.4f} | "
                f"Location ({tile['x']}, {tile['y']}) | "
                f"Tissue {tile['tissue_ratio']:.3f}"
            )

    print()
    print("=" * 70)
    print("✓ CONCH TEXT → IMAGE RETRIEVAL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
