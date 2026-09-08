from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from huggingface_hub import get_token

from pathoverse.models.adapters.conch import ConchAdapter


INDEX_PATH = Path(
    "data/processed/embeddings/"
    "CMU-1-Small-Region_conch.faiss"
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
    print("PATHOVERSE — CONCH FAISS TEXT → IMAGE RETRIEVAL")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("PyTorch       :", torch.__version__)
    print("GPU           :", torch.cuda.get_device_name(0))

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    tiles = metadata["tiles"]

    print()
    print("FAISS dimension :", index.d)
    print("FAISS vectors   :", index.ntotal)
    print("Tile metadata   :", len(tiles))

    if index.ntotal != len(tiles):
        raise RuntimeError(
            "FAISS vector count does not match tile metadata."
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
    print("Running FAISS semantic retrieval...")

    for query in QUERIES:

        text_embedding = adapter.encode_text(
            [query]
        )

        query_vector = (
            text_embedding
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        faiss.normalize_L2(query_vector)

        scores, indices = index.search(
            query_vector,
            TOP_K,
        )

        print()
        print("=" * 70)
        print(f'QUERY: "{query}"')
        print("=" * 70)

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            tile = tiles[int(index_id)]

            print(
                f"Rank {rank} | "
                f"Tile {tile['tile_id']} | "
                f"Score {score:.4f} | "
                f"Location ({tile['x']}, {tile['y']}) | "
                f"Tissue {tile['tissue_ratio']:.3f}"
            )

    print()
    print("=" * 70)
    print("✓ CONCH FAISS RETRIEVAL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
