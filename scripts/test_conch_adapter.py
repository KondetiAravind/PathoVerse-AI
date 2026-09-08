from __future__ import annotations

import time

import torch
from PIL import Image

from huggingface_hub import get_token

from pathoverse.models.adapters.conch import ConchAdapter


IMAGE_PATH = (
    "data/processed/tiles/"
    "CMU-1-Small-Region_L0_X000896_Y000224.png"
)


def main() -> None:
    print("=" * 70)
    print("PATHOVERSE — CONCH ADAPTER TEST")
    print("=" * 70)

    print("PyTorch       :", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU           :", torch.cuda.get_device_name(0))

    token = get_token()

    adapter = ConchAdapter(
        device="cuda",
        batch_size=4,
        hf_auth_token=token,
    )

    print()
    print("Loading CONCH adapter...")

    start = time.perf_counter()

    adapter.load()

    load_time = time.perf_counter() - start

    print()
    print("Model loaded")
    print("Load time     :", f"{load_time:.2f} sec")
    print("Model info    :", adapter.info())

    image = Image.open(IMAGE_PATH).convert("RGB")

    print()
    print("Encoding image...")

    image_embedding = adapter.encode([image])

    print()
    print("IMAGE EMBEDDING")
    print("-" * 70)
    print("Shape         :", tuple(image_embedding.shape))
    print("Dtype         :", image_embedding.dtype)
    print("Device        :", image_embedding.device)
    print("Finite        :", bool(torch.isfinite(image_embedding).all()))
    print(
        "L2 norm       :",
        float(torch.linalg.norm(image_embedding[0])),
    )

    texts = [
        "tumor tissue",
        "normal tissue",
        "lymphocyte-rich tissue",
        "necrotic tissue",
        "epithelial tissue",
    ]

    print()
    print("Encoding text prompts...")

    text_embedding = adapter.encode_text(texts)

    print()
    print("TEXT EMBEDDINGS")
    print("-" * 70)
    print("Shape         :", tuple(text_embedding.shape))
    print("Dtype         :", text_embedding.dtype)
    print("Device        :", text_embedding.device)
    print("Finite        :", bool(torch.isfinite(text_embedding).all()))

    similarity = image_embedding @ text_embedding.T

    print()
    print("IMAGE ↔ TEXT SIMILARITY")
    print("-" * 70)

    scores = similarity[0].detach().cpu().tolist()

    ranked = sorted(
        zip(texts, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (text, score) in enumerate(ranked, start=1):
        print(
            f"Rank {rank} | "
            f"{text:<35} | "
            f"Score {score:.4f}"
        )

    print()
    print("=" * 70)
    print("✓ CONCH ADAPTER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
