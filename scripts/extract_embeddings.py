from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from pathoverse.models import (
    EmbeddingStore,
    ModelEngine,
)
from pathoverse.tiles.dataset import PathoVerseTileDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract foundation-model embeddings from pathology tiles."
    )

    parser.add_argument(
        "--manifest",
        default="data/metadata/tiles/extracted_tiles.json",
    )

    parser.add_argument(
        "--model",
        default="vit-base",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-name",
        default="CMU-1-Small-Region_vit-base",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 72)
    print("PATHOVERSE AI — FOUNDATION MODEL EMBEDDING EXTRACTION")
    print("=" * 72)

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    print(f"Manifest : {manifest_path}")
    print(f"Model    : {args.model}")
    print(f"Batch    : {args.batch_size}")
    print(f"Device   : {args.device}")

    dataset = PathoVerseTileDataset(
        manifest_path=str(manifest_path),
    )

    print(f"Tiles    : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    engine = ModelEngine(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("\nLoading model...")
    engine.load()

    print("Model loaded.")
    print("Model info:", engine.info())

    all_embeddings = []
    all_metadata = []

    print("\nExtracting embeddings...")

    for batch_index, batch in enumerate(loader):

        images = batch["image"]

        image_list = [
            Image.fromarray(
                (
                    image.permute(1, 2, 0).numpy() * 255
                ).clip(0, 255).astype("uint8")
            )
            for image in images
        ]

        embeddings = engine.encode_images(
            image_list
        )

        all_embeddings.append(
            embeddings
        )

        batch_size_actual = images.shape[0]

        for i in range(batch_size_actual):

            all_metadata.append(
                {
                    "tile_id": int(batch["tile_id"][i]),
                    "slide_id": batch["slide_id"][i],
                    "x": int(batch["x"][i]),
                    "y": int(batch["y"][i]),
                    "level": int(batch["level"][i]),
                    "tissue_ratio": float(batch["tissue_ratio"][i]),
                    "image_path": batch["image_path"][i],
                }
            )

        print(
            f"Batch {batch_index + 1:02d} | "
            f"{len(all_metadata):03d}/{len(dataset)} tiles"
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    print("\n" + "=" * 72)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 72)

    print(
        "Embedding shape:",
        tuple(embeddings.shape),
    )

    print(
        "Embedding dtype:",
        embeddings.dtype,
    )

    print(
        "Embedding range:",
        float(embeddings.min()),
        "to",
        float(embeddings.max()),
    )

    store = EmbeddingStore()

    embedding_path, metadata_path = store.save(
        embeddings=embeddings,
        metadata=all_metadata,
        name=args.output_name,
    )

    print("\nSaved:")
    print("Embeddings:", embedding_path)
    print("Metadata  :", metadata_path)

    print("\n✓ FOUNDATION MODEL EMBEDDINGS SAVED")


if __name__ == "__main__":
    main()