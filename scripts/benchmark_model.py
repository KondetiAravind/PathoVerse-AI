from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from pathoverse.models import ModelEngine
from pathoverse.tiles.dataset import PathoVerseTileDataset


MANIFEST = (
    "data/metadata/tiles/"
    "extracted_tiles.json"
)

OUTPUT = (
    "results/benchmarks/"
    "vit-base_cmu_small_region.json"
)


def main():

    print("=" * 72)
    print("PATHOVERSE AI — FOUNDATION MODEL BENCHMARK")
    print("=" * 72)

    device = "cuda"

    dataset = PathoVerseTileDataset(
        MANIFEST
    )

    batch_size = 8

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    engine = ModelEngine(
        model_name="vit-base",
        device=device,
        batch_size=batch_size,
    )

    print("Loading model...")

    engine.load()

    print("Model:", engine.info())

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    total_tiles = 0
    embedding_dim = None

    with torch.inference_mode():

        for batch in loader:

            images = batch["image"]

            image_list = [
                Image.fromarray(
                    (
                        image
                        .permute(1, 2, 0)
                        .numpy()
                        * 255
                    )
                    .clip(0, 255)
                    .astype("uint8")
                )
                for image in images
            ]

            embeddings = engine.encode_images(
                image_list
            )

            total_tiles += len(
                image_list
            )

            embedding_dim = embeddings.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    peak_memory_mb = 0.0

    if torch.cuda.is_available():

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    throughput = (
        total_tiles / elapsed
    )

    latency_ms = (
        elapsed / total_tiles
    ) * 1000

    result = {

        "model_name": "vit-base",

        "model_id":
            engine.info().model_id,

        "device":
            str(engine.adapter.device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),

        "num_tiles":
            total_tiles,

        "batch_size":
            batch_size,

        "embedding_dimension":
            embedding_dim,

        "total_time_seconds":
            elapsed,

        "average_latency_ms_per_tile":
            latency_ms,

        "throughput_tiles_per_second":
            throughput,

        "peak_gpu_memory_mb":
            peak_memory_mb,
    }

    Path(
        OUTPUT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    print(
        f"Tiles       : {total_tiles}"
    )

    print(
        f"Time        : {elapsed:.4f} sec"
    )

    print(
        f"Latency     : {latency_ms:.2f} ms/tile"
    )

    print(
        f"Throughput  : {throughput:.2f} tiles/sec"
    )

    print(
        f"GPU Memory  : {peak_memory_mb:.2f} MB"
    )

    print(
        f"Embedding   : {embedding_dim}"
    )

    print(
        f"\nSaved: {OUTPUT}"
    )

    print(
        "\n✓ FOUNDATION MODEL BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()