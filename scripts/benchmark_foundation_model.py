from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pathoverse.models.adapters import (
    TimmVisionAdapter,
    GigaPathFlashAdapter,
)
from pathoverse.models.benchmark import ModelBenchmark


MANIFEST = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

OUTPUT_DIR = Path(
    "results/benchmarks"
)

BATCH_SIZE = 8


def load_tiles():

    with MANIFEST.open() as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]

    if not tiles:
        raise ValueError(
            "No tiles found in manifest."
        )

    return tiles


def load_images(tiles):

    images = []

    for tile in tiles:

        path = Path(
            tile["image_path"]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing tile: {path}"
            )

        with Image.open(path) as image:

            images.append(
                image.convert("RGB")
            )

    return images


def create_adapter(model_name):

    if model_name == "vit":

        return TimmVisionAdapter(
            model_id="vit_base_patch16_224",
            device="cuda",
            batch_size=BATCH_SIZE,
            pretrained=True,
        )

    if model_name == "gigapath":

        return GigaPathFlashAdapter(
            device="cuda",
            batch_size=BATCH_SIZE,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "vit",
            "gigapath",
        ],
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        f"PATHOVERSE — SINGLE MODEL BENCHMARK: "
        f"{args.model.upper()}"
    )
    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    tiles = load_tiles()

    images = load_images(tiles)

    print(
        "Tiles:",
        len(tiles),
    )

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------

    adapter = create_adapter(
        args.model
    )

    print("\nLoading model...")

    adapter.load()

    info = adapter.info()

    print(
        "Model:",
        info.name,
    )

    print(
        "Embedding dimension:",
        info.embedding_dim,
    )

    # ----------------------------------------------------------
    # Prepare batches
    # ----------------------------------------------------------

    batches = []

    for start in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        batch_images = images[
            start:start + BATCH_SIZE
        ]

        batch = adapter.preprocess(
            batch_images
        )

        batches.append(batch)

    print(
        "Batches:",
        len(batches),
    )

    # ----------------------------------------------------------
    # Warm-up
    # ----------------------------------------------------------

    print("\nWarm-up...")

    for _ in range(3):

        adapter.encode(
            batches[0]
        )

    torch.cuda.synchronize()

    # ----------------------------------------------------------
    # Benchmark
    # ----------------------------------------------------------

    benchmark = ModelBenchmark(
        model_name=info.name,
        device="cuda",
    )

    result = benchmark.run(
        inference_fn=adapter.encode,
        batches=batches,
        num_samples=len(images),
        batch_size=BATCH_SIZE,
        embedding_dim=info.embedding_dim,
    )

    result_dict = {
        "model_name": result.model_name,
        "device": result.device,
        "num_samples": result.num_samples,
        "batch_size": result.batch_size,
        "embedding_dim": result.embedding_dim,
        "total_time_seconds": (
            result.total_time_seconds
        ),
        "average_latency_ms": (
            result.average_latency_ms
        ),
        "throughput_samples_per_second": (
            result.throughput_samples_per_second
        ),
        "peak_gpu_memory_mb": (
            result.peak_gpu_memory_mb
        ),
    }

    output_path = (
        OUTPUT_DIR
        / f"{args.model}_standard_benchmark.json"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:

        json.dump(
            result_dict,
            f,
            indent=2,
        )

    # ----------------------------------------------------------
    # Print
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK RESULT")
    print("=" * 70)

    print(
        f"\nModel       : "
        f"{result.model_name}"
    )

    print(
        f"Embedding   : "
        f"{result.embedding_dim}"
    )

    print(
        f"Latency     : "
        f"{result.average_latency_ms:.2f} ms/tile"
    )

    print(
        f"Throughput  : "
        f"{result.throughput_samples_per_second:.2f} "
        f"tiles/sec"
    )

    print(
        f"GPU memory  : "
        f"{result.peak_gpu_memory_mb:.2f} MB"
    )

    print(
        f"Total time  : "
        f"{result.total_time_seconds:.4f} sec"
    )

    print(
        f"\nSaved       : "
        f"{output_path}"
    )

    print(
        "\n✓ SINGLE MODEL BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()