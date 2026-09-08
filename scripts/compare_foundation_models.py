from __future__ import annotations

import json
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

OUTPUT = Path(
    "results/benchmarks/"
    "foundation_model_comparison.json"
)

BATCH_SIZE = 8


def load_tiles():
    with MANIFEST.open() as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]

    if not tiles:
        raise ValueError("No tiles found.")

    return tiles


def load_images(tiles):
    images = []

    for tile in tiles:

        image_path = Path(
            tile["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Tile not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        images.append(image)

    return images


def make_batches(adapter, images):
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

    return batches


def benchmark_model(
    adapter,
    model_name,
    images,
):

    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {model_name}")
    print("=" * 70)

    adapter.load()

    info = adapter.info()

    print(
        "\nModel          :",
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

    batches = make_batches(
        adapter,
        images,
    )

    # ----------------------------------------------------------
    # Warm-up
    # ----------------------------------------------------------

    print("\nWarm-up...")

    warmup_batch = batches[0]

    for _ in range(3):
        adapter.encode(
            warmup_batch
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ----------------------------------------------------------
    # Unified benchmark
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

    print("\nResults:")
    print(
        f"Latency        : "
        f"{result.average_latency_ms:.2f} ms/tile"
    )

    print(
        f"Throughput     : "
        f"{result.throughput_samples_per_second:.2f} "
        f"tiles/sec"
    )

    print(
        f"GPU memory     : "
        f"{result.peak_gpu_memory_mb:.2f} MB"
    )

    print(
        f"Total time     : "
        f"{result.total_time_seconds:.4f} sec"
    )

    return result


def result_to_dict(result):

    return {
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


def main():

    print("=" * 70)
    print(
        "PATHOVERSE — FOUNDATION MODEL "
        "COMPARISON"
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

    print(
        "Tiles:",
        len(tiles),
    )

    images = load_images(tiles)

    # ----------------------------------------------------------
    # ViT-B/16
    # ----------------------------------------------------------

    vit_adapter = TimmVisionAdapter(
        model_id="vit_base_patch16_224",
        device="cuda",
        batch_size=BATCH_SIZE,
        pretrained=True,
    )

    vit_result = benchmark_model(
        vit_adapter,
        "ViT-B/16",
        images,
    )

    # ----------------------------------------------------------
    # GigaPath-Flash
    # ----------------------------------------------------------

    gigapath_adapter = (
        GigaPathFlashAdapter(
            device="cuda",
            batch_size=BATCH_SIZE,
        )
    )

    gigapath_result = benchmark_model(
        gigapath_adapter,
        "GigaPath-Flash",
        images,
    )

    # ----------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------

    results = [
        result_to_dict(
            vit_result
        ),
        result_to_dict(
            gigapath_result
        ),
    ]

    vit = results[0]
    giga = results[1]

    comparison = {
        "dataset": {
            "slide": (
                "CMU-1-Small-Region"
            ),
            "num_tiles": len(tiles),
            "batch_size": BATCH_SIZE,
        },
        "models": results,
        "comparison": {
            "throughput_speedup_x": (
                giga[
                    "throughput_samples_per_second"
                ]
                / vit[
                    "throughput_samples_per_second"
                ]
            ),
            "latency_reduction_x": (
                vit[
                    "average_latency_ms"
                ]
                / giga[
                    "average_latency_ms"
                ]
            ),
            "gpu_memory_reduction_percent": (
                (
                    1
                    -
                    giga[
                        "peak_gpu_memory_mb"
                    ]
                    /
                    vit[
                        "peak_gpu_memory_mb"
                    ]
                )
                * 100
            ),
        },
    }

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w") as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    # ----------------------------------------------------------
    # Final report
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FOUNDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        "\nModel              "
        "Embedding   "
        "Latency      "
        "Throughput      "
        "GPU Memory"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['model_name']:<19}"
            f"{result['embedding_dim']:<12}"
            f"{result['average_latency_ms']:<13.2f}"
            f"{result['throughput_samples_per_second']:<16.2f}"
            f"{result['peak_gpu_memory_mb']:.2f} MB"
        )

    print("\nGigaPath-Flash vs ViT-B/16:")

    print(
        f"Throughput speedup : "
        f"{comparison['comparison']['throughput_speedup_x']:.2f}×"
    )

    print(
        f"Latency advantage  : "
        f"{comparison['comparison']['latency_reduction_x']:.2f}×"
    )

    print(
        f"GPU memory reduced : "
        f"{comparison['comparison']['gpu_memory_reduction_percent']:.2f}%"
    )

    print(
        f"\nSaved: {OUTPUT}"
    )

    print(
        "\n✓ FOUNDATION MODEL COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()