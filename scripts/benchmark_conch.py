from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from huggingface_hub import get_token
from PIL import Image

from pathoverse.models.adapters.conch import ConchAdapter


TILES_MANIFEST = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

OUTPUT_PATH = Path(
    "results/benchmarks/conch_standard_benchmark.json"
)

BATCH_SIZE = 4


def main() -> None:
    print("=" * 70)
    print("PATHOVERSE — STANDARD CONCH BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("PyTorch       :", torch.__version__)
    print("GPU           :", torch.cuda.get_device_name(0))
    print("Batch size    :", BATCH_SIZE)

    with open(TILES_MANIFEST, "r") as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]

    print("Tiles         :", len(tiles))

    token = get_token()

    adapter = ConchAdapter(
        device="cuda",
        batch_size=BATCH_SIZE,
        hf_auth_token=token,
    )

    print()
    print("Loading CONCH...")

    load_start = time.perf_counter()

    adapter.load()

    load_time = time.perf_counter() - load_start

    print(
        "Load time     :",
        f"{load_time:.2f} sec",
    )

    # ------------------------------------------------------
    # Prepare all images before benchmark.
    # This keeps disk I/O outside the measured inference time.
    # ------------------------------------------------------

    print()
    print("Preparing images...")

    images = []

    for tile in tiles:
        image = Image.open(
            tile["image_path"]
        ).convert("RGB")

        images.append(image)

    print("Prepared      :", len(images))

    # ------------------------------------------------------
    # Warmup
    # ------------------------------------------------------

    print()
    print("Running warmup...")

    with torch.inference_mode():
        warmup_images = images[:BATCH_SIZE]
        adapter.encode(warmup_images)

    torch.cuda.synchronize()

    # Clear previous allocator state before measurement.
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------

    print("Running benchmark...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    all_embeddings = []

    with torch.inference_mode():

        for batch_start in range(
            0,
            len(images),
            BATCH_SIZE,
        ):

            batch_images = images[
                batch_start :
                batch_start + BATCH_SIZE
            ]

            embeddings = adapter.encode(
                batch_images
            )

            all_embeddings.append(
                embeddings.detach()
            )

    torch.cuda.synchronize()

    total_time = time.perf_counter() - start

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    tile_count = len(images)

    latency_ms = (
        total_time / tile_count
    ) * 1000.0

    throughput = (
        tile_count / total_time
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if embeddings.shape != (
        tile_count,
        512,
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{tuple(embeddings.shape)}"
        )

    if not torch.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            "Embeddings contain NaN or Inf."
        )

    norms = torch.linalg.norm(
        embeddings,
        dim=1,
    )

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-4,
    ):
        raise RuntimeError(
            "CONCH embeddings are not normalized."
        )

    result = {
        "model": {
            "name": "CONCH",
            "model_id": "conch_ViT-B-16",
            "embedding_dim": 512,
            "input_size": 448,
            "modality": "vision-language",
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "device": "cuda:0",
            "pytorch": torch.__version__,
        },
        "benchmark": {
            "tiles": tile_count,
            "batch_size": BATCH_SIZE,
            "load_time_sec": load_time,
            "total_inference_time_sec": total_time,
            "latency_ms_per_tile": latency_ms,
            "throughput_tiles_per_sec": throughput,
            "peak_gpu_memory_mb": peak_memory,
        },
        "validation": {
            "embedding_shape": list(
                embeddings.shape
            ),
            "finite": True,
            "normalized": True,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("CONCH BENCHMARK COMPLETE")
    print("=" * 70)

    print("Model         : CONCH")
    print("Embedding     : 512")
    print("Tiles         :", tile_count)
    print(
        "Total time    :",
        f"{total_time:.4f} sec",
    )
    print(
        "Latency       :",
        f"{latency_ms:.2f} ms/tile",
    )
    print(
        "Throughput    :",
        f"{throughput:.2f} tiles/sec",
    )
    print(
        "Peak GPU      :",
        f"{peak_memory:.2f} MB",
    )

    print()
    print("Saved:")
    print(" ", OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
