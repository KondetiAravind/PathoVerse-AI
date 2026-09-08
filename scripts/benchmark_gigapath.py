import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pathoverse.models.adapters import GigaPathFlashAdapter


MANIFEST = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

RESULT_PATH = Path(
    "results/benchmarks/"
    "gigapath-flash_cmu_small_region.json"
)

BATCH_SIZE = 8


def main():

    print("=" * 70)
    print("PATHOVERSE — GIGAPATH-FLASH BENCHMARK")
    print("=" * 70)

    with MANIFEST.open() as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]

    print("\nTiles       :", len(tiles))
    print("Batch size  :", BATCH_SIZE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )

    adapter = GigaPathFlashAdapter(
        device="cuda",
        batch_size=BATCH_SIZE,
    )

    print("\nLoading model...")

    adapter.load()

    info = adapter.info()

    print("Model       :", info.name)
    print("Embedding   :", info.embedding_dim)

    # --------------------------------------------------------------
    # Warmup
    # --------------------------------------------------------------

    first_image = Image.open(
        tiles[0]["image_path"]
    ).convert("RGB")

    warmup_batch = adapter.preprocess(
        [first_image] * min(BATCH_SIZE, len(tiles))
    )

    for _ in range(3):
        _ = adapter.encode(warmup_batch)

    torch.cuda.synchronize()

    # --------------------------------------------------------------
    # Reset memory statistics
    # --------------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

    embeddings = []

    start_time = time.perf_counter()

    for start in range(
        0,
        len(tiles),
        BATCH_SIZE
    ):

        batch_records = tiles[
            start:start + BATCH_SIZE
        ]

        images = []

        for tile in batch_records:

            image = Image.open(
                tile["image_path"]
            ).convert("RGB")

            images.append(image)

        batch = adapter.preprocess(images)

        output = adapter.encode(batch)

        embeddings.append(
            output.detach().cpu()
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    latency_ms = (
        elapsed
        / len(tiles)
        * 1000
    )

    throughput = (
        len(tiles)
        / elapsed
    )

    result = {
        "model": info.name,
        "model_id": info.model_id,
        "slide": "CMU-1-Small-Region",
        "tiles": len(tiles),
        "embedding_dim": info.embedding_dim,
        "batch_size": BATCH_SIZE,
        "total_time_seconds": elapsed,
        "latency_ms_per_tile": latency_ms,
        "throughput_tiles_per_second": throughput,
        "peak_gpu_memory_mb": peak_memory,
        "embedding_shape": list(
            embeddings.shape
        ),
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with RESULT_PATH.open("w") as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nTiles              : {len(tiles)}"
    )

    print(
        f"Embedding          : "
        f"{info.embedding_dim}"
    )

    print(
        f"Total time         : "
        f"{elapsed:.4f} sec"
    )

    print(
        f"Latency            : "
        f"{latency_ms:.2f} ms/tile"
    )

    print(
        f"Throughput         : "
        f"{throughput:.2f} tiles/sec"
    )

    print(
        f"Peak GPU memory    : "
        f"{peak_memory:.2f} MB"
    )

    print(
        f"\nSaved              : "
        f"{RESULT_PATH}"
    )

    print(
        "\n✓ GIGAPATH-FLASH BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()