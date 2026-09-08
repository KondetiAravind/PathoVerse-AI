from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_DIR = Path("results/benchmarks")

OUTPUT_PATH = (
    BENCHMARK_DIR / "foundation_model_leaderboard.json"
)

FILES = [
    "vit_standard_benchmark.json",
    "gigapath_standard_benchmark.json",
    "conch_standard_benchmark.json",
]


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_benchmark(data: dict, filename: str) -> dict:
    """
    Normalize both historical and current benchmark schemas
    into one common internal representation.

    Supported schemas:

    Legacy:
        {
            "model_name": ...,
            "num_samples": ...,
            "batch_size": ...,
            "embedding_dim": ...,
            "total_time_seconds": ...,
            "average_latency_ms": ...,
            "throughput_samples_per_second": ...,
            "peak_gpu_memory_mb": ...
        }

    Current:
        {
            "model": {...},
            "hardware": {...},
            "benchmark": {...},
            "validation": {...}
        }
    """

    # ======================================================
    # Legacy benchmark schema
    # ======================================================

    if "model_name" in data:

        return {
            "model": data["model_name"],
            "model_id": data["model_name"],
            "embedding_dim": int(
                data["embedding_dim"]
            ),
            "latency_ms_per_tile": float(
                data["average_latency_ms"]
            ),
            "throughput_tiles_per_sec": float(
                data["throughput_samples_per_second"]
            ),
            "peak_gpu_memory_mb": float(
                data["peak_gpu_memory_mb"]
            ),
            "tiles": int(
                data["num_samples"]
            ),
            "batch_size": int(
                data["batch_size"]
            ),
            "total_time_sec": float(
                data["total_time_seconds"]
            ),
            "source_file": filename,
        }

    # ======================================================
    # Current benchmark schema
    # ======================================================

    if (
        "model" in data
        and "benchmark" in data
    ):

        model = data["model"]
        benchmark = data["benchmark"]

        return {
            "model": model["name"],
            "model_id": model["model_id"],
            "embedding_dim": int(
                model["embedding_dim"]
            ),
            "latency_ms_per_tile": float(
                benchmark["latency_ms_per_tile"]
            ),
            "throughput_tiles_per_sec": float(
                benchmark["throughput_tiles_per_sec"]
            ),
            "peak_gpu_memory_mb": float(
                benchmark["peak_gpu_memory_mb"]
            ),
            "tiles": int(
                benchmark["tiles"]
            ),
            "batch_size": int(
                benchmark["batch_size"]
            ),
            "total_time_sec": float(
                benchmark["total_inference_time_sec"]
            ),
            "source_file": filename,
        }

    raise ValueError(
        f"Unsupported benchmark schema in {filename}. "
        f"Top-level keys: {list(data.keys())}"
    )


def main() -> None:

    print("=" * 82)
    print(
        "PATHOVERSE — FOUNDATION MODEL LEADERBOARD"
    )
    print("=" * 82)

    # ======================================================
    # Load + normalize
    # ======================================================

    rows = []

    for filename in FILES:

        path = BENCHMARK_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark file: {path}"
            )

        raw = load_json(path)

        normalized = normalize_benchmark(
            raw,
            filename,
        )

        rows.append(normalized)

    # ======================================================
    # Validate benchmark consistency
    # ======================================================

    tile_counts = {
        row["tiles"]
        for row in rows
    }

    if len(tile_counts) != 1:
        raise ValueError(
            "Benchmark tile counts do not match: "
            f"{tile_counts}"
        )

    # ======================================================
    # Rankings
    # ======================================================

    throughput_rank = sorted(
        rows,
        key=lambda row: row[
            "throughput_tiles_per_sec"
        ],
        reverse=True,
    )

    latency_rank = sorted(
        rows,
        key=lambda row: row[
            "latency_ms_per_tile"
        ],
    )

    memory_rank = sorted(
        rows,
        key=lambda row: row[
            "peak_gpu_memory_mb"
        ],
    )

    # ======================================================
    # ViT-B/16 baseline
    # ======================================================

    baseline = next(
        (
            row
            for row in rows
            if row["model"] == "ViT-B/16"
        ),
        None,
    )

    if baseline is None:
        raise ValueError(
            "ViT-B/16 benchmark is required "
            "as the baseline."
        )

    # ======================================================
    # Relative metrics
    # ======================================================

    for row in rows:

        row["throughput_speedup_vs_vit"] = (
            row["throughput_tiles_per_sec"]
            / baseline[
                "throughput_tiles_per_sec"
            ]
        )

        row["latency_ratio_vs_vit"] = (
            row["latency_ms_per_tile"]
            / baseline[
                "latency_ms_per_tile"
            ]
        )

        row["memory_ratio_vs_vit"] = (
            row["peak_gpu_memory_mb"]
            / baseline[
                "peak_gpu_memory_mb"
            ]
        )

        row["memory_reduction_vs_vit_percent"] = (
            1
            - row[
                "peak_gpu_memory_mb"
            ]
            / baseline[
                "peak_gpu_memory_mb"
            ]
        ) * 100.0

    # ======================================================
    # Final leaderboard object
    # ======================================================

    leaderboard = {
        "schema_version": "1.0",

        "benchmark_protocol": {
            "description": (
                "Standardized fresh-process "
                "single-model inference benchmark "
                "over the same extracted pathology "
                "tile set."
            ),
            "dataset": "CMU-1-Small-Region",
            "tiles": rows[0]["tiles"],
            "hardware": (
                "NVIDIA RTX PRO 4500 Blackwell"
            ),
            "device": "cuda:0",
        },

        "models": rows,

        "rankings": {
            "fastest_throughput": (
                throughput_rank[0]["model"]
            ),
            "lowest_latency": (
                latency_rank[0]["model"]
            ),
            "lowest_gpu_memory": (
                memory_rank[0]["model"]
            ),
        },

        "baseline": {
            "model": baseline["model"],

            "throughput_speedup": {
                row["model"]: row[
                    "throughput_speedup_vs_vit"
                ]
                for row in rows
            },

            "latency_ratio": {
                row["model"]: row[
                    "latency_ratio_vs_vit"
                ]
                for row in rows
            },

            "memory_ratio": {
                row["model"]: row[
                    "memory_ratio_vs_vit"
                ]
                for row in rows
            },

            "memory_reduction_percent": {
                row["model"]: row[
                    "memory_reduction_vs_vit_percent"
                ]
                for row in rows
            },
        },
    }

    # ======================================================
    # Save
    # ======================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
    ) as f:

        json.dump(
            leaderboard,
            f,
            indent=2,
        )

    # ======================================================
    # Display
    # ======================================================

    print()

    print(
        f"{'Rank':<6}"
        f"{'Model':<20}"
        f"{'Dim':>8}"
        f"{'Latency':>15}"
        f"{'Throughput':>18}"
        f"{'GPU Memory':>17}"
    )

    print("-" * 82)

    for rank, row in enumerate(
        throughput_rank,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{row['model']:<20}"
            f"{row['embedding_dim']:>8}"
            f"{row['latency_ms_per_tile']:>11.2f} ms"
            f"{row['throughput_tiles_per_sec']:>14.2f}/s"
            f"{row['peak_gpu_memory_mb']:>13.2f} MB"
        )

    print()
    print("Rankings")
    print("-" * 82)

    print(
        "Fastest throughput :",
        throughput_rank[0]["model"],
    )

    print(
        "Lowest latency      :",
        latency_rank[0]["model"],
    )

    print(
        "Lowest GPU memory   :",
        memory_rank[0]["model"],
    )

    print()
    print("Against ViT-B/16 baseline")
    print("-" * 82)

    for row in rows:

        print(
            f"{row['model']:<20}"
            f"Throughput: "
            f"{row['throughput_speedup_vs_vit']:.2f}x | "
            f"Latency: "
            f"{row['latency_ratio_vs_vit']:.2f}x | "
            f"Memory: "
            f"{row['memory_ratio_vs_vit']:.2f}x"
        )

    print()
    print("Saved:")
    print(" ", OUTPUT_PATH)

    print("=" * 82)


if __name__ == "__main__":
    main()
