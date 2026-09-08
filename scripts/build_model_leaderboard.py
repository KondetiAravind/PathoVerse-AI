from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_DIR = Path(
    "results/benchmarks"
)

OUTPUT = (
    BENCHMARK_DIR
    / "foundation_model_leaderboard.json"
)


BENCHMARK_FILES = [
    "vit_standard_benchmark.json",
    "gigapath_standard_benchmark.json",
]


def load_results():

    results = []

    for filename in BENCHMARK_FILES:

        path = BENCHMARK_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Benchmark file not found: {path}"
            )

        with path.open() as f:
            results.append(json.load(f))

    return results


def main():

    print("=" * 75)
    print(
        "PATHOVERSE — FOUNDATION MODEL LEADERBOARD"
    )
    print("=" * 75)

    results = load_results()

    # ----------------------------------------------------------
    # Rank models
    # ----------------------------------------------------------

    by_throughput = sorted(
        results,
        key=lambda x: x[
            "throughput_samples_per_second"
        ],
        reverse=True,
    )

    by_latency = sorted(
        results,
        key=lambda x: x[
            "average_latency_ms"
        ],
    )

    by_memory = sorted(
        results,
        key=lambda x: x[
            "peak_gpu_memory_mb"
        ],
    )

    leaderboard = []

    for rank, result in enumerate(
        by_throughput,
        start=1,
    ):

        leaderboard.append(
            {
                "rank": rank,
                "model_name": result[
                    "model_name"
                ],
                "embedding_dim": result[
                    "embedding_dim"
                ],
                "latency_ms": result[
                    "average_latency_ms"
                ],
                "throughput_tiles_per_second": (
                    result[
                        "throughput_samples_per_second"
                    ]
                ),
                "peak_gpu_memory_mb": result[
                    "peak_gpu_memory_mb"
                ],
            }
        )

    # ----------------------------------------------------------
    # Pairwise comparison
    # ----------------------------------------------------------

    if len(results) >= 2:

        baseline = results[0]
        best = by_throughput[0]

        throughput_speedup = (
            best[
                "throughput_samples_per_second"
            ]
            /
            baseline[
                "throughput_samples_per_second"
            ]
        )

        latency_ratio = (
            baseline[
                "average_latency_ms"
            ]
            /
            best[
                "average_latency_ms"
            ]
        )

        memory_reduction = (
            (
                1
                -
                best[
                    "peak_gpu_memory_mb"
                ]
                /
                baseline[
                    "peak_gpu_memory_mb"
                ]
            )
            * 100
        )

    else:

        throughput_speedup = None
        latency_ratio = None
        memory_reduction = None

    # ----------------------------------------------------------
    # Final artifact
    # ----------------------------------------------------------

    output = {
        "schema_version": "1.0",

        "benchmark_configuration": {
            "num_models": len(results),
            "num_tiles": results[0][
                "num_samples"
            ],
            "batch_size": results[0][
                "batch_size"
            ],
            "device": results[0][
                "device"
            ],
        },

        "leaderboard": leaderboard,

        "rankings": {
            "fastest_throughput": (
                by_throughput[0][
                    "model_name"
                ]
            ),
            "lowest_latency": (
                by_latency[0][
                    "model_name"
                ]
            ),
            "lowest_gpu_memory": (
                by_memory[0][
                    "model_name"
                ]
            ),
        },

        "comparison": {
            "throughput_speedup_x": (
                throughput_speedup
            ),
            "latency_advantage_x": (
                latency_ratio
            ),
            "gpu_memory_reduction_percent": (
                memory_reduction
            ),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w") as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ----------------------------------------------------------
    # Display
    # ----------------------------------------------------------

    print(
        "\nModel Leaderboard"
    )

    print("-" * 75)

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'Dim':<8}"
        f"{'Latency':<14}"
        f"{'Throughput':<16}"
        f"{'GPU Memory'}"
    )

    print("-" * 75)

    for row in leaderboard:

        print(
            f"{row['rank']:<6}"
            f"{row['model_name']:<22}"
            f"{row['embedding_dim']:<8}"
            f"{row['latency_ms']:<14.2f}"
            f"{row['throughput_tiles_per_second']:<16.2f}"
            f"{row['peak_gpu_memory_mb']:.2f} MB"
        )

    print(
        "\nRankings:"
    )

    print(
        "Fastest throughput :",
        output["rankings"][
            "fastest_throughput"
        ],
    )

    print(
        "Lowest latency      :",
        output["rankings"][
            "lowest_latency"
        ],
    )

    print(
        "Lowest GPU memory   :",
        output["rankings"][
            "lowest_gpu_memory"
        ],
    )

    if throughput_speedup is not None:

        print(
            "\nBest-vs-baseline:"
        )

        print(
            f"Throughput speedup : "
            f"{throughput_speedup:.2f}×"
        )

        print(
            f"Latency advantage  : "
            f"{latency_ratio:.2f}×"
        )

        print(
            f"Memory reduction   : "
            f"{memory_reduction:.2f}%"
        )

    print(
        f"\nSaved: {OUTPUT}"
    )

    print(
        "\n✓ FOUNDATION MODEL LEADERBOARD COMPLETE"
    )


if __name__ == "__main__":
    main()