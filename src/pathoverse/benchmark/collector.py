from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import BenchmarkRecord


class BenchmarkCollector:
    """
    Collect existing PathoVerse experiment artifacts
    into a common benchmark representation.
    """

    def __init__(
        self,
        benchmark_dir: str | Path,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.records: list[BenchmarkRecord] = []

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:

        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark artifact not found: {path}"
            )

        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _number(value):

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ========================================================
    # FOUNDATION MODEL EFFICIENCY
    # ========================================================

    def add_efficiency_result(
        self,
        path: str | Path,
    ):

        path = Path(path)
        data = self._load_json(path)

        # ----------------------------------------------------
        # ViT / GigaPath format
        # ----------------------------------------------------

        if "model_name" in data:

            model_name = data["model_name"]

            model_id = model_name

            embedding_dim = data.get(
                "embedding_dim"
            )

            samples = data.get(
                "num_samples"
            )

            latency_ms = data.get(
                "average_latency_ms"
            )

            throughput = data.get(
                "throughput_samples_per_second"
            )

            gpu_memory = data.get(
                "peak_gpu_memory_mb"
            )

            total_time = data.get(
                "total_time_seconds"
            )

        # ----------------------------------------------------
        # CONCH format
        # ----------------------------------------------------

        elif "model" in data:

            model = data.get(
                "model",
                {}
            )

            benchmark = data.get(
                "benchmark",
                {}
            )

            model_name = model.get(
                "name",
                "unknown"
            )

            model_id = model.get(
                "model_id",
                model_name
            )

            embedding_dim = model.get(
                "embedding_dim"
            )

            samples = benchmark.get(
                "tiles"
            )

            latency_ms = benchmark.get(
                "latency_ms_per_tile"
            )

            throughput = benchmark.get(
                "throughput_tiles_per_sec"
            )

            gpu_memory = benchmark.get(
                "peak_gpu_memory_mb"
            )

            total_time = benchmark.get(
                "total_inference_time_sec"
            )

        else:

            raise ValueError(
                f"Unsupported efficiency schema: {path}"
            )

        record = BenchmarkRecord(

            model=str(
                model_name
            ),

            model_id=str(
                model_id
            ),

            task="foundation_efficiency",

            embedding_dimension=(
                int(embedding_dim)
                if embedding_dim is not None
                else None
            ),

            samples=(
                int(samples)
                if samples is not None
                else None
            ),

            latency_ms=self._number(
                latency_ms
            ),

            throughput=self._number(
                throughput
            ),

            gpu_memory_mb=self._number(
                gpu_memory
            ),

            inference_time_sec=self._number(
                total_time
            ),

            status="completed",

            metadata={
                "source_file": str(path),
                "device": data.get(
                    "device",
                    data.get(
                        "hardware",
                        {}
                    ).get(
                        "device"
                    )
                ),
                "batch_size": data.get(
                    "batch_size",
                    data.get(
                        "benchmark",
                        {}
                    ).get(
                        "batch_size"
                    )
                ),
            },
        )

        self.records.append(record)

    # ========================================================
    # IMAGE RETRIEVAL
    # ========================================================

    def add_retrieval_result(
        self,
        path: str | Path,
    ):

        path = Path(path)
        data = self._load_json(path)

        model = data.get(
            "model",
            {}
        )

        dataset = data.get(
            "dataset",
            {}
        )

        overall = data.get(
            "overall",
            {}
        )

        if not model:

            raise ValueError(
                f"Missing model section: {path}"
            )

        if not overall:

            raise ValueError(
                f"Missing overall retrieval metrics: {path}"
            )

        model_name = model.get(
            "name",
            "unknown"
        )

        model_id = model.get(
            "model_id",
            model_name
        )

        embedding_dim = model.get(
            "embedding_dimension"
        )

        record = BenchmarkRecord(

            model=str(
                model_name
            ),

            model_id=str(
                model_id
            ),

            task="image_retrieval",

            embedding_dimension=(
                int(embedding_dim)
                if embedding_dim is not None
                else None
            ),

            dataset=dataset.get(
                "slide_id",
                "CMU-1-Small-Region"
            ),

            samples=dataset.get(
                "queries"
            ),

            recall_at_1=self._number(
                overall.get(
                    "recall_at_1"
                )
            ),

            recall_at_5=self._number(
                overall.get(
                    "recall_at_5"
                )
            ),

            recall_at_10=self._number(
                overall.get(
                    "recall_at_10"
                )
            ),

            status="completed",

            metadata={
                "source_file": str(path),

                "slide_id": dataset.get(
                    "slide_id"
                ),

                "tiles": dataset.get(
                    "tiles"
                ),

                "query_count": dataset.get(
                    "queries"
                ),

                "protocol": data.get(
                    "protocol"
                ),

                "ground_truth": dataset.get(
                    "ground_truth"
                ),
            },
        )

        self.records.append(record)

    # ========================================================
    # PCAM CLASSIFICATION
    # ========================================================

    def add_classification_result(
        self,
        path: str | Path,
    ):

        path = Path(path)
        data = self._load_json(path)

        model_info = data.get(
            "model",
            {}
        )

        test = data.get(
            "test",
            {}
        )

        dataset_info = data.get(
            "dataset",
            {}
        )

        model_name = model_info.get(
            "name",
            "unknown"
        )

        model_id = model_info.get(
            "model_id",
            model_name
        )

        embedding_dim = model_info.get(
            "embedding_dimension",
            model_info.get(
                "embedding_dim"
            )
        )

        test_samples = dataset_info.get(
            "test_samples"
        )

        record = BenchmarkRecord(

            model=str(
                model_name
            ),

            model_id=str(
                model_id
            ),

            task="classification",

            embedding_dimension=(
                int(embedding_dim)
                if embedding_dim is not None
                else None
            ),

            dataset=dataset_info.get(
                "name",
                "PatchCamelyon"
            ),

            samples=(
                int(test_samples)
                if test_samples is not None
                else None
            ),

            accuracy=self._number(
                test.get(
                    "accuracy"
                )
            ),

            auroc=self._number(
                test.get(
                    "auroc"
                )
            ),

            f1=self._number(
                test.get(
                    "f1"
                )
            ),

            precision=self._number(
                test.get(
                    "precision"
                )
            ),

            recall=self._number(
                test.get(
                    "recall"
                )
            ),

            sensitivity=self._number(
                test.get(
                    "sensitivity"
                )
            ),

            specificity=self._number(
                test.get(
                    "specificity"
                )
            ),

            inference_time_sec=self._number(
                data.get(
                    "inference",
                    {}
                ).get(
                    "test_evaluation_time_sec"
                )
            ),

            status="completed",

            metadata={
                "source_file": str(path),

                "foundation_model_frozen": (
                    model_info.get(
                        "foundation_model_frozen"
                    )
                ),
            },
        )

        self.records.append(record)

    # ========================================================
    # WSI MIL PROTOTYPE
    # ========================================================

    def add_mil_result(
        self,
        path: str | Path,
    ):

        path = Path(path)
        data = self._load_json(path)

        foundation = data.get(
            "foundation_model",
            {}
        )

        model_name = foundation.get(
            "name",
            "unknown"
        )

        # Normalize model naming across all
        # PathoVerse benchmark artifacts.
        model_name_normalized = {
            "gigapath_flash": "GigaPath-Flash",
            "gigapath-flash": "GigaPath-Flash",
            "GigaPath Flash": "GigaPath-Flash",
        }.get(
            str(model_name),
            str(model_name),
        )

        embedding_dim = foundation.get(
            "embedding_dimension",
            foundation.get(
                "embedding_dim"
            )
        )

        attention = data.get(
            "attention",
            {}
        )

        tile_count = attention.get(
            "tile_count"
        )

        record = BenchmarkRecord(

            model=model_name_normalized,

            model_id=model_name_normalized,

            task="wsi_mil_prototype",

            embedding_dimension=(
                int(embedding_dim)
                if embedding_dim is not None
                else None
            ),

            dataset="WSI",

            samples=(
                int(tile_count)
                if tile_count is not None
                else None
            ),

            status="prototype",

            metadata={
                "source_file": str(path),

                "slide_id": data.get(
                    "slide_id"
                ),

                "trained": data.get(
                    "mil",
                    {}
                ).get(
                    "trained",
                    False
                ),
            },
        )

        self.records.append(record)

    # ========================================================
    # EXPORT
    # ========================================================

    def to_list(
        self,
    ) -> list[dict[str, Any]]:

        return [
            record.to_dict()
            for record in self.records
        ]