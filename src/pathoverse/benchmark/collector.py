from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# CANONICAL MODEL IDENTIFIERS
# ============================================================

CANONICAL_MODEL_IDS = {
    "vit-b-16",
    "gigapath-flash",
    "conch",
}


MODEL_ALIASES = {
    # ViT
    "vit-base": "vit-b-16",
    "vit_b_16": "vit-b-16",
    "vit-b/16": "vit-b-16",
    "vit-b-16": "vit-b-16",
    "ViT-B/16": "vit-b-16",
    "ViT-B-16": "vit-b-16",

    # GigaPath
    "gigapath_flash": "gigapath-flash",
    "gigapath-flash": "gigapath-flash",
    "GigaPath-Flash": "gigapath-flash",

    # CONCH
    "CONCH": "conch",
    "conch": "conch",
    "conch_ViT-B-16": "conch",
    "conch-vit-b-16": "conch",
}


MODEL_DISPLAY_NAMES = {
    "vit-b-16": "ViT-B/16",
    "gigapath-flash": "GigaPath-Flash",
    "conch": "CONCH",
}


# ============================================================
# HELPERS
# ============================================================


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# BENCHMARK RECORD
# ============================================================


@dataclass
class BenchmarkRecord:

    model: str
    model_id: str

    task: str
    dataset: str
    split: str

    samples: int | None = None

    embedding_dimension: int | None = None

    accuracy: float | None = None
    auroc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None

    recall_at_1: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None

    latency_ms: float | None = None
    throughput: float | None = None
    gpu_memory_mb: float | None = None

    status: str = "completed"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_id": self.model_id,
            "task": self.task,
            "dataset": self.dataset,
            "split": self.split,
            "samples": self.samples,
            "embedding_dimension": self.embedding_dimension,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "gpu_memory_mb": self.gpu_memory_mb,
            "status": self.status,
            "metadata": self.metadata,
        }


# ============================================================
# BENCHMARK COLLECTOR
# ============================================================


class BenchmarkCollector:

    def __init__(
        self,
        root_dir: str | Path | None = None,
    ) -> None:

        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else Path(".")
        )

        self.records: list[BenchmarkRecord] = []

    # ========================================================
    # PATH RESOLUTION
    # ========================================================

    def _resolve_path(
        self,
        data: str | Path,
    ) -> Path:

        path = Path(data)

        if path.is_absolute():
            return path

        # Prefer the path exactly as supplied from CWD.
        # This prevents results/results/... when root_dir=results.
        direct_path = Path.cwd() / path

        if direct_path.exists():
            return direct_path

        return self.root_dir / path

    # ========================================================
    # INPUT / FILE HANDLING
    # ========================================================

    def _load_data(
        self,
        data: dict[str, Any] | str | Path,
    ) -> dict[str, Any]:

        if isinstance(data, dict):
            return data

        path = self._resolve_path(data)

        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Benchmark path is not a file: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                loaded = json.load(f)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON benchmark file: {path}"
            ) from exc

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected JSON object in benchmark file: {path}"
            )

        return loaded

    # ========================================================
    # MODEL IDENTIFICATION
    # ========================================================

    @staticmethod
    def _canonical_model_id(
        model_id: Any = None,
        model_name: Any = None,
        source_path: str | Path | None = None,
    ) -> str:

        candidates = [
            _clean_string(model_id),
            _clean_string(model_name),
        ]

        alias_lookup = {
            str(key).lower(): value
            for key, value in MODEL_ALIASES.items()
        }

        for candidate in candidates:

            if candidate is None:
                continue

            if candidate in CANONICAL_MODEL_IDS:
                return candidate

            if candidate in MODEL_ALIASES:
                return MODEL_ALIASES[candidate]

            lowered = candidate.lower()

            if lowered in CANONICAL_MODEL_IDS:
                return lowered

            if lowered in alias_lookup:
                return alias_lookup[lowered]

        for candidate in candidates:

            if candidate is None:
                continue

            lowered = candidate.lower()

            if "gigapath" in lowered:
                return "gigapath-flash"

            if "conch" in lowered:
                return "conch"

            if (
                "vit" in lowered
                or "vision transformer" in lowered
            ):
                return "vit-b-16"

        if source_path is not None:

            filename = Path(
                source_path
            ).name.lower()

            if (
                "gigapath_flash" in filename
                or "gigapath-flash" in filename
                or "gigapath" in filename
            ):
                return "gigapath-flash"

            if "conch" in filename:
                return "conch"

            if (
                "vit_b_16" in filename
                or "vit-b-16" in filename
                or "vit_base" in filename
                or "vit-base" in filename
                or "vit" in filename
            ):
                return "vit-b-16"

        fallback = _clean_string(model_id)

        if fallback is not None:
            return fallback

        fallback = _clean_string(model_name)

        if fallback is not None:
            return fallback

        raise ValueError(
            "Unable to resolve benchmark model ID. "
            f"model_id={model_id}, "
            f"model_name={model_name}, "
            f"source_path={source_path}"
        )

    @staticmethod
    def _get_model_display_name(
        canonical_model_id: str,
        original_model_name: Any = None,
    ) -> str:

        if canonical_model_id in MODEL_DISPLAY_NAMES:
            return MODEL_DISPLAY_NAMES[
                canonical_model_id
            ]

        original = _clean_string(
            original_model_name
        )

        if original is not None:
            return original

        return canonical_model_id

    # ========================================================
    # GENERIC ADD
    # ========================================================

    def add(
        self,
        *,
        model: Any,
        task: str,
        dataset: str,
        split: str,
        samples: int | None = None,
        embedding_dimension: int | None = None,
        accuracy: float | None = None,
        auroc: float | None = None,
        f1: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        recall_at_1: float | None = None,
        recall_at_5: float | None = None,
        recall_at_10: float | None = None,
        latency_ms: float | None = None,
        throughput: float | None = None,
        gpu_memory_mb: float | None = None,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
        source_path: str | Path | None = None,
    ) -> BenchmarkRecord:

        model_id = None
        model_name = None

        if isinstance(model, str):

            model_id = model
            model_name = model

        elif isinstance(model, dict):

            model_id = _first_not_none(
                model.get("canonical_model_id"),
                model.get("model_id"),
                model.get("id"),
            )

            model_name = _first_not_none(
                model.get("name"),
                model.get("model_name"),
            )

        else:

            model_id = _first_not_none(
                getattr(
                    model,
                    "canonical_model_id",
                    None,
                ),
                getattr(
                    model,
                    "model_id",
                    None,
                ),
            )

            model_name = getattr(
                model,
                "name",
                None,
            )

        canonical_id = self._canonical_model_id(
            model_id=model_id,
            model_name=model_name,
            source_path=source_path,
        )

        display_name = self._get_model_display_name(
            canonical_id,
            model_name,
        )

        record = BenchmarkRecord(
            model=display_name,
            model_id=canonical_id,
            task=task,
            dataset=dataset,
            split=split,
            samples=samples,
            embedding_dimension=embedding_dimension,
            accuracy=accuracy,
            auroc=auroc,
            f1=f1,
            precision=precision,
            recall=recall,
            recall_at_1=recall_at_1,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            latency_ms=latency_ms,
            throughput=throughput,
            gpu_memory_mb=gpu_memory_mb,
            status=status,
            metadata=metadata or {},
        )

        self.records.append(record)

        return record

    # ========================================================
    # FOUNDATION MODEL EFFICIENCY
    # ========================================================

    def add_efficiency_result(
        self,
        data: dict[str, Any] | str | Path,
    ) -> BenchmarkRecord:

        source_path = (
            self._resolve_path(data)
            if not isinstance(data, dict)
            else None
        )

        data = self._load_data(data)

        model_data = data.get(
            "model",
            {},
        )

        if not isinstance(
            model_data,
            dict,
        ):
            model_data = {}

        # Current PathoVerse foundation benchmark artifacts
        # store runtime metrics under `benchmark`.
        benchmark_data = data.get(
            "benchmark",
            {},
        )

        if not isinstance(
            benchmark_data,
            dict,
        ):
            benchmark_data = {}

        model_id = _first_not_none(
            data.get("canonical_model_id"),
            data.get("model_id"),
            model_data.get("canonical_model_id"),
            model_data.get("model_id"),
            model_data.get("id"),
        )

        model_name = _first_not_none(
            data.get("model_name"),
            data.get("name"),
            model_data.get("name"),
            model_data.get("model_name"),
        )

        return self.add(
            model={
                "model_id": model_id,
                "name": model_name,
            },
            task="foundation_efficiency",
            dataset=_clean_string(
                data.get(
                    "dataset",
                    "WSI",
                )
            ) or "WSI",
            split=_clean_string(
                data.get(
                    "split",
                    "benchmark",
                )
            ) or "benchmark",
            samples=_to_int(
                _first_not_none(
                    data.get("samples"),
                    data.get("num_samples"),
                    data.get("tile_count"),
                    benchmark_data.get("tiles"),
                )
            ),
            embedding_dimension=_to_int(
                _first_not_none(
                    data.get("embedding_dimension"),
                    data.get("embedding_dim"),
                    model_data.get("embedding_dimension"),
                    model_data.get("embedding_dim"),
                )
            ),
            latency_ms=_to_float(
                _first_not_none(
                    data.get("latency_ms"),
                    data.get("average_latency_ms"),
                    data.get("mean_latency_ms"),
                    benchmark_data.get("latency_ms_per_tile"),
                )
            ),
            throughput=_to_float(
                _first_not_none(
                    data.get("throughput"),
                    data.get("throughput_samples_per_second"),
                    data.get("tiles_per_second"),
                    benchmark_data.get("throughput_tiles_per_sec"),
                )
            ),
            gpu_memory_mb=_to_float(
                _first_not_none(
                    data.get("gpu_memory_mb"),
                    data.get("peak_gpu_memory_mb"),
                    benchmark_data.get("peak_gpu_memory_mb"),
                )
            ),
            status=_clean_string(
                data.get(
                    "status",
                    "completed",
                )
            ) or "completed",
            source_path=source_path,
        )

    # ========================================================
    # IMAGE RETRIEVAL
    # ========================================================

    def add_retrieval_result(
        self,
        data: dict[str, Any] | str | Path,
    ) -> BenchmarkRecord:

        source_path = (
            self._resolve_path(data)
            if not isinstance(data, dict)
            else None
        )

        data = self._load_data(data)

        model_data = data.get(
            "model",
            {},
        )

        if not isinstance(
            model_data,
            dict,
        ):
            model_data = {}

        model_id = _first_not_none(
            data.get("canonical_model_id"),
            data.get("model_id"),
            model_data.get("canonical_model_id"),
            model_data.get("model_id"),
            model_data.get("id"),
        )

        model_name = _first_not_none(
            data.get("model_name"),
            data.get("name"),
            model_data.get("name"),
            model_data.get("model_name"),
        )

        overall = data.get(
            "overall",
            {},
        )

        if not isinstance(
            overall,
            dict,
        ):
            overall = {}

        metrics = data.get(
            "metrics",
            {},
        )

        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}

        evaluation = data.get(
            "evaluation",
            {},
        )

        if not isinstance(
            evaluation,
            dict,
        ):
            evaluation = {}

        results = data.get(
            "results",
            {},
        )

        if not isinstance(
            results,
            dict,
        ):
            results = {}

        def retrieval_metric(
            *keys: str,
        ) -> float | None:

            for key in keys:

                converted = _to_float(
                    overall.get(key)
                )

                if converted is not None:
                    return converted

            for container in (
                metrics,
                evaluation,
                results,
                data,
            ):

                for key in keys:

                    converted = _to_float(
                        container.get(key)
                    )

                    if converted is not None:
                        return converted

            return None

        dataset_data = data.get(
            "dataset",
            {},
        )

        if isinstance(
            dataset_data,
            dict,
        ):

            dataset_name = (
                _clean_string(
                    dataset_data.get("name")
                )
                or "WSI"
            )

            samples = _to_int(
                _first_not_none(
                    dataset_data.get("queries"),
                    dataset_data.get("tiles"),
                    dataset_data.get("samples"),
                )
            )

        else:

            dataset_name = (
                str(dataset_data)
                if dataset_data
                else "WSI"
            )

            samples = None

        protocol = data.get(
            "protocol",
            {},
        )

        if not isinstance(
            protocol,
            dict,
        ):
            protocol = {}

        split = (
            _clean_string(
                data.get("split")
            )
            or _clean_string(
                protocol.get("split")
            )
            or "benchmark"
        )

        metadata = {
            "source_file": (
                str(source_path)
                if source_path is not None
                else None
            ),
            "schema_version": data.get(
                "schema_version"
            ),
        }

        return self.add(
            model={
                "model_id": model_id,
                "name": model_name,
            },
            task="image_retrieval",
            dataset=dataset_name,
            split=split,
            samples=samples,
            embedding_dimension=_to_int(
                _first_not_none(
                    data.get("embedding_dimension"),
                    data.get("embedding_dim"),
                    model_data.get("embedding_dimension"),
                    model_data.get("embedding_dim"),
                )
            ),
            recall_at_1=retrieval_metric(
                "recall_at_1",
                "R@1",
                "r@1",
            ),
            recall_at_5=retrieval_metric(
                "recall_at_5",
                "R@5",
                "r@5",
            ),
            recall_at_10=retrieval_metric(
                "recall_at_10",
                "R@10",
                "r@10",
            ),
            metadata=metadata,
            source_path=source_path,
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def add_classification_result(
        self,
        data: dict[str, Any] | str | Path,
    ) -> BenchmarkRecord:

        source_path = (
            self._resolve_path(data)
            if not isinstance(data, dict)
            else None
        )

        data = self._load_data(data)

        model_data = data.get(
            "model",
            {},
        )

        if not isinstance(
            model_data,
            dict,
        ):
            model_data = {}

        model_id = _first_not_none(
            data.get("canonical_model_id"),
            data.get("model_id"),
            model_data.get("canonical_model_id"),
            model_data.get("model_id"),
        )

        model_name = _first_not_none(
            data.get("model_name"),
            data.get("name"),
            model_data.get("name"),
            model_data.get("model_name"),
        )

        metrics = data.get(
            "metrics",
            {},
        )

        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}

        dataset_data = data.get(
            "dataset",
            {},
        )

        if isinstance(
            dataset_data,
            dict,
        ):

            dataset_name = dataset_data.get(
                "name",
                "PatchCamelyon",
            )

            dataset_samples = dataset_data.get(
                "test_samples"
            )

        else:

            dataset_name = (
                str(dataset_data)
                if dataset_data
                else "PatchCamelyon"
            )

            dataset_samples = None

        test_data = data.get(
            "test",
            {},
        )

        if not isinstance(
            test_data,
            dict,
        ):
            test_data = {}

        def metric(
            *keys: str,
        ):

            values = []

            for key in keys:
                values.append(
                    data.get(key)
                )

            for key in keys:
                values.append(
                    metrics.get(key)
                )

            for key in keys:
                values.append(
                    test_data.get(key)
                )

            return _first_not_none(
                *values
            )

        return self.add(
            model={
                "model_id": model_id,
                "name": model_name,
            },
            task="classification",
            dataset=str(
                dataset_name
            ),
            split="test",
            samples=_to_int(
                _first_not_none(
                    data.get("samples"),
                    dataset_samples,
                    test_data.get("samples"),
                )
            ),
            embedding_dimension=_to_int(
                _first_not_none(
                    data.get("embedding_dimension"),
                    data.get("embedding_dim"),
                    model_data.get("embedding_dimension"),
                    model_data.get("embedding_dim"),
                )
            ),
            accuracy=_to_float(
                metric("accuracy")
            ),
            auroc=_to_float(
                metric(
                    "auroc",
                    "roc_auc",
                )
            ),
            f1=_to_float(
                metric("f1")
            ),
            precision=_to_float(
                metric("precision")
            ),
            recall=_to_float(
                metric(
                    "recall",
                    "sensitivity",
                )
            ),
            source_path=source_path,
        )

    # ========================================================
    # WSI MIL
    # ========================================================

    def add_mil_result(
        self,
        data: dict[str, Any] | str | Path,
    ) -> BenchmarkRecord:

        source_path = (
            self._resolve_path(data)
            if not isinstance(data, dict)
            else None
        )

        data = self._load_data(data)

        model_data = data.get(
            "model",
            {},
        )

        if not isinstance(
            model_data,
            dict,
        ):
            model_data = {}

        model_id = _first_not_none(
            data.get("canonical_model_id"),
            data.get("model_id"),
            model_data.get("canonical_model_id"),
            model_data.get("model_id"),
        )

        model_name = _first_not_none(
            data.get("model_name"),
            data.get("name"),
            model_data.get("name"),
            model_data.get("model_name"),
        )

        if (
            model_id is None
            and model_name is None
            and source_path is not None
        ):

            filename = source_path.name.lower()

            if (
                "gigapath_flash" in filename
                or "gigapath-flash" in filename
                or "gigapath" in filename
            ):
                model_id = "gigapath-flash"

            elif "conch" in filename:
                model_id = "conch"

            elif (
                "vit_b_16" in filename
                or "vit-b-16" in filename
                or "vit_base" in filename
                or "vit-base" in filename
            ):
                model_id = "vit-b-16"

        return self.add(
            model={
                "model_id": model_id,
                "name": model_name,
            },
            task="wsi_mil_prototype",
            dataset=_clean_string(
                data.get(
                    "dataset",
                    "WSI",
                )
            ) or "WSI",
            split=_clean_string(
                data.get(
                    "split",
                    "prototype",
                )
            ) or "prototype",
            samples=_to_int(
                _first_not_none(
                    data.get("samples"),
                    data.get("tile_count"),
                    data.get("tiles"),
                )
            ),
            embedding_dimension=_to_int(
                _first_not_none(
                    data.get("embedding_dimension"),
                    data.get("embedding_dim"),
                    model_data.get("embedding_dimension"),
                    model_data.get("embedding_dim"),
                )
            ),
            status="prototype",
            metadata={
                "source_file": (
                    str(source_path)
                    if source_path is not None
                    else None
                ),
                "schema_version": data.get(
                    "schema_version"
                ),
            },
            source_path=source_path,
        )

    # ========================================================
    # BULK
    # ========================================================

    def extend(
        self,
        records: Iterable[BenchmarkRecord],
    ) -> None:

        self.records.extend(
            records
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        require_production_models: bool = False,
    ) -> None:

        if not self.records:
            raise ValueError(
                "Benchmark collector contains no records."
            )

        for record in self.records:

            if not record.model_id:
                raise ValueError(
                    "Benchmark record has no model_id."
                )

            if not record.task:
                raise ValueError(
                    "Benchmark record has no task."
                )

            if not record.dataset:
                raise ValueError(
                    "Benchmark record has no dataset."
                )

        if not require_production_models:
            return

        required = {
            (
                "vit-b-16",
                "foundation_efficiency",
            ),
            (
                "gigapath-flash",
                "foundation_efficiency",
            ),
            (
                "conch",
                "foundation_efficiency",
            ),
            (
                "vit-b-16",
                "image_retrieval",
            ),
            (
                "gigapath-flash",
                "image_retrieval",
            ),
            (
                "conch",
                "image_retrieval",
            ),
            (
                "vit-b-16",
                "classification",
            ),
            (
                "gigapath-flash",
                "classification",
            ),
            (
                "conch",
                "classification",
            ),
            (
                "gigapath-flash",
                "wsi_mil_prototype",
            ),
        }

        actual = {
            (
                record.model_id,
                record.task,
            )
            for record in self.records
        }

        missing = sorted(
            required - actual
        )

        if missing:

            formatted = ", ".join(
                f"{model}/{task}"
                for model, task in missing
            )

            raise ValueError(
                "Unified benchmark is missing "
                f"required production records: "
                f"{formatted}"
            )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_list(
        self,
    ) -> list[dict[str, Any]]:

        return [
            record.to_dict()
            for record in self.records
        ]

    # ========================================================
    # UTILITIES
    # ========================================================

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)
