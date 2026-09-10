from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from pathoverse.models.registry import registry


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    name: str
    model_id: str
    embedding_dimension: int
    input_size: int
    modality: str
    description: str


# ============================================================
# MODEL CATALOG
# ============================================================

MODEL_DEFINITIONS = {
    "vit-b-16": ModelDefinition(
        key="vit-b-16",
        name="ViT-B/16",
        model_id=(
            "google/"
            "vit-base-patch16-224-in21k"
        ),
        embedding_dimension=768,
        input_size=224,
        modality="vision",
        description=(
            "Vision Transformer baseline foundation encoder."
        ),
    ),

    "gigapath-flash": ModelDefinition(
        key="gigapath-flash",
        name="GigaPath-Flash",
        model_id=(
            "prov-gigapath/"
            "gigapath-flash"
        ),
        embedding_dimension=384,
        input_size=224,
        modality="histopathology",
        description=(
            "Efficient pathology foundation encoder."
        ),
    ),

    "conch": ModelDefinition(
        key="conch",
        name="CONCH",
        model_id="MahmoodLab/CONCH",
        embedding_dimension=512,
        input_size=448,
        modality="vision-language",
        description=(
            "Pathology vision-language foundation model."
        ),
    ),
}


# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

MODEL_ALIASES = {
    # ViT
    "vit-base": "vit-b-16",
    "vit_base": "vit-b-16",
    "vit_b_16": "vit-b-16",
    "vit-b/16": "vit-b-16",
    "vit-b-16": "vit-b-16",
    "ViT-B/16": "vit-b-16",
    "ViT-B-16": "vit-b-16",

    # GigaPath
    "gigapath": "gigapath-flash",
    "gigapath_flash": "gigapath-flash",
    "GigaPath-Flash": "gigapath-flash",
    "GigaPath": "gigapath-flash",

    # CONCH
    "CONCH": "conch",
    "conch_ViT-B-16": "conch",
    "conch-vit-b-16": "conch",
}


# ============================================================
# MODEL ID NORMALIZATION
# ============================================================


def normalize_model_id(
    model: str,
) -> str:

    if not isinstance(
        model,
        str,
    ):
        raise TypeError(
            "Model ID must be a string."
        )

    value = model.strip()

    if not value:
        raise ValueError(
            "Model ID cannot be empty."
        )

    if value in MODEL_DEFINITIONS:
        return value

    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]

    lowered = value.lower()

    for canonical in MODEL_DEFINITIONS:

        if canonical.lower() == lowered:
            return canonical

    for alias, canonical in MODEL_ALIASES.items():

        if alias.lower() == lowered:
            return canonical

    return value


# ============================================================
# MODEL SERVICE
# ============================================================


class ModelService:
    """
    Production foundation-model lifecycle manager.

    Properties:

    - lazy model loading
    - in-memory caching
    - model reuse
    - explicit unloading
    - CUDA memory cleanup
    - thread-safe lifecycle operations
    - no model loading during API startup
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 4,
    ) -> None:

        requested_device = (
            device
            or os.getenv(
                "PATHOVERSE_DEVICE"
            )
        )

        if requested_device:
            self.device = requested_device

        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.batch_size = max(
            1,
            int(batch_size),
        )

        self._loaded_models: dict[
            str,
            Any,
        ] = {}

        self._load_errors: dict[
            str,
            str,
        ] = {}

        self._lock = threading.RLock()

    # ==========================================================
    # Catalog
    # ==========================================================

    def list_models(
        self,
    ) -> list[ModelDefinition]:

        return list(
            MODEL_DEFINITIONS.values()
        )

    def get_definition(
        self,
        key: str,
    ) -> ModelDefinition:

        canonical = normalize_model_id(
            key
        )

        if canonical not in MODEL_DEFINITIONS:

            raise KeyError(
                f"Unknown model '{key}'. "
                f"Available models: "
                f"{', '.join(sorted(MODEL_DEFINITIONS))}"
            )

        return MODEL_DEFINITIONS[
            canonical
        ]

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def is_loaded(
        self,
        key: str,
    ) -> bool:

        canonical = normalize_model_id(
            key
        )

        with self._lock:

            return canonical in (
                self._loaded_models
            )

    def loaded_models(
        self,
    ) -> list[str]:

        with self._lock:

            return sorted(
                self._loaded_models
            )

    def load(
        self,
        key: str,
    ) -> Any:
        """
        Lazily load and cache a foundation model.

        If the model is already loaded, the exact same
        adapter instance is returned.
        """

        definition = self.get_definition(
            key
        )

        canonical = definition.key

        with self._lock:

            existing = (
                self._loaded_models.get(
                    canonical
                )
            )

            if existing is not None:
                return existing

            self._load_errors.pop(
                canonical,
                None,
            )

            try:

                adapter = registry.create(
                    canonical,
                    device=self.device,
                    batch_size=self.batch_size,
                )

                adapter.load()

            except Exception as exc:

                self._load_errors[
                    canonical
                ] = str(exc)

                raise

            self._loaded_models[
                canonical
            ] = adapter

            return adapter

    def get(
        self,
        key: str,
        load: bool = False,
    ) -> Any | None:
        """
        Retrieve a cached adapter.

        load=False:
            Return None when not loaded.

        load=True:
            Lazily load the adapter.
        """

        canonical = normalize_model_id(
            key
        )

        with self._lock:

            adapter = (
                self._loaded_models.get(
                    canonical
                )
            )

        if adapter is not None:
            return adapter

        if load:
            return self.load(
                canonical
            )

        return None

    def unload(
        self,
        key: str,
    ) -> None:
        """
        Unload one PathoVerse model and release its resources.
        """

        canonical = normalize_model_id(
            key
        )

        # Validate first.
        self.get_definition(
            canonical
        )

        with self._lock:

            adapter = (
                self._loaded_models.pop(
                    canonical,
                    None,
                )
            )

            self._load_errors.pop(
                canonical,
                None,
            )

            if adapter is None:
                return

            unload_method = getattr(
                adapter,
                "unload",
                None,
            )

            if callable(
                unload_method
            ):

                try:
                    unload_method()
                except Exception:
                    # Resource cleanup should not leave the
                    # model cached. Continue with Python/CUDA
                    # cleanup even if adapter cleanup fails.
                    pass

            del adapter

            gc.collect()

            self._clear_cuda_cache()

    def unload_all(
        self,
    ) -> None:
        """
        Unload every currently cached PathoVerse model.
        """

        with self._lock:

            keys = list(
                self._loaded_models.keys()
            )

            for key in keys:
                self.unload(
                    key
                )

            self._clear_cuda_cache()

    # ==========================================================
    # CUDA / RESOURCE MANAGEMENT
    # ==========================================================

    @staticmethod
    def _clear_cuda_cache() -> None:

        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # ==========================================================
    # Status
    # ==========================================================

    def status(
        self,
        key: str,
    ) -> dict[str, Any]:

        definition = self.get_definition(
            key
        )

        canonical = definition.key

        with self._lock:

            loaded = (
                canonical
                in self._loaded_models
            )

            load_error = (
                self._load_errors.get(
                    canonical
                )
            )

        return {
            "key": definition.key,
            "name": definition.name,
            "model_id": definition.model_id,
            "embedding_dimension": (
                definition.embedding_dimension
            ),
            "input_size": (
                definition.input_size
            ),
            "modality": definition.modality,
            "description": definition.description,
            "device": self.device,
            "batch_size": self.batch_size,
            "loaded": loaded,
            "status": (
                "loaded"
                if loaded
                else (
                    "error"
                    if load_error
                    else "available"
                )
            ),
            "load_error": load_error,
        }

    def system_status(
        self,
    ) -> dict[str, Any]:

        cuda_available = (
            torch.cuda.is_available()
        )

        gpu_name = None
        gpu_index = None
        gpu_memory_total_mb = 0.0
        gpu_memory_allocated_mb = 0.0
        gpu_memory_reserved_mb = 0.0

        if cuda_available:

            try:

                gpu_index = (
                    torch.cuda.current_device()
                )

                gpu_name = (
                    torch.cuda.get_device_name(
                        gpu_index
                    )
                )

                properties = (
                    torch.cuda.get_device_properties(
                        gpu_index
                    )
                )

                gpu_memory_total_mb = (
                    properties.total_memory
                    / (1024 ** 2)
                )

                gpu_memory_allocated_mb = (
                    torch.cuda.memory_allocated(
                        gpu_index
                    )
                    / (1024 ** 2)
                )

                gpu_memory_reserved_mb = (
                    torch.cuda.memory_reserved(
                        gpu_index
                    )
                    / (1024 ** 2)
                )

            except Exception:
                pass

        with self._lock:

            loaded = sorted(
                self._loaded_models
            )

            errors = dict(
                self._load_errors
            )

        return {
            "device": self.device,
            "cuda_available": cuda_available,
            "gpu_index": gpu_index,
            "gpu": gpu_name,
            "gpu_memory_total_mb": (
                gpu_memory_total_mb
            ),
            "gpu_memory_allocated_mb": (
                gpu_memory_allocated_mb
            ),
            "gpu_memory_reserved_mb": (
                gpu_memory_reserved_mb
            ),
            "loaded_models": loaded,
            "loaded_model_count": len(
                loaded
            ),
            "load_errors": errors,
        }


# ============================================================
# GLOBAL SERVICE
# ============================================================

model_service = ModelService()