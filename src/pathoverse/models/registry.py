from __future__ import annotations

from typing import Callable

from pathoverse.models.adapters.conch import ConchAdapter
from pathoverse.models.adapters.gigapath import GigaPathFlashAdapter
from pathoverse.models.adapters.hf_vision import HFVisionAdapter


class ModelRegistry:
    """
    Central registry for PathoVerse foundation model adapters.

    Canonical model IDs are used internally while aliases are
    supported for backward compatibility.

    Models are registered as factories, so importing the registry
    does not load any model weights.
    """

    def __init__(self) -> None:
        self._models: dict[str, Callable] = {}
        self._aliases: dict[str, str] = {}

    # ==========================================================
    # Registration
    # ==========================================================

    def register(
        self,
        name: str,
        factory: Callable,
    ) -> None:

        name = name.strip()

        if not name:
            raise ValueError(
                "Model name cannot be empty."
            )

        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        if name in self._aliases:
            raise ValueError(
                f"Model '{name}' conflicts with an alias."
            )

        self._models[name] = factory

    def register_alias(
        self,
        alias: str,
        canonical_name: str,
    ) -> None:

        alias = alias.strip()
        canonical_name = canonical_name.strip()

        if canonical_name not in self._models:
            raise KeyError(
                f"Cannot create alias '{alias}': "
                f"model '{canonical_name}' is not registered."
            )

        if alias in self._models:
            raise ValueError(
                f"Alias '{alias}' conflicts with a registered model."
            )

        if alias in self._aliases:
            raise ValueError(
                f"Alias '{alias}' is already registered."
            )

        self._aliases[alias] = canonical_name

    # ==========================================================
    # Resolution
    # ==========================================================

    def _resolve_name(
        self,
        name: str,
    ) -> str:

        if not isinstance(name, str):
            raise TypeError(
                "Model name must be a string."
            )

        value = name.strip()

        if value in self._models:
            return value

        if value in self._aliases:
            return self._aliases[value]

        lowered = value.lower()

        for registered_name in self._models:

            if registered_name.lower() == lowered:
                return registered_name

        for alias, canonical in self._aliases.items():

            if alias.lower() == lowered:
                return canonical

        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available models: "
            f"{', '.join(sorted(self._models))}"
        )

    # ==========================================================
    # Factory
    # ==========================================================

    def create(
        self,
        name: str,
        **kwargs,
    ):
        """
        Create an adapter instance.

        IMPORTANT:
        This does not load model weights.
        """

        canonical_name = self._resolve_name(
            name
        )

        return self._models[
            canonical_name
        ](
            **kwargs
        )

    # ==========================================================
    # Introspection
    # ==========================================================

    def list_models(
        self,
    ) -> list[str]:

        return sorted(
            self._models
        )

    def list_aliases(
        self,
    ) -> dict[str, str]:

        return dict(
            self._aliases
        )

    def resolve(
        self,
        name: str,
    ) -> str:

        return self._resolve_name(
            name
        )


# ============================================================
# GLOBAL REGISTRY
# ============================================================

registry = ModelRegistry()


# ============================================================
# CANONICAL MODELS
# ============================================================

registry.register(
    "vit-b-16",
    lambda **kwargs: HFVisionAdapter(
        model_id=(
            "google/"
            "vit-base-patch16-224-in21k"
        ),
        **kwargs,
    ),
)


registry.register(
    "gigapath-flash",
    lambda **kwargs: GigaPathFlashAdapter(
        **kwargs,
    ),
)


registry.register(
    "conch",
    lambda **kwargs: ConchAdapter(
        # PathoVerse canonical ID remains `conch`,
        # while the adapter keeps the official upstream
        # CONCH internal model identifier.
        model_id="conch_ViT-B-16",
        checkpoint_id=(
            "hf_hub:"
            "MahmoodLab/"
            "conch"
        ),
        **kwargs,
    ),
)


# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

# ViT
registry.register_alias(
    "vit-base",
    "vit-b-16",
)

registry.register_alias(
    "vit_base",
    "vit-b-16",
)

registry.register_alias(
    "vit_b_16",
    "vit-b-16",
)

registry.register_alias(
    "vit-b/16",
    "vit-b-16",
)

registry.register_alias(
    "ViT-B/16",
    "vit-b-16",
)

registry.register_alias(
    "ViT-B-16",
    "vit-b-16",
)


# GigaPath
registry.register_alias(
    "gigapath",
    "gigapath-flash",
)

registry.register_alias(
    "gigapath_flash",
    "gigapath-flash",
)

registry.register_alias(
    "GigaPath",
    "gigapath-flash",
)

registry.register_alias(
    "GigaPath-Flash",
    "gigapath-flash",
)


# CONCH
registry.register_alias(
    "CONCH",
    "conch",
)

registry.register_alias(
    "conch_ViT-B-16",
    "conch",
)

registry.register_alias(
    "conch-vit-b-16",
    "conch",
)