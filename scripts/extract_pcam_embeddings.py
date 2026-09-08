from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pathoverse.datasets.patchcamelyon import PatchCamelyonDataset
from pathoverse.models.adapters.conch import ConchAdapter
from pathoverse.models.adapters.gigapath import GigaPathFlashAdapter
from pathoverse.models.adapters.timm_vision import TimmVisionAdapter


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "data/raw/patchcamelyon"
)

OUTPUT_DIR = Path(
    "data/processed/patchcamelyon/embeddings"
)

METADATA_DIR = Path(
    "data/metadata/patchcamelyon"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {

    "vit_b_16": {
        "factory": lambda device: TimmVisionAdapter(
            model_id="vit_base_patch16_224",
            device=device,
            batch_size=32,
        ),
    },

    "gigapath_flash": {
        "factory": lambda device: GigaPathFlashAdapter(
            device=device,
            batch_size=32,
        ),
    },

    "conch": {
        "factory": lambda device: ConchAdapter(
            device=device,
            batch_size=16,
        ),
    },

}


# ============================================================
# DATA SAMPLING
# ============================================================

def select_indices(
    dataset_size: int,
    requested_count: int,
    seed: int,
) -> np.ndarray:
    """
    Select a deterministic random subset of dataset indices.

    The same seed produces the same subset.
    """

    count = min(
        requested_count,
        dataset_size,
    )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        dataset_size,
        size=count,
        replace=False,
    )

    return np.sort(
        indices
    )


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(
    adapter,
    dataset,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract foundation-model embeddings for selected PCam samples.

    Important:
    The dataset returns PIL images.
    Each model adapter performs its own official preprocessing.

    Returns
    -------
    embeddings:
        Float32 array of shape (N, embedding_dim)

    labels:
        Int64 array of shape (N,)

    elapsed:
        Total extraction time in seconds.
    """

    embeddings = []
    labels = []

    total = len(indices)

    start_time = time.perf_counter()

    batch_size = adapter.batch_size

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        batch_images = []
        batch_labels = []

        # ----------------------------------------------------
        # Load images from HDF5-backed dataset
        # ----------------------------------------------------

        for index in batch_indices:

            image, label = dataset[
                int(index)
            ]

            # IMPORTANT:
            # Do NOT apply torchvision transforms here.
            #
            # The individual model adapters handle their own
            # preprocessing:
            #
            # ViT       -> 224 x 224 + ImageNet normalization
            # GigaPath  -> official GigaPath transform
            # CONCH     -> 448 x 448 + OpenAI normalization

            batch_images.append(
                image
            )

            batch_labels.append(
                int(label)
            )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.inference_mode():

            batch_embeddings = adapter.encode(
                batch_images
            )

        batch_embeddings = (
            batch_embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        embeddings.append(
            batch_embeddings
        )

        labels.extend(
            batch_labels
        )

        processed = min(
            start + batch_size,
            total,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if elapsed > 0:

            throughput = (
                processed / elapsed
            )

        else:

            throughput = 0.0

        print(
            f"\rProcessed "
            f"{processed}/{total} "
            f"| {throughput:.2f} samples/sec",
            end="",
            flush=True,
        )

    print()

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return (
        embeddings,
        labels,
        elapsed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen foundation-model "
            "embeddings from PatchCamelyon."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIGS.keys(),
        help=(
            "Foundation model to use."
        ),
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=10000,
        help=(
            "Number of training samples "
            "to embed."
        ),
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=2000,
        help=(
            "Number of validation samples "
            "to embed."
        ),
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=2000,
        help=(
            "Number of test samples "
            "to embed."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "Torch device. Default: cuda:0"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for "
            "deterministic sampling."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    # ========================================================
    # Output directories
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Model configuration
    # ========================================================

    config = MODEL_CONFIGS[
        args.model
    ]

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PATHOVERSE — PCAM FOUNDATION EMBEDDINGS"
    )
    print("=" * 72)

    print(
        f"Model  : {args.model}"
    )

    print(
        f"Device : {args.device}"
    )

    print(
        f"Seed   : {args.seed}"
    )

    print("=" * 72)

    # ========================================================
    # Dataset
    # ========================================================

    dataset = PatchCamelyonDataset(
        ROOT
    )

    print(
        "PCam dataset loaded."
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Loading foundation model..."
    )

    model_load_start = time.perf_counter()

    adapter = config["factory"](
        args.device
    )

    adapter.ensure_loaded()

    model_load_time = (
        time.perf_counter()
        - model_load_start
    )

    model_info = adapter.info()

    print(
        f"Model name        : "
        f"{model_info.name}"
    )

    print(
        f"Model ID          : "
        f"{model_info.model_id}"
    )

    print(
        f"Embedding dim     : "
        f"{model_info.embedding_dim}"
    )

    print(
        f"Input size        : "
        f"{model_info.input_size}"
    )

    print(
        f"Model load time   : "
        f"{model_load_time:.2f} sec"
    )

    # ========================================================
    # Split configuration
    # ========================================================

    split_sizes = {

        "train": args.train_samples,

        "validation": args.val_samples,

        "test": args.test_samples,

    }

    all_results = {}

    # ========================================================
    # Process each split
    # ========================================================

    for split, requested_count in split_sizes.items():

        print()
        print("-" * 72)

        print(
            f"Processing split: {split}"
        )

        print("-" * 72)

        # ----------------------------------------------------
        # Get split
        # ----------------------------------------------------

        base_dataset = dataset.get_split(
            split
        )

        dataset_size = len(
            base_dataset
        )

        count = min(
            requested_count,
            dataset_size,
        )

        print(
            f"Available samples : "
            f"{dataset_size}"
        )

        print(
            f"Requested samples : "
            f"{requested_count}"
        )

        print(
            f"Selected samples  : "
            f"{count}"
        )

        # ----------------------------------------------------
        # Deterministic sampling
        # ----------------------------------------------------

        indices = select_indices(
            dataset_size=dataset_size,
            requested_count=count,
            seed=args.seed,
        )

        print(
            f"First indices     : "
            f"{indices[:10].tolist()}"
        )

        # ----------------------------------------------------
        # Extract embeddings
        # ----------------------------------------------------

        embeddings, labels, extraction_time = (
            extract_embeddings(
                adapter=adapter,
                dataset=base_dataset,
                indices=indices,
            )
        )

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        if embeddings.ndim != 2:

            raise RuntimeError(
                "Embedding output must be "
                "a 2D array."
            )

        if len(embeddings) != len(labels):

            raise RuntimeError(
                "Embedding count does not "
                "match label count."
            )

        if len(embeddings) != len(indices):

            raise RuntimeError(
                "Embedding count does not "
                "match selected sample count."
            )

        if not np.isfinite(
            embeddings
        ).all():

            raise RuntimeError(
                "Embeddings contain "
                "NaN or Inf values."
            )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        embedding_path = (
            OUTPUT_DIR
            / (
                f"pcam_{args.model}_"
                f"{split}_embeddings.npy"
            )
        )

        label_path = (
            OUTPUT_DIR
            / (
                f"pcam_{args.model}_"
                f"{split}_labels.npy"
            )
        )

        metadata_path = (
            METADATA_DIR
            / (
                f"pcam_{args.model}_"
                f"{split}_embeddings.json"
            )
        )

        # ----------------------------------------------------
        # Save embeddings
        # ----------------------------------------------------

        np.save(
            embedding_path,
            embeddings,
        )

        np.save(
            label_path,
            labels,
        )

        # ----------------------------------------------------
        # Class statistics
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        class_counts = {
            str(int(label)): int(count_value)
            for label, count_value in zip(
                unique_labels,
                counts,
            )
        }

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {

            "schema_version": "1.0",

            "dataset": {
                "name": "patchcamelyon",
                "split": split,
            },

            "model": {
                "name": model_info.name,
                "model_id": model_info.model_id,
                "embedding_dimension": int(
                    embeddings.shape[1]
                ),
                "input_size": int(
                    model_info.input_size
                ),
                "modality": model_info.modality,
            },

            "sampling": {
                "requested_samples": int(
                    requested_count
                ),
                "actual_samples": int(
                    len(labels)
                ),
                "seed": int(
                    args.seed
                ),
                "source_indices": (
                    indices.tolist()
                ),
            },

            "statistics": {
                "class_counts": class_counts,
                "extraction_time_sec": float(
                    extraction_time
                ),
                "throughput_samples_per_sec": float(
                    len(labels)
                    / extraction_time
                    if extraction_time > 0
                    else 0.0
                ),
            },

            "artifacts": {
                "embedding_file": str(
                    embedding_path
                ),
                "label_file": str(
                    label_path
                ),
                "metadata_file": str(
                    metadata_path
                ),
            },

        }

        with open(
            metadata_path,
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()

        print(
            "Embedding shape :",
            embeddings.shape,
        )

        print(
            "Embedding dtype :",
            embeddings.dtype,
        )

        print(
            "Finite           :",
            bool(
                np.isfinite(
                    embeddings
                ).all()
            ),
        )

        print(
            "Extraction time  :",
            f"{extraction_time:.2f} sec",
        )

        print(
            "Throughput       :",
            f"{len(labels) / extraction_time:.2f} "
            "samples/sec"
            if extraction_time > 0
            else "N/A",
        )

        print(
            "Class counts     :",
            class_counts,
        )

        print()
        print(
            "Saved embeddings:",
            embedding_path,
        )

        print(
            "Saved labels    :",
            label_path,
        )

        print(
            "Saved metadata  :",
            metadata_path,
        )

        # ----------------------------------------------------
        # Store split result for summary
        # ----------------------------------------------------

        all_results[split] = metadata

    # ========================================================
    # Save summary
    # ========================================================

    summary_path = (
        METADATA_DIR
        / (
            f"pcam_{args.model}_"
            f"embedding_summary.json"
        )
    )

    summary = {

        "schema_version": "1.0",

        "dataset": "patchcamelyon",

        "model": {
            "name": model_info.name,
            "model_id": model_info.model_id,
            "embedding_dimension": int(
                model_info.embedding_dim
            ),
        },

        "device": args.device,

        "seed": args.seed,

        "model_load_time_sec": float(
            model_load_time
        ),

        "splits": all_results,

    }

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 72)

    print(
        "PCAM EMBEDDING EXTRACTION COMPLETE"
    )

    print("=" * 72)

    print(
        f"Model              : "
        f"{model_info.name}"
    )

    print(
        f"Embedding dimension: "
        f"{model_info.embedding_dim}"
    )

    print(
        f"Train samples      : "
        f"{len(all_results['train']['sampling']['source_indices'])}"
    )

    print(
        f"Validation samples : "
        f"{len(all_results['validation']['sampling']['source_indices'])}"
    )

    print(
        f"Test samples       : "
        f"{len(all_results['test']['sampling']['source_indices'])}"
    )

    print()

    print(
        "Summary saved:",
        summary_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()