from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pathoverse.classification.classifier import (
    LinearClassifier,
)
from pathoverse.classification.evaluation import (
    evaluate_classifier,
)


# ============================================================
# PATHS
# ============================================================

EMBEDDING_DIR = Path(
    "data/processed/patchcamelyon/embeddings"
)

RESULTS_DIR = Path(
    "results/classification"
)

MODEL_DIR = Path(
    "models/pcam_classification"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {

    "vit_b_16": {
        "embedding_dim": 768,
        "name": "ViT-B/16",
    },

    "gigapath_flash": {
        "embedding_dim": 384,
        "name": "GigaPath-Flash",
    },

    "conch": {
        "embedding_dim": 512,
        "name": "CONCH",
    },

}


# ============================================================
# DATA LOADING
# ============================================================

def load_embeddings(
    model_name: str,
    split: str,
):
    embedding_path = (
        EMBEDDING_DIR
        / (
            f"pcam_{model_name}_"
            f"{split}_embeddings.npy"
        )
    )

    label_path = (
        EMBEDDING_DIR
        / (
            f"pcam_{model_name}_"
            f"{split}_labels.npy"
        )
    )

    if not embedding_path.exists():

        raise FileNotFoundError(
            f"Missing embeddings: "
            f"{embedding_path}"
        )

    if not label_path.exists():

        raise FileNotFoundError(
            f"Missing labels: "
            f"{label_path}"
        )

    embeddings = np.load(
        embedding_path
    )

    labels = np.load(
        label_path
    )

    return embeddings, labels


# ============================================================
# TRAINING
# ============================================================

def train_classifier(
    model,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
):
    dataset = TensorDataset(
        torch.from_numpy(
            train_embeddings
        ).float(),
        torch.from_numpy(
            train_labels
        ).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_auroc = -float("inf")

    best_state = None

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        sample_count = 0

        epoch_start = time.perf_counter()

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            batch_count = batch_x.size(0)

            running_loss += (
                loss.item()
                * batch_count
            )

            sample_count += (
                batch_count
            )

        train_loss = (
            running_loss
            / sample_count
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        val_metrics = evaluate_classifier(
            model=model,
            embeddings=val_embeddings,
            labels=val_labels,
            device=device,
        )

        current_auroc = (
            val_metrics.auroc
        )

        if np.isnan(
            current_auroc
        ):

            comparison_score = (
                val_metrics.accuracy
            )

        else:

            comparison_score = (
                current_auroc
            )

        if comparison_score > best_val_auroc:

            best_val_auroc = (
                comparison_score
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "validation_accuracy": (
                    val_metrics.accuracy
                ),
                "validation_auroc": (
                    val_metrics.auroc
                ),
                "validation_f1": (
                    val_metrics.f1
                ),
                "epoch_time_sec": (
                    epoch_time
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| Loss {train_loss:.4f} "
            f"| Val Acc "
            f"{val_metrics.accuracy:.4f} "
            f"| Val AUROC "
            f"{val_metrics.auroc:.4f} "
            f"| Val F1 "
            f"{val_metrics.f1:.4f}"
        )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return history


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a linear classifier on "
            "frozen PCam foundation-model embeddings."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIGS.keys(),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    # Directories
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Configuration
    # ========================================================

    config = MODEL_CONFIGS[
        args.model
    ]

    embedding_dim = (
        config["embedding_dim"]
    )

    model_name = (
        config["name"]
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PATHOVERSE — PCAM FROZEN EMBEDDING CLASSIFICATION"
    )
    print("=" * 72)

    print(
        f"Model          : {model_name}"
    )

    print(
        f"Embedding dim  : {embedding_dim}"
    )

    print(
        f"Device         : {args.device}"
    )

    print(
        f"Epochs         : {args.epochs}"
    )

    print(
        f"Batch size     : {args.batch_size}"
    )

    print(
        f"Learning rate  : {args.learning_rate}"
    )

    print(
        f"Seed           : {args.seed}"
    )

    print("=" * 72)

    # ========================================================
    # Load data
    # ========================================================

    train_embeddings, train_labels = (
        load_embeddings(
            args.model,
            "train",
        )
    )

    val_embeddings, val_labels = (
        load_embeddings(
            args.model,
            "validation",
        )
    )

    test_embeddings, test_labels = (
        load_embeddings(
            args.model,
            "test",
        )
    )

    # ========================================================
    # Validate dimensions
    # ========================================================

    if train_embeddings.shape[1] != embedding_dim:

        raise RuntimeError(
            "Training embedding dimension "
            "does not match model configuration."
        )

    if val_embeddings.shape[1] != embedding_dim:

        raise RuntimeError(
            "Validation embedding dimension "
            "does not match model configuration."
        )

    if test_embeddings.shape[1] != embedding_dim:

        raise RuntimeError(
            "Test embedding dimension "
            "does not match model configuration."
        )

    # ========================================================
    # Dataset information
    # ========================================================

    print()
    print(
        "Training samples   :",
        len(train_labels),
    )

    print(
        "Validation samples :",
        len(val_labels),
    )

    print(
        "Test samples       :",
        len(test_labels),
    )

    print(
        "Train class counts :",
        {
            str(int(label)): int(
                np.sum(
                    train_labels == label
                )
            )
            for label in np.unique(
                train_labels
            )
        },
    )

    # ========================================================
    # Model
    # ========================================================

    model = LinearClassifier(
        input_dim=embedding_dim,
        num_classes=2,
    ).to(
        args.device
    )

    # ========================================================
    # Training
    # ========================================================

    print()
    print("-" * 72)
    print(
        "TRAINING LINEAR CLASSIFIER"
    )
    print("-" * 72)

    training_start = time.perf_counter()

    history = train_classifier(
        model=model,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ========================================================
    # Final validation
    # ========================================================

    print()
    print("-" * 72)
    print(
        "FINAL VALIDATION"
    )
    print("-" * 72)

    val_metrics = evaluate_classifier(
        model=model,
        embeddings=val_embeddings,
        labels=val_labels,
        device=args.device,
    )

    # ========================================================
    # Final test
    # ========================================================

    print()
    print("-" * 72)
    print(
        "FINAL TEST"
    )
    print("-" * 72)

    test_start = time.perf_counter()

    test_metrics = evaluate_classifier(
        model=model,
        embeddings=test_embeddings,
        labels=test_labels,
        device=args.device,
    )

    test_time = (
        time.perf_counter()
        - test_start
    )

    # ========================================================
    # Save checkpoint
    # ========================================================

    checkpoint_path = (
        MODEL_DIR
        / f"{args.model}_linear_classifier.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": args.model,
            "embedding_dimension": embedding_dim,
            "num_classes": 2,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    # ========================================================
    # Convert metrics
    # ========================================================

    val_result = {
        "accuracy": val_metrics.accuracy,
        "auroc": val_metrics.auroc,
        "f1": val_metrics.f1,
        "precision": val_metrics.precision,
        "recall": val_metrics.recall,
        "sensitivity": val_metrics.sensitivity,
        "specificity": val_metrics.specificity,
        "tn": val_metrics.tn,
        "fp": val_metrics.fp,
        "fn": val_metrics.fn,
        "tp": val_metrics.tp,
    }

    test_result = {
        "accuracy": test_metrics.accuracy,
        "auroc": test_metrics.auroc,
        "f1": test_metrics.f1,
        "precision": test_metrics.precision,
        "recall": test_metrics.recall,
        "sensitivity": test_metrics.sensitivity,
        "specificity": test_metrics.specificity,
        "tn": test_metrics.tn,
        "fp": test_metrics.fp,
        "fn": test_metrics.fn,
        "tp": test_metrics.tp,
    }

    # ========================================================
    # Save result
    # ========================================================

    result = {

        "schema_version": "1.0",

        "task": "patchcamelyon_binary_classification",

        "model": {
            "name": model_name,
            "model_id": args.model,
            "embedding_dimension": embedding_dim,
            "foundation_model_frozen": True,
            "classifier": "linear",
        },

        "dataset": {
            "name": "PatchCamelyon",
            "train_samples": int(
                len(train_labels)
            ),
            "validation_samples": int(
                len(val_labels)
            ),
            "test_samples": int(
                len(test_labels)
            ),
        },

        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "training_time_sec": float(
                training_time
            ),
        },

        "validation": val_result,

        "test": test_result,

        "inference": {
            "test_evaluation_time_sec": float(
                test_time
            ),
            "test_samples_per_sec": float(
                len(test_labels)
                / test_time
                if test_time > 0
                else 0.0
            ),
        },

        "artifacts": {
            "checkpoint": str(
                checkpoint_path
            ),
        },

        "history": history,

    }

    result_path = (
        RESULTS_DIR
        / f"{args.model}_classification.json"
    )

    with open(
        result_path,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 72)
    print(
        "CLASSIFICATION COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        f"Model       : {model_name}"
    )

    print()
    print(
        "TEST RESULTS"
    )

    print(
        f"Accuracy    : {test_metrics.accuracy:.4f}"
    )

    print(
        f"AUROC       : {test_metrics.auroc:.4f}"
    )

    print(
        f"F1          : {test_metrics.f1:.4f}"
    )

    print(
        f"Precision   : {test_metrics.precision:.4f}"
    )

    print(
        f"Recall      : {test_metrics.recall:.4f}"
    )

    print(
        f"Sensitivity : {test_metrics.sensitivity:.4f}"
    )

    print(
        f"Specificity : {test_metrics.specificity:.4f}"
    )

    print()
    print(
        "Confusion Matrix"
    )

    print(
        f"TN={test_metrics.tn} "
        f"FP={test_metrics.fp}"
    )

    print(
        f"FN={test_metrics.fn} "
        f"TP={test_metrics.tp}"
    )

    print()
    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Results:",
        result_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()