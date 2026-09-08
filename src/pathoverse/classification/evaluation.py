from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)


def evaluate_classifier(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> ClassificationMetrics:

    model.eval()

    x = torch.from_numpy(
        embeddings
    ).float()

    y = torch.from_numpy(
        labels
    ).long()

    dataset = TensorDataset(
        x,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_probabilities = []
    all_labels = []

    with torch.inference_mode():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                device
            )

            logits = model(
                batch_x
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                batch_y.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    )

    return compute_classification_metrics(
        labels=labels,
        probabilities=probabilities,
    )