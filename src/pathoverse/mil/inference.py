from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import AttentionMIL


@dataclass(frozen=True)
class MILInferenceResult:
    """
    Result of WSI-level attention MIL inference.
    """

    prediction: int
    probability: float
    slide_embedding: np.ndarray
    attention_weights: np.ndarray
    top_tile_indices: list[int]


def run_mil_inference(
    model: AttentionMIL,
    embeddings: np.ndarray,
    device: str = "cuda:0",
    top_k: int = 10,
) -> MILInferenceResult:
    """
    Run attention MIL on one whole-slide bag.

    Parameters
    ----------
    model:
        Trained AttentionMIL model.

    embeddings:
        NumPy array of shape [N, embedding_dim].

    device:
        Torch device.

    top_k:
        Number of highest-attention tiles to return.
    """

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must have shape "
            "[N, embedding_dim]."
        )

    model = model.to(
        device
    )

    model.eval()

    x = torch.from_numpy(
        embeddings
    ).float().to(
        device
    )

    with torch.inference_mode():

        output = model(
            x
        )

        logits = output[
            "slide_logits"
        ]

        probabilities = torch.softmax(
            logits,
            dim=0,
        )

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        probability = float(
            probabilities[
                prediction
            ].item()
        )

        slide_embedding = (
            output[
                "slide_embedding"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        attention_weights = (
            output[
                "attention_weights"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    order = np.argsort(
        attention_weights
    )[::-1]

    top_tile_indices = (
        order[
            :min(
                top_k,
                len(order),
            )
        ]
        .astype(int)
        .tolist()
    )

    return MILInferenceResult(
        prediction=prediction,
        probability=probability,
        slide_embedding=slide_embedding,
        attention_weights=attention_weights,
        top_tile_indices=top_tile_indices,
    )