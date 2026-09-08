from __future__ import annotations

import torch
from torch import nn

from .attention import GatedAttention


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model.

    The model receives all tile embeddings belonging to one WSI
    and aggregates them into a single slide-level representation.

    Input:
        [N, embedding_dim]

    Output:
        slide_logits
        slide_embedding
        attention_weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes

        # ----------------------------------------------------
        # Tile projection
        # ----------------------------------------------------

        self.instance_encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = GatedAttention(
            input_dim=hidden_dim,
            hidden_dim=attention_dim,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Slide classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        instances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        instances:
            Tile embeddings with shape [N, input_dim].

        Returns
        -------
        Dictionary containing:

            slide_logits
            slide_embedding
            attention_weights
        """

        if instances.ndim != 2:

            raise ValueError(
                "Expected input shape "
                "[N, embedding_dim]."
            )

        encoded = self.instance_encoder(
            instances
        )

        attention_weights = self.attention(
            encoded
        )

        slide_embedding = torch.sum(
            attention_weights.unsqueeze(-1)
            * encoded,
            dim=0,
        )

        slide_logits = self.classifier(
            slide_embedding
        )

        return {
            "slide_logits": slide_logits,
            "slide_embedding": slide_embedding,
            "attention_weights": attention_weights,
        }