from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """
    Lightweight linear classifier operating on frozen
    foundation-model embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.classifier(x)