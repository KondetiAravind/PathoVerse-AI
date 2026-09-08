from __future__ import annotations

import torch
from torch import nn


class GatedAttention(nn.Module):
    """
    Gated attention mechanism for Multiple Instance Learning.

    Input:
        H -> [N, input_dim]

    Output:
        attention_logits -> [N]

    The formulation is:

        A = softmax(
                w^T(
                    tanh(VH) * sigmoid(UH)
                )
            )

    This allows the model to learn which instances/tiles
    contribute most strongly to the slide representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.v = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

        self.u = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.Sigmoid(),
        )

        self.attention = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        instances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        instances:
            Tensor of shape [N, input_dim]

        Returns
        -------
        attention_weights:
            Tensor of shape [N]
            Sum equals 1.
        """

        if instances.ndim != 2:

            raise ValueError(
                "Expected instances with shape "
                "[N, input_dim]."
            )

        if instances.shape[1] != self.input_dim:

            raise ValueError(
                "Input embedding dimension "
                f"{instances.shape[1]} does not match "
                f"configured dimension {self.input_dim}."
            )

        v = self.v(
            instances
        )

        u = self.u(
            instances
        )

        gated = (
            v * u
        )

        logits = self.attention(
            gated
        ).squeeze(
            -1
        )

        weights = torch.softmax(
            logits,
            dim=0,
        )

        return weights