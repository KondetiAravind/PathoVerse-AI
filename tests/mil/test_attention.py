import torch

from pathoverse.mil.attention import (
    GatedAttention,
)


def test_attention_shape():

    module = GatedAttention(
        input_dim=384,
        hidden_dim=64,
    )

    x = torch.randn(
        45,
        384,
    )

    weights = module(x)

    assert weights.shape == (
        45,
    )


def test_attention_sums_to_one():

    module = GatedAttention(
        input_dim=384,
        hidden_dim=64,
    )

    x = torch.randn(
        45,
        384,
    )

    weights = module(x)

    assert torch.isclose(
        weights.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_attention_is_finite():

    module = GatedAttention(
        input_dim=384,
        hidden_dim=64,
    )

    x = torch.randn(
        45,
        384,
    )

    weights = module(x)

    assert torch.isfinite(
        weights
    ).all()