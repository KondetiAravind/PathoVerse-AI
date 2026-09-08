import torch

from pathoverse.mil.model import (
    AttentionMIL,
)


def test_mil_output_shapes():

    model = AttentionMIL(
        input_dim=384,
        hidden_dim=128,
        attention_dim=64,
        num_classes=2,
    )

    x = torch.randn(
        45,
        384,
    )

    output = model(x)

    assert output[
        "slide_logits"
    ].shape == (
        2,
    )

    assert output[
        "slide_embedding"
    ].shape == (
        128,
    )

    assert output[
        "attention_weights"
    ].shape == (
        45,
    )


def test_mil_outputs_are_finite():

    model = AttentionMIL(
        input_dim=512,
        hidden_dim=128,
        attention_dim=64,
        num_classes=2,
    )

    x = torch.randn(
        20,
        512,
    )

    output = model(x)

    for value in output.values():

        assert torch.isfinite(
            value
        ).all()


def test_mil_attention_sum():

    model = AttentionMIL(
        input_dim=384,
        hidden_dim=128,
        attention_dim=64,
        num_classes=2,
    )

    x = torch.randn(
        45,
        384,
    )

    output = model(x)

    weights = output[
        "attention_weights"
    ]

    assert torch.isclose(
        weights.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )