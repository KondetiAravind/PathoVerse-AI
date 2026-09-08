import torch

from pathoverse.classification.classifier import (
    LinearClassifier,
)


def test_linear_classifier_output_shape():

    model = LinearClassifier(
        input_dim=768,
        num_classes=2,
    )

    x = torch.randn(
        8,
        768,
    )

    output = model(x)

    assert output.shape == (
        8,
        2,
    )


def test_linear_classifier_forward_is_finite():

    model = LinearClassifier(
        input_dim=384,
        num_classes=2,
    )

    x = torch.randn(
        4,
        384,
    )

    output = model(x)

    assert torch.isfinite(
        output
    ).all()