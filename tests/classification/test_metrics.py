import numpy as np

from pathoverse.classification.metrics import (
    compute_classification_metrics,
)


def test_classification_metrics():

    labels = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = compute_classification_metrics(
        labels,
        probabilities,
    )

    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.auroc == 1.0


def test_confusion_matrix():

    labels = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.9, 0.1, 0.9, 0.1]
    )

    metrics = compute_classification_metrics(
        labels,
        probabilities,
    )

    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.tp == 1