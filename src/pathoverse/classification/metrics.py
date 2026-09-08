from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auroc: float
    f1: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    tn: int
    fp: int
    fn: int
    tp: int


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    sensitivity = recall

    try:
        auroc = roc_auc_score(
            labels,
            probabilities,
        )
    except ValueError:
        auroc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy),
        auroc=float(auroc),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )