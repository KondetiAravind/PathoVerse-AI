import json

from pathoverse.benchmark.collector import (
    BenchmarkCollector,
)


def test_collector_classification(
    tmp_path,
):

    path = (
        tmp_path
        / "classification.json"
    )

    data = {

        "model": {
            "name": "TestModel",
            "model_id": "test-model",
            "embedding_dimension": 384,
        },

        "dataset": {
            "name": "PatchCamelyon",
            "test_samples": 10,
        },

        "test": {
            "accuracy": 0.9,
            "auroc": 0.95,
            "f1": 0.88,
            "precision": 0.87,
            "recall": 0.89,
            "sensitivity": 0.89,
            "specificity": 0.91,
        },

    }

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            data,
            f,
        )

    collector = BenchmarkCollector(
        tmp_path
    )

    collector.add_classification_result(
        path
    )

    records = collector.to_list()

    assert len(records) == 1

    record = records[0]

    assert record["model"] == "TestModel"

    assert (
        record["embedding_dimension"]
        == 384
    )

    assert record["auroc"] == 0.95


def test_collector_missing_file(
    tmp_path,
):

    collector = BenchmarkCollector(
        tmp_path
    )

    missing = (
        tmp_path
        / "missing.json"
    )

    try:

        collector.add_classification_result(
            missing
        )

        assert False

    except FileNotFoundError:

        assert True