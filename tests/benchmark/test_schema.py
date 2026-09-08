from pathoverse.benchmark.schema import (
    BenchmarkRecord,
)


def test_benchmark_record():

    record = BenchmarkRecord(
        model="Test",
        model_id="test",
        task="classification",
        embedding_dimension=384,
        accuracy=0.9,
        auroc=0.95,
    )

    data = record.to_dict()

    assert data["model"] == "Test"

    assert (
        data["embedding_dimension"]
        == 384
    )

    assert data["auroc"] == 0.95