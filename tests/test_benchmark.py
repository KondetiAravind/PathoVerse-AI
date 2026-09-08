import torch

from pathoverse.models.benchmark import ModelBenchmark


def test_benchmark():

    benchmark = ModelBenchmark(
        model_name="test-model",
        device="cpu",
    )

    batches = [
        torch.randn(2, 3),
        torch.randn(2, 3),
    ]

    def inference(batch):
        return batch * 2

    result = benchmark.run(
        inference_fn=inference,
        batches=batches,
        num_samples=4,
        batch_size=2,
        embedding_dim=3,
    )

    assert result.num_samples == 4
    assert result.batch_size == 2
    assert result.embedding_dim == 3
    assert result.total_time_seconds > 0
    assert result.throughput_samples_per_second > 0
    assert result.average_latency_ms > 0