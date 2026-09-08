from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    num_samples: int
    batch_size: int
    embedding_dim: int
    total_time_seconds: float
    average_latency_ms: float
    throughput_samples_per_second: float
    peak_gpu_memory_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


class ModelBenchmark:
    """
    Benchmark inference performance of a PathoVerse model.

    Measures:
        - total inference time
        - average latency
        - throughput
        - peak GPU memory
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device

    def run(
        self,
        inference_fn: Callable,
        batches: list,
        num_samples: int,
        batch_size: int,
        embedding_dim: int,
    ) -> BenchmarkResult:

        if not batches:
            raise ValueError("No batches supplied for benchmarking.")

        use_cuda = (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
        )

        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()

        for batch in batches:
            inference_fn(batch)

        if use_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if use_cuda:
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        else:
            peak_memory = 0.0

        latency_ms = (
            elapsed / num_samples
        ) * 1000

        throughput = (
            num_samples / elapsed
            if elapsed > 0
            else 0.0
        )

        return BenchmarkResult(
            model_name=self.model_name,
            device=self.device,
            num_samples=num_samples,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            total_time_seconds=elapsed,
            average_latency_ms=latency_ms,
            throughput_samples_per_second=throughput,
            peak_gpu_memory_mb=peak_memory,
        )