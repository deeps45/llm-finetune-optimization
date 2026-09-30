"""Inference latency and throughput measurement."""

from __future__ import annotations

import time

import torch


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_inference(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
) -> list[dict]:
    """Time forward passes. Latency is per batch; throughput is examples per second."""
    if not texts:
        raise ValueError("benchmark texts are empty")
    model.eval()
    model.to(device)
    rows = []
    for batch_size in batch_sizes:
        chunk = [texts[i % len(texts)] for i in range(batch_size)]
        encoded = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            for _ in range(warmup):
                model(**encoded)
            _sync(device)
            times = []
            for _ in range(iters):
                _sync(device)
                start = time.perf_counter()
                model(**encoded)
                _sync(device)
                times.append(time.perf_counter() - start)
        times.sort()
        mean = sum(times) / len(times)
        p50 = times[len(times) // 2]
        p95_index = min(len(times) - 1, int(round(0.95 * (len(times) - 1))))
        rows.append(
            {
                "batch_size": batch_size,
                "latency_ms_p50": p50 * 1000,
                "latency_ms_p95": times[p95_index] * 1000,
                "latency_ms_mean": mean * 1000,
                "ms_per_sample_p50": p50 * 1000 / batch_size,
                "throughput_samples_per_sec": batch_size / mean,
            }
        )
    return rows
