"""Re-time every saved model under the same GPU conditions.

Training benchmarks each model when that run finishes, so the first call pays
MPS warmup and later calls see a hot chip. This pass loads the merged
checkpoints and rotates through them, then reports the median round.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from src.benchmark import _sync
from src.data import load_splits, load_tokenizer
from src.modeling import load_sequence_model, pick_device


def _load_merged(run_dir: Path, cfg: dict):
    tokenizer = load_tokenizer(cfg["model_name"])
    model_dir = run_dir / "model"
    if model_dir.exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model, None, tokenizer
    base = load_sequence_model(cfg["model_name"], cfg["num_labels"])
    peft_model = PeftModel.from_pretrained(base, run_dir / "adapter")
    return peft_model.merge_and_unload(), peft_model, tokenizer


def _encode(tokenizer, texts, batch_size, max_length, device):
    chunk = [texts[i % len(texts)] for i in range(batch_size)]
    encoded = tokenizer(
        chunk,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _mean_latency(model, encoded, device, iters: int) -> float:
    with torch.inference_mode():
        samples = []
        for _ in range(iters):
            _sync(device)
            start = time.perf_counter()
            model(**encoded)
            _sync(device)
            samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples)


def rebenchmark_runs(
    cfg: dict,
    runs_root: Path,
    warmup: int = 12,
    rounds: int = 7,
    iters_per_round: int = 8,
) -> None:
    tokenizer = load_tokenizer(cfg["model_name"])
    _, _, eval_texts, _ = load_splits(cfg, tokenizer)
    device = pick_device()
    run_dirs = sorted(path for path in runs_root.iterdir() if (path / "metrics.json").exists())
    loaded = []
    for run_dir in run_dirs:
        merged, _peft, _tokenizer = _load_merged(run_dir, cfg)
        merged.eval()
        merged.to(device)
        loaded.append((run_dir, merged))

    batch_sizes = cfg["benchmark_batch_sizes"]
    # One shared encoding per batch size; every model sees the same tokens.
    encoded = {
        batch_size: _encode(tokenizer, eval_texts, batch_size, cfg["max_length"], device)
        for batch_size in batch_sizes
    }
    with torch.inference_mode():
        for _run_dir, model in loaded:
            for batch_size in batch_sizes:
                for _ in range(warmup):
                    model(**encoded[batch_size])
            _sync(device)

    medians: dict[Path, list[dict]] = {run_dir: [] for run_dir, _ in loaded}
    for batch_size in batch_sizes:
        per_model: dict[Path, list[float]] = {run_dir: [] for run_dir, _ in loaded}
        for _ in range(rounds):
            for run_dir, model in loaded:
                per_model[run_dir].append(
                    _mean_latency(model, encoded[batch_size], device, iters_per_round)
                )
        for run_dir, samples in per_model.items():
            latency = statistics.median(samples)
            medians[run_dir].append(
                {
                    "batch_size": batch_size,
                    "latency_ms_p50": latency * 1000,
                    "latency_ms_p95": max(samples) * 1000,
                    "latency_ms_mean": (sum(samples) / len(samples)) * 1000,
                    "ms_per_sample_p50": latency * 1000 / batch_size,
                    "throughput_samples_per_sec": batch_size / latency,
                }
            )

    for run_dir, _model in loaded:
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["inference_merged"] = medians[run_dir]
        metrics["inference_benchmark"] = "round_robin_median"
        metrics_path.write_text(json.dumps(metrics, indent=2))
