"""Two-process DDP benchmark comparing full fine-tuning with LoRA.

The quality runs use a single accelerator. This script runs data-parallel
training through gloo so all-reduce volume and step time are measured.
On a CUDA machine the same trainer also launches under torchrun, which sets
LOCAL_RANK and lets Hugging Face Trainer wrap the model in DDP.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_splits, load_tokenizer
from src.metrics import allreduce_bytes_per_step
from src.modeling import build_model, count_parameters, set_seed


def _pad_column(sequences, pad_value: int, max_length: int) -> torch.Tensor:
    rows = []
    for sequence in sequences:
        trimmed = list(sequence)[:max_length]
        trimmed.extend([pad_value] * (max_length - len(trimmed)))
        rows.append(trimmed)
    return torch.tensor(rows, dtype=torch.long)


def _batches(cfg: dict, rank: int, world_size: int, batch_size: int):
    tokenizer = load_tokenizer(cfg["model_name"])
    train_ds, _, _, _ = load_splits(cfg, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = _pad_column(train_ds["input_ids"], pad_id, cfg["max_length"])[rank::world_size]
    attention = _pad_column(train_ds["attention_mask"], 0, cfg["max_length"])[rank::world_size]
    labels = torch.tensor(train_ds["labels"], dtype=torch.long)[rank::world_size]
    dataset = TensorDataset(input_ids, attention, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    batches = list(loader)
    if not batches:
        raise RuntimeError("Distributed shard is smaller than the batch size.")
    return batches


def _ddp_lora_rank(cfg: dict) -> int:
    ranks = cfg["lora_ranks"]
    if 8 in ranks:
        return 8
    return ranks[len(ranks) // 2]


def _worker(rank: int, world_size: int, mode: str, rank_lora: int | None, cfg: dict, result_path: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(cfg["distributed_port"])
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=180))
    try:
        set_seed(cfg["seed"] + rank)
        batches = _batches(cfg, rank, world_size, cfg["distributed_batch_size"])
        model, _ = build_model(cfg, mode, rank_lora)
        model.cpu()
        trainable, total = count_parameters(model)
        model = DDP(model)
        learning_rate = cfg["lora_learning_rate"] if mode == "lora" else cfg["full_learning_rate"]
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        warmup = cfg["distributed_warmup_steps"]
        measured = cfg["distributed_steps"]
        times: list[float] = []
        loss_first = None
        loss_last = None
        for step in range(warmup + measured):
            input_ids, attention, labels = batches[step % len(batches)]
            dist.barrier()
            start = time.perf_counter()
            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            elapsed = time.perf_counter() - start
            loss_value = float(loss.detach())
            if step == 0:
                loss_first = loss_value
            loss_last = loss_value
            if step >= warmup:
                times.append(elapsed)
        if rank == 0:
            times.sort()
            payload = {
                "mode": mode,
                "rank": rank_lora,
                "world_size": world_size,
                "backend": "gloo",
                "device": "cpu",
                "trainable_params": trainable,
                "total_params": total,
                "allreduce_bytes_per_step": allreduce_bytes_per_step(trainable, world_size),
                "steps_measured": measured,
                "warmup_steps": warmup,
                "batch_size": cfg["distributed_batch_size"],
                "mean_step_sec": sum(times) / len(times),
                "median_step_sec": times[len(times) // 2],
                "loss_first": loss_first,
                "loss_last": loss_last,
            }
            Path(result_path).write_text(json.dumps(payload, indent=2))
            print(
                f"ddp {mode} r={rank_lora} trainable={trainable} "
                f"allreduce_mb={payload['allreduce_bytes_per_step'] / 1e6:.3f} "
                f"step_ms={payload['mean_step_sec'] * 1000:.1f}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(cfg: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    world_size = cfg["distributed_world_size"]
    jobs = [("full", None), ("lora", _ddp_lora_rank(cfg))]
    runs = []
    for mode, lora_rank in jobs:
        name = "full" if mode == "full" else f"lora_r{lora_rank}"
        result_path = output_dir / f"ddp_{name}.json"
        if result_path.exists():
            result_path.unlink()
        mp.spawn(
            _worker,
            args=(world_size, mode, lora_rank, cfg, str(result_path)),
            nprocs=world_size,
            join=True,
        )
        runs.append(json.loads(result_path.read_text()))
    summary = {"world_size": world_size, "backend": "gloo", "runs": runs}
    (output_dir / "distributed.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Measure DDP step time and all-reduce volume")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/distributed")
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = json.load(handle)
    run_distributed(cfg, Path(args.output))


if __name__ == "__main__":
    main()
