"""Train one configuration and write its metrics."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import psutil
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.benchmark import benchmark_inference
from src.data import load_splits, load_tokenizer
from src.metrics import allreduce_bytes_per_step, optimizer_and_grad_bytes, training_cost_bytes
from src.modeling import build_model, count_parameters, device_name, pick_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
for _logger_name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
log = logging.getLogger("train_one")


def _load_cfg(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


class ResourceCallback(TrainerCallback):
    def __init__(self):
        self.baseline_rss = psutil.Process().memory_info().rss
        self.peak_rss = self.baseline_rss
        self.peak_mps = 0
        self.epoch_times: list[float] = []
        self._epoch_start: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.baseline_rss = psutil.Process().memory_info().rss
        self.peak_rss = self.baseline_rss

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)

    def on_step_end(self, args, state, control, **kwargs):
        rss = psutil.Process().memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        if torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory()
            if allocated > self.peak_mps:
                self.peak_mps = allocated


def _compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
    }


def _training_args(cfg: dict, output_dir: Path, learning_rate: float) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="linear",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to=[],
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
    )
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**accepted)


def _make_trainer(model, args, train_ds, eval_ds, tokenizer, collator, callbacks):
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters and "tokenizer" not in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def _eval_history(trainer: Trainer) -> list[dict]:
    history = []
    for entry in trainer.state.log_history:
        if "eval_accuracy" not in entry:
            continue
        history.append(
            {
                "epoch": entry.get("epoch"),
                "accuracy": float(entry["eval_accuracy"]),
                "f1": float(entry.get("eval_f1", 0.0)),
                "eval_loss": float(entry["eval_loss"]) if "eval_loss" in entry else None,
            }
        )
    return history


def _inference_rows(model, tokenizer, texts, cfg, device):
    return benchmark_inference(
        model,
        tokenizer,
        texts,
        device,
        max_length=cfg["max_length"],
        batch_sizes=cfg["benchmark_batch_sizes"],
        warmup=cfg["benchmark_warmup"],
        iters=cfg["benchmark_iters"],
    )


def train_one(cfg: dict, mode: str, rank: int | None, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    set_seed(cfg["seed"])
    learning_rate = cfg["full_learning_rate"] if mode == "full" else cfg["lora_learning_rate"]
    log.info("Building %s rank=%s lr=%s", mode, rank, learning_rate)

    tokenizer = load_tokenizer(cfg["model_name"])
    train_ds, eval_ds, eval_texts, collator = load_splits(cfg, tokenizer)
    model, spec = build_model(cfg, mode, rank)
    trainable, total = count_parameters(model)
    log.info("Parameters total=%d trainable=%d (%.3f%%)", total, trainable, 100 * trainable / total)

    rss_after_model = psutil.Process().memory_info().rss
    meter = ResourceCallback()
    args = _training_args(cfg, output_dir, learning_rate)
    trainer = _make_trainer(model, args, train_ds, eval_ds, tokenizer, collator, [meter])

    train_result = trainer.train()
    final_eval = trainer.evaluate()
    history = _eval_history(trainer)
    steps = int(trainer.state.global_step)
    device = pick_device()

    inference_merged = None
    inference_unmerged = None
    if mode == "lora":
        trainer.model.save_pretrained(output_dir / "adapter")
        inference_unmerged = _inference_rows(trainer.model, tokenizer, eval_texts, cfg, device)
        merged = trainer.model.merge_and_unload()
        inference_merged = _inference_rows(merged, tokenizer, eval_texts, cfg, device)
    else:
        inference_merged = _inference_rows(trainer.model, tokenizer, eval_texts, cfg, device)
        trainer.save_model(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    world = cfg["distributed_world_size"]
    metrics = {
        "run_name": output_dir.name,
        "mode": mode,
        "rank": spec["rank"],
        "lora_alpha": spec["lora_alpha"],
        "target_modules": spec["target_modules"],
        "modules_to_save": spec["modules_to_save"],
        "learning_rate": learning_rate,
        "model_name": cfg["model_name"],
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "batch_size": cfg["batch_size"],
        "max_length": cfg["max_length"],
        "device": device_name(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
            "datasets": version("datasets"),
        },
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total,
        "optimizer_and_grad_bytes": optimizer_and_grad_bytes(trainable),
        "training_cost_bytes": training_cost_bytes(trainable, steps),
        "allreduce_bytes_per_step": allreduce_bytes_per_step(trainable, world),
        "global_steps": steps,
        "train_runtime_sec": float(train_result.metrics["train_runtime"]),
        "train_samples_per_second": float(train_result.metrics["train_samples_per_second"]),
        "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
        "rss_after_model_bytes": rss_after_model,
        "peak_rss_bytes": meter.peak_rss,
        "memory_overhead_bytes": max(0, meter.peak_rss - rss_after_model),
        "peak_mps_bytes": meter.peak_mps,
        "epoch_times_sec": meter.epoch_times,
        "accuracy": float(final_eval["eval_accuracy"]),
        "f1": float(final_eval["eval_f1"]),
        "eval_loss": float(final_eval["eval_loss"]),
        "eval_history": history,
        "inference_merged": inference_merged,
        "inference_unmerged": inference_unmerged,
        "cost_definition": (
            "trainable_params * (8 bytes Adam moments + 4 bytes gradient) * optimizer steps"
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info(
        "Finished %s accuracy=%.4f f1=%.4f runtime=%.1fs",
        output_dir.name,
        metrics["accuracy"],
        metrics["f1"],
        metrics["train_runtime_sec"],
    )
    del trainer
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train one full or LoRA configuration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    train_one(cfg, args.mode, args.rank, Path(args.output))


if __name__ == "__main__":
    main()
