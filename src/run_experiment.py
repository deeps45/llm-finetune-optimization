"""Run the full fine-tune vs LoRA study and write results/."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
for _logger_name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
log = logging.getLogger("experiment")


def load_config(path: Path | None) -> dict:
    config_path = path or (ROOT / "configs" / "experiment.json")
    with open(config_path) as handle:
        return json.load(handle)


def apply_quick(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.update(
        {
            "train_samples": 256,
            "eval_samples": 128,
            "epochs": 1,
            "lora_ranks": [8],
            "benchmark_warmup": 2,
            "benchmark_iters": 4,
            "distributed_steps": 4,
            "distributed_warmup_steps": 1,
            "max_length": 32,
        }
    )
    return cfg


def _run(args: list[str]) -> None:
    log.info("Command: %s", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def prefetch(cfg: dict) -> None:
    log.info("Downloading SST-2 and %s", cfg["model_name"])
    _run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; from datasets import load_dataset; "
                "from src.modeling import load_sequence_model; "
                "from src.data import load_tokenizer; "
                "cfg=json.loads(sys.argv[1]); "
                "load_dataset(cfg['dataset'], cfg['dataset_config']); "
                "load_tokenizer(cfg['model_name']); "
                "load_sequence_model(cfg['model_name'], cfg['num_labels'])"
            ),
            json.dumps(cfg),
        ]
    )


def run_experiment(cfg: dict, skip_distributed: bool = False) -> dict:
    from src.report import write_results

    outputs = ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    resolved = outputs / "resolved_config.json"
    resolved.write_text(json.dumps(cfg, indent=2))
    prefetch(cfg)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    runs_root = outputs / "runs"
    jobs = [("full", None, "full")]
    for rank in cfg["lora_ranks"]:
        jobs.append(("lora", rank, f"lora_r{rank}"))

    for mode, rank, name in jobs:
        command = [
            sys.executable,
            "-m",
            "src.train_one",
            "--config",
            str(resolved),
            "--mode",
            mode,
            "--output",
            str(runs_root / name),
        ]
        if rank is not None:
            command.extend(["--rank", str(rank)])
        _run(command)

    distributed = None
    if not skip_distributed:
        from src.distributed import run_distributed

        distributed = run_distributed(cfg, outputs / "distributed")

    runs = []
    for _, _, name in jobs:
        runs.append(json.loads((runs_root / name / "metrics.json").read_text()))

    from src.rebenchmark import rebenchmark_runs

    log.info("Rebenchmarking saved models back-to-back")
    rebenchmark_runs(cfg, runs_root)
    runs = []
    for _, _, name in jobs:
        runs.append(json.loads((runs_root / name / "metrics.json").read_text()))

    summary = write_results(runs, distributed, cfg, ROOT / "results", ROOT / "README.md")
    if summary["passed"]:
        chosen = summary["operating_point"]
        log.info(
            "PASS %s cost reduction %.1f%% accuracy %.3f vs full %.3f",
            chosen["run_name"],
            100 * chosen["verdict"]["cost_reduction"],
            chosen["accuracy"],
            summary["full"]["accuracy"],
        )
    else:
        log.error("FAIL no LoRA run met the cost and quality gates")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT with LoRA and compare it to full fine-tuning")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Small subset used to check the pipeline")
    parser.add_argument("--skip-distributed", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick(cfg)
    summary = run_experiment(cfg, skip_distributed=args.skip_distributed)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
