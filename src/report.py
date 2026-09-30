"""Aggregate run metrics into a report, a figure, and the README results block."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import meets_targets, reduction_ratio


def _mb(num_bytes: float) -> float:
    return num_bytes / 1e6


def _label(run: dict) -> str:
    if run["mode"] == "full":
        return "Full fine-tune"
    return f"LoRA r={run['rank']} ({run['epochs']} ep)"


def _batch_row(rows: list[dict] | None, batch_size: int) -> dict | None:
    if not rows:
        return None
    for row in rows:
        if row["batch_size"] == batch_size:
            return row
    return rows[0]


def _annotate(runs: list[dict], full: dict, target_reduction: float, max_accuracy_drop: float) -> list[dict]:
    annotated = []
    for run in runs:
        item = dict(run)
        if run["mode"] == "lora":
            item["verdict"] = meets_targets(
                full["training_cost_bytes"],
                run["training_cost_bytes"],
                full["accuracy"],
                run["accuracy"],
                target_reduction=target_reduction,
                max_accuracy_drop=max_accuracy_drop,
            )
            item["wall_clock_reduction"] = reduction_ratio(full["train_runtime_sec"], run["train_runtime_sec"])
            full_mem = full["memory_overhead_bytes"] or 1
            item["memory_overhead_reduction"] = reduction_ratio(full_mem, max(run["memory_overhead_bytes"], 0))
        annotated.append(item)
    return annotated


def select_operating_point(annotated: list[dict]) -> dict | None:
    viable = [run for run in annotated if run["mode"] == "lora" and run["verdict"]["passed"]]
    if not viable:
        return None
    # Highest accuracy among runs that already clear the cost and quality gates.
    return max(viable, key=lambda run: (run["accuracy"], -run["training_cost_bytes"]))


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _pp(value: float) -> str:
    return f"{100 * value:.2f}"


def render_markdown(summary: dict) -> str:
    full = summary["full"]
    chosen = summary["operating_point"]
    lines = []
    lines.append("## Results")
    lines.append("")
    lines.append(
        f"Measured on {full['device']} ({full['platform']}) with `{full['model_name']}` "
        f"fine-tuned on GLUE SST-2 ({full['train_samples']:,} train examples, "
        f"{full['eval_samples']:,} validation examples, {full['epochs']} epochs, "
        f"sequence length {full['max_length']}, seed {full['seed']})."
    )
    lines.append("")
    if chosen is None:
        lines.append("No LoRA configuration cleared the cost and quality gates. See the table below.")
    else:
        verdict = chosen["verdict"]
        direction = "higher" if verdict["accuracy_drop"] < 0 else "lower" if verdict["accuracy_drop"] > 0 else "the same"
        lines.append(
            f"**Operating point: {_label(chosen)}.** Optimizer and gradient training cost was "
            f"{_pct(1 - verdict['cost_reduction'])} of full fine-tuning, a **{_pct(verdict['cost_reduction'])} reduction** "
            f"(target {_pct(summary['cost_reduction_target'])}). Validation accuracy was "
            f"**{_pct(chosen['accuracy'])}** for {_label(chosen)} and **{_pct(full['accuracy'])}** for full fine-tuning "
            f"({direction} by {abs(verdict['accuracy_drop']) * 100:.2f} percentage points; "
            f"quality band is {summary['max_accuracy_drop'] * 100:.1f} points). "
            f"Binary F1 was {_pct(chosen['f1'])} for {_label(chosen)} and {_pct(full['f1'])} for full fine-tuning."
        )
        lines.append("")
        lines.append(
            f"Wall-clock training time was {full['train_runtime_sec']:.1f}s for full fine-tuning "
            f"({full['train_samples_per_second']:.1f} examples/s) and "
            f"{chosen['train_runtime_sec']:.1f}s for {_label(chosen)} "
            f"({chosen['train_samples_per_second']:.1f} examples/s). "
            f"Peak training-memory overhead was {_mb(full['memory_overhead_bytes']):.0f} MB for full fine-tuning and "
            f"{_mb(chosen['memory_overhead_bytes']):.0f} MB for {_label(chosen)}."
        )
    lines.append("")
    lines.append("### Training comparison")
    lines.append("")
    lines.append(
        "| Run | Epochs | Trainable params | Train state | Cost vs full | Val accuracy | Val F1 | Train time |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in summary["runs"]:
        state = f"{_mb(run['optimizer_and_grad_bytes']):.2f} MB"
        if run["mode"] == "full":
            cost = "baseline"
        else:
            cost = _pct(run["verdict"]["cost_reduction"]) + " lower"
        lines.append(
            f"| {_label(run)} | {run['epochs']} | {run['trainable_params']:,} | {state} | {cost} | "
            f"{_pct(run['accuracy'])} | {_pct(run['f1'])} | {run['train_runtime_sec']:.1f}s |"
        )
    lines.append("")
    lines.append("### Inference trade-offs")
    lines.append("")
    lines.append(
        "Latency and throughput below are for the merged model, measured by rotating through every checkpoint so each one sees the same inputs and the same GPU state. "
        "Merging folds the adapters into the query and value matrices, which is the network you would deploy."
    )
    lines.append("")
    lines.append("| Run | Batch 1 latency | Batch 8 throughput | Batch 32 throughput |")
    lines.append("| --- | ---: | ---: | ---: |")
    for run in summary["runs"]:
        single = _batch_row(run["inference_merged"], 1)
        mid = _batch_row(run["inference_merged"], 8)
        wide = _batch_row(run["inference_merged"], 32)
        lines.append(
            f"| {_label(run)} | {single['latency_ms_p50']:.2f} ms | "
            f"{mid['throughput_samples_per_sec']:.0f} ex/s | {wide['throughput_samples_per_sec']:.0f} ex/s |"
        )
    lines.append("")
    distributed = summary.get("distributed")
    lines.append("### Distributed training")
    lines.append("")
    if not distributed:
        lines.append("Distributed benchmark was not run.")
    else:
        lines.append(
            f"Data-parallel training used {distributed['world_size']} processes, backend `{distributed['backend']}`, on CPU. "
            "Each step is a real forward, backward, and AdamW update with a ring all-reduce of the gradient. "
            "All-reduce volume below is `2 * (n - 1) / n` times the fp32 gradient."
        )
        lines.append("")
        lines.append("| Run | Trainable params | All-reduce / step | Mean step time | First loss | Last loss |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for run in distributed["runs"]:
            name = "Full fine-tune" if run["mode"] == "full" else f"LoRA r={run['rank']}"
            lines.append(
                f"| {name} | {run['trainable_params']:,} | {_mb(run['allreduce_bytes_per_step']):.3f} MB | "
                f"{run['mean_step_sec'] * 1000:.1f} ms | {run['loss_first']:.4f} | {run['loss_last']:.4f} |"
            )
        full_d = next(run for run in distributed["runs"] if run["mode"] == "full")
        lora_d = next(run for run in distributed["runs"] if run["mode"] == "lora")
        comm_reduction = reduction_ratio(full_d["allreduce_bytes_per_step"], lora_d["allreduce_bytes_per_step"])
        lines.append("")
        lines.append(
            f"Gradient all-reduce volume for LoRA r={lora_d['rank']} "
            f"was {_pct(comm_reduction)} lower than full fine-tuning. "
            f"Mean step time was {full_d['mean_step_sec'] * 1000:.1f} ms for full fine-tuning and "
            f"{lora_d['mean_step_sec'] * 1000:.1f} ms for LoRA."
        )
    lines.append("")
    lines.append("### Cost definition")
    lines.append("")
    lines.append(
        "Training cost is the AdamW state plus the gradient, multiplied by the number of optimizer steps: "
        "`trainable_params * (8 bytes of moments + 4 bytes of gradient) * steps`. "
        "Frozen backbone weights do not enter that quantity. "
        "The same trainable-parameter count sets distributed all-reduce volume. "
        "Wall-clock time and peak process memory are reported beside it because a single-machine forward pass still executes the full network."
    )
    lines.append("")
    status = "PASSED" if summary["passed"] else "FAILED"
    lines.append(
        f"Gate: cost reduction ≥ {_pct(summary['cost_reduction_target'])} and accuracy drop ≤ "
        f"{summary['max_accuracy_drop'] * 100:.1f} percentage points. Result: **{status}**."
    )
    lines.append("")
    return "\n".join(lines)


def _colors(runs: list[dict]) -> list[str]:
    palette = {4: "#8FCB9B", 8: "#1F7A4D", 16: "#145A32"}
    return ["#1F2933" if run["mode"] == "full" else palette.get(run["rank"], "#1F7A4D") for run in runs]


def write_figure(runs: list[dict], path: Path) -> None:
    labels = ["Full" if run["mode"] == "full" else f"r={run['rank']}\n{run['epochs']} ep" for run in runs]
    colors = _colors(runs)
    costs = [_mb(run["optimizer_and_grad_bytes"]) for run in runs]
    accuracy = [100 * run["accuracy"] for run in runs]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(labels, costs, color=colors)
    axes[0].set_ylabel("Adam state + gradient (MB)")
    axes[0].set_title("Training state")
    axes[1].bar(labels, accuracy, color=colors)
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("SST-2 quality")
    ymin = max(0, min(accuracy) - 3)
    axes[1].set_ylim(ymin, min(100, max(accuracy) + 2))
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def update_readme(readme_path: Path, results_md: str) -> None:
    start = "<!-- RESULTS:START -->"
    end = "<!-- RESULTS:END -->"
    block = f"{start}\n{results_md.rstrip()}\n{end}"
    text = readme_path.read_text()
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        readme_path.write_text(pre + block + post)
        return
    readme_path.write_text(text.rstrip() + "\n\n" + block + "\n")


def build_summary(runs: list[dict], distributed: dict | None, cfg: dict) -> dict:
    ordered = sorted(runs, key=lambda run: (0 if run["mode"] == "full" else 1, run["rank"] or 0, run.get("epochs") or 0))
    full = next(run for run in ordered if run["mode"] == "full")
    annotated = _annotate(
        ordered,
        full,
        target_reduction=cfg["cost_reduction_target"],
        max_accuracy_drop=cfg["max_accuracy_drop"],
    )
    chosen = select_operating_point(annotated)
    passed = chosen is not None
    return {
        "model_name": full["model_name"],
        "device": full["device"],
        "platform": full["platform"],
        "versions": full.get("versions", {}),
        "python": full.get("python"),
        "cost_reduction_target": cfg["cost_reduction_target"],
        "max_accuracy_drop": cfg["max_accuracy_drop"],
        "cost_definition": full["cost_definition"],
        "runs": annotated,
        "full": full,
        "operating_point": chosen,
        "passed": passed,
        "distributed": distributed,
    }


def write_results(runs: list[dict], distributed: dict | None, cfg: dict, results_dir: Path, readme_path: Path) -> dict:
    summary = build_summary(runs, distributed, cfg)
    # The operating point object is shared with runs; copy a compact pointer for the JSON.
    serializable = json.loads(json.dumps(summary, default=str))
    if serializable["operating_point"] is not None:
        serializable["operating_point_name"] = serializable["operating_point"]["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(serializable, indent=2))
    markdown = render_markdown(summary)
    (results_dir / "REPORT.md").write_text(markdown)
    write_figure(summary["runs"], results_dir / "tradeoffs.png")
    update_readme(readme_path, markdown)
    return summary
