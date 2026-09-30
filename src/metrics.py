"""Training-cost definitions used by the experiment.

LoRA leaves the backbone frozen and trains a small set of adapter weights.
AdamW keeps two fp32 moment estimates per trainable parameter and the backward
pass materializes an fp32 gradient for each of them. Data-parallel training
then all-reduces that gradient every step. Those three terms are the training
cost that grows with model size and GPU count; they scale directly with the
number of trainable parameters.
"""

from __future__ import annotations

ADAM_STATE_BYTES_PER_PARAM = 8  # fp32 first moment + fp32 second moment
GRAD_BYTES_PER_PARAM = 4  # fp32 gradient


def optimizer_and_grad_bytes(trainable_params: int) -> int:
    """Bytes of Adam state and gradients stored for the trainable tensors."""
    if trainable_params < 0:
        raise ValueError("trainable_params must be non-negative")
    return trainable_params * (ADAM_STATE_BYTES_PER_PARAM + GRAD_BYTES_PER_PARAM)


def training_cost_bytes(trainable_params: int, steps: int) -> int:
    """Optimizer-state and gradient traffic across the whole run.

    Each step reads and writes the Adam moments and produces a gradient.
    Frozen weights contribute nothing to this quantity.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return optimizer_and_grad_bytes(trainable_params) * steps


def allreduce_bytes_per_step(trainable_params: int, world_size: int, dtype_bytes: int = 4) -> float:
    """Ring all-reduce volume for one DDP step.

    A ring all-reduce moves ``2 * (n - 1) / n`` of the gradient buffer.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return 0.0
    grad_bytes = trainable_params * dtype_bytes
    return 2.0 * (world_size - 1) / world_size * grad_bytes


def reduction_ratio(baseline: float, candidate: float) -> float:
    """Fraction of ``baseline`` removed by ``candidate``. 0.6 means a 60% reduction."""
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (baseline - candidate) / baseline


def meets_targets(
    full_cost: float,
    lora_cost: float,
    full_accuracy: float,
    lora_accuracy: float,
    target_reduction: float = 0.60,
    max_accuracy_drop: float = 0.01,
) -> dict:
    """Cost must fall by at least ``target_reduction`` and accuracy drop stay within band."""
    reduction = reduction_ratio(full_cost, lora_cost)
    drop = full_accuracy - lora_accuracy
    cost_ok = reduction + 1e-12 >= target_reduction
    quality_ok = drop <= max_accuracy_drop + 1e-12
    return {
        "cost_reduction": reduction,
        "accuracy_drop": drop,
        "cost_ok": cost_ok,
        "quality_ok": quality_ok,
        "passed": cost_ok and quality_ok,
    }
