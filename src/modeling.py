"""Model construction for full fine-tuning and LoRA."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

QUERY_VALUE_MODULES = ("q_lin", "v_lin", "query", "value", "q_proj", "v_proj")
HEAD_MODULES = ("pre_classifier", "classifier", "score")


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def _leaf_names(model) -> set[str]:
    return {name.split(".")[-1] for name, _ in model.named_modules()}


def lora_target_modules(model, configured: list[str] | None = None) -> list[str]:
    if configured:
        return list(configured)
    leaves = _leaf_names(model)
    targets = [name for name in QUERY_VALUE_MODULES if name in leaves]
    if not targets:
        raise RuntimeError(
            "Could not find query/value projections. "
            f"Linear leaves include: {sorted(leaves)}"
        )
    return targets


def head_modules(model) -> list[str]:
    leaves = _leaf_names(model)
    found = [name for name in HEAD_MODULES if name in leaves]
    if not found:
        raise RuntimeError("Could not find a classification head to train.")
    return found


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_sequence_model(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )


def build_model(cfg: dict, mode: str, rank: int | None = None):
    """Return a sequence classifier, full fine-tune or LoRA.

    The classification head is randomly initialized by Hugging Face and is
    always trainable. LoRA adapters are added on the query and value
    projections; every other backbone weight stays frozen.
    """
    model = load_sequence_model(cfg["model_name"], cfg["num_labels"])
    spec = {
        "mode": mode,
        "rank": None,
        "lora_alpha": None,
        "target_modules": None,
        "modules_to_save": None,
    }
    if mode == "full":
        return model, spec
    if mode != "lora":
        raise ValueError(f"Unknown mode {mode}")
    if rank is None:
        raise ValueError("LoRA requires a rank")

    targets = lora_target_modules(model, cfg.get("lora_target_modules"))
    head = head_modules(model)
    alpha = int(rank * cfg["lora_alpha_multiplier"])
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=cfg["lora_dropout"],
        target_modules=targets,
        modules_to_save=head,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    spec.update(
        {
            "rank": rank,
            "lora_alpha": alpha,
            "target_modules": targets,
            "modules_to_save": head,
        }
    )
    return model, spec
