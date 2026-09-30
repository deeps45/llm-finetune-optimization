"""GLUE SST-2 loading and tokenization."""

from __future__ import annotations

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_splits(cfg: dict, tokenizer):
    """Tokenize SST-2. Train is shuffled and capped; validation stays labeled."""
    raw = load_dataset(cfg["dataset"], cfg["dataset_config"])
    train = raw["train"]
    eval_ds = raw["validation"]
    seed = cfg["seed"]
    text_column = cfg["text_column"]

    train_cap = cfg.get("train_samples")
    if train_cap and train_cap < len(train):
        train = train.shuffle(seed=seed).select(range(train_cap))

    eval_cap = cfg.get("eval_samples")
    if eval_cap and eval_cap < len(eval_ds):
        eval_ds = eval_ds.shuffle(seed=seed).select(range(eval_cap))

    eval_texts = list(eval_ds[text_column])

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=cfg["max_length"],
        )

    drop = [column for column in train.column_names if column != "label"]
    train = train.map(tokenize, batched=True, remove_columns=drop)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=drop)
    train = train.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return train, eval_ds, eval_texts, collator
