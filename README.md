# LLM fine-tuning and optimization

Reproducible comparison of full transformer fine-tuning and LoRA/PEFT on GLUE SST-2, plus a data-parallel training benchmark and an inference latency/throughput sweep.

No LLM API key is required. The run downloads the open `distilbert-base-uncased` checkpoint and the public GLUE SST-2 split from Hugging Face.

## What it measures

- **Training cost.** AdamW moments and gradients scale with the number of trainable parameters, and so does the data-parallel all-reduce. The gate is a reduction of at least 60% against full fine-tuning.
- **Quality.** Validation accuracy has to stay within 1 percentage point of the fully fine-tuned model. Binary F1 is reported alongside it.
- **Inference.** Latency (p50/p95) and throughput across batch sizes, for merged and unmerged LoRA weights.
- **Distributed training.** A real two-process DDP loop (gloo) records all-reduce volume and step time.

LoRA is applied to the query and value projections. The classification head is trained in every run because DistilBERT's sequence-classification head is randomly initialized.

<!-- RESULTS:START -->
## Results

Measured on Apple MPS (macOS-26.6.2-arm64-arm-64bit) with `distilbert-base-uncased` fine-tuned on GLUE SST-2 (8,000 train examples, 872 validation examples, 3 epochs, sequence length 64, seed 42).

**Operating point: LoRA r=16 (3 ep).** Optimizer and gradient training cost was 1.3% of full fine-tuning, a **98.7% reduction** (target 60.0%). Validation accuracy was **88.2%** for LoRA r=16 (3 ep) and **89.1%** for full fine-tuning (lower by 0.92 percentage points; quality band is 1.0 points). Binary F1 was 88.4% for LoRA r=16 (3 ep) and 89.4% for full fine-tuning.

Wall-clock training time was 543.7s for full fine-tuning (44.1 examples/s) and 269.8s for LoRA r=16 (3 ep) (89.0 examples/s). Peak training-memory overhead was 632 MB for full fine-tuning and 539 MB for LoRA r=16 (3 ep).

### Training comparison

| Run | Epochs | Trainable params | Train state | Cost vs full | Val accuracy | Val F1 | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full fine-tune | 3 | 66,955,010 | 803.46 MB | baseline | 89.1% | 89.4% | 543.7s |
| LoRA r=4 (3 ep) | 3 | 665,858 | 7.99 MB | 99.0% lower | 87.6% | 87.8% | 278.3s |
| LoRA r=8 (3 ep) | 3 | 739,586 | 8.88 MB | 98.9% lower | 87.5% | 87.8% | 375.6s |
| LoRA r=16 (3 ep) | 3 | 887,042 | 10.64 MB | 98.7% lower | 88.2% | 88.4% | 269.8s |

### Inference trade-offs

Latency and throughput below are for the merged model, measured by rotating through every checkpoint so each one sees the same inputs and the same GPU state. Merging folds the adapters into the query and value matrices, which is the network you would deploy.

| Run | Batch 1 latency | Batch 8 throughput | Batch 32 throughput |
| --- | ---: | ---: | ---: |
| Full fine-tune | 71.60 ms | 107 ex/s | 228 ex/s |
| LoRA r=4 (3 ep) | 74.52 ms | 105 ex/s | 235 ex/s |
| LoRA r=8 (3 ep) | 72.88 ms | 116 ex/s | 238 ex/s |
| LoRA r=16 (3 ep) | 75.41 ms | 103 ex/s | 242 ex/s |

### Distributed training

Data-parallel training used 2 processes, backend `gloo`, on CPU. Each step is a real forward, backward, and AdamW update with a ring all-reduce of the gradient. All-reduce volume below is `2 * (n - 1) / n` times the fp32 gradient.

| Run | Trainable params | All-reduce / step | Mean step time | First loss | Last loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full fine-tune | 66,955,010 | 267.820 MB | 1564.7 ms | 0.6924 | 0.5555 |
| LoRA r=8 | 739,586 | 2.958 MB | 361.1 ms | 0.6924 | 0.4852 |

Gradient all-reduce volume for LoRA r=8 was 98.9% lower than full fine-tuning. Mean step time was 1564.7 ms for full fine-tuning and 361.1 ms for LoRA.

### Cost definition

Training cost is the AdamW state plus the gradient, multiplied by the number of optimizer steps: `trainable_params * (8 bytes of moments + 4 bytes of gradient) * steps`. Frozen backbone weights do not enter that quantity. The same trainable-parameter count sets distributed all-reduce volume. Wall-clock time and peak process memory are reported beside it because a single-machine forward pass still executes the full network.

Gate: cost reduction ≥ 60.0% and accuracy drop ≤ 1.0 percentage points. Result: **PASSED**.
<!-- RESULTS:END -->

## Reproduce

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest tests.test_metrics -v
python -m src.run_experiment
```

`--quick` trains a 256-example subset for one epoch so the pipeline can be checked without the full sweep. The committed numbers come from the default config in `configs/experiment.json`.

On a CUDA machine, the same single-run entrypoint is data-parallel under torchrun:

```bash
torchrun --nproc_per_node=2 -m src.train_one \
  --config configs/experiment.json --mode lora --rank 8 --output outputs/runs/lora_r8
```

## Layout

| Path | Role |
| --- | --- |
| `configs/experiment.json` | Model, data budget, LoRA ranks, gates |
| `src/train_one.py` | One full or LoRA fine-tune, evaluation, inference benchmark |
| `src/distributed.py` | Two-process DDP cost benchmark |
| `src/report.py` | Tables, figure, README results block |
| `src/metrics.py` | Cost formulas and the pass/fail gate |
| `results/` | Metrics, report, and trade-off figure from the last run |

Checkpoints and adapters are written under `outputs/` and are not part of the git history. `results/metrics.json` is the source for the numbers in this README.
