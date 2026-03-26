# LoRA vs Full Fine-Tuning

Comparing LoRA and full fine-tuning on Qwen2.5 models across multiple classification benchmarks. Includes SVD analysis of weight updates to understand the intrinsic dimensionality of fine-tuned models.

## Setup

Designed for JUWELS Booster (4x A100 GPUs per node). Requires a Python venv with dependencies from `pyproject.toml`.

```bash
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5 CUDA/13
uv venv /path/to/venv --python python3
source /path/to/venv/bin/activate
uv pip install -e .
```

Models must be pre-cached in `~/.cache/huggingface/hub/` (compute nodes have no internet).

## Supported Models

| Model | Short name |
|---|---|
| `Qwen/Qwen2.5-0.5B` | `qwen-0.5b` |
| `Qwen/Qwen2.5-1.5B` | `qwen-1.5b` |
| `Qwen/Qwen2.5-3B` | `qwen-3b` |
| `Qwen/Qwen2.5-7B` | `qwen-7b` |
| `meta-llama/Llama-2-7b-hf` | `llama-7b` |

## Datasets

All datasets are pre-converted to alpaca format (`data/*_alpaca.json`):

- **SST-2** — binary sentiment classification
- **BoolQ** — yes/no reading comprehension
- **ARC-Easy** — elementary science (multiple choice)
- **ARC-Challenge** — harder science (multiple choice)
- **CommonsenseQA** — commonsense reasoning (multiple choice)

## Usage

### Training

Edit the top of `job_train.sh` to set model, dataset, method (full/lora), and hyperparameters, then submit:

```bash
sbatch job_train.sh
```

Checkpoints are saved to `new_saves/{model}/{dataset}/{method}_lr{lr}_wd{wd}_{sched}_ep{ep}/`.

Key settings:
- `FINETUNING_TYPE`: `"full"` or `"lora"`
- `LORA_RANK`: LoRA rank (default 64, only used for LoRA)
- `LEARNING_RATE`, `WEIGHT_DECAY`, `LR_SCHEDULER`, `WARMUP_RATIO`
- `NUM_EPOCHS`, `PER_DEVICE_BATCH`, `GRAD_ACCUM`

Uses DeepSpeed ZeRO Stage 3 for distributed training across 4 GPUs.

### Evaluation

Edit `job_eval.sh` to point to a trained checkpoint, then:

```bash
sbatch job_eval.sh
```

For LoRA checkpoints, the adapter is automatically merged with the base model before evaluation. Uses [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

### SVD Analysis

Computes SVD of the weight delta (W_finetuned - W_base) for each linear layer type to analyze the intrinsic dimensionality of full fine-tuning updates:

```bash
sbatch job_svd.sh
```

Generates singular value decay and cumulative explained variance plots in `plots/`.

## Project Structure

```
train.py          # Training script (~140 lines) — Trainer + PEFT
svd_full_ft.py    # SVD analysis of weight deltas
job_train.sh      # SLURM job for training
job_eval.sh       # SLURM job for lm-eval evaluation
job_svd.sh        # SLURM job for SVD analysis
pyproject.toml    # Dependencies
data/             # Alpaca-format datasets
```
