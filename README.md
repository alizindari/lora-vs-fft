# LoRA vs Full Fine-Tuning

Comparing LoRA and full fine-tuning on Qwen 2.5 and RoBERTa models. Includes SVD analysis of weight updates to understand the intrinsic dimensionality of fine-tuned models.

## Setup

```bash
uv sync
```

Data files go in `data/` (not tracked by git). For Qwen, place `commonsense_170k.json` there. GLUE datasets are downloaded automatically.

## Supported Models

- **Qwen 2.5**: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`
- **RoBERTa**: `roberta-base`, `roberta-large`

## Pipelines

**Qwen (causal LM)**: Fine-tune on commonsense_170k, evaluate with lm-eval-harness on 8 benchmarks (ARC, BoolQ, PIQA, WinoGrande, HellaSwag, OpenBookQA, CommonsenseQA).

**RoBERTa (classification)**: Fine-tune and evaluate on GLUE tasks (SST-2, CoLA, MRPC, STS-B, QNLI, RTE, MNLI).

## Usage

### Training

Edit the config at the top of `jobs/job_train.sh`, then submit:

```bash
sbatch jobs/job_train.sh
```

Outputs are saved to `saves/<model>/<task>/<method>/<hparams>/`.

### Evaluation

Edit `jobs/job_eval.sh` to point to a checkpoint, then:

```bash
sbatch jobs/job_eval.sh
```

For LoRA checkpoints, the adapter is merged with the base model before evaluation.

### SVD Analysis

Computes SVD of the weight delta (W_finetuned - W_base) for each linear layer type:

```bash
sbatch jobs/job_svd.sh
```

Generates spectral decay and cumulative variance plots in `plots/`.

## Project Structure

```
src/
  train.py          # Training: Qwen causal LM + RoBERTa GLUE classification
  eval.py           # Evaluation: lm-eval-harness (Qwen) + GLUE dev set (RoBERTa)
  svd_full_ft.py    # SVD analysis of weight deltas
jobs/
  job_train.sh      # SLURM training job
  job_eval.sh       # SLURM evaluation job
  job_svd.sh        # SLURM SVD analysis job
pyproject.toml      # Dependencies
data/               # Datasets (not tracked)
```
