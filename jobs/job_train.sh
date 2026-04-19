#!/bin/bash
#SBATCH --job-name=ft-train
#SBATCH --output=job-train-%j.out
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
# NOTE: set --gres gpu count to match NUM_GPUS below

#############################################
# CHANGE ONLY THESE
#############################################

# --- Model & data ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}" # Qwen/Qwen2.5-{0.5B,1.5B,3B,7B} | roberta-{base,large}
GLUE_TASK=""                             # sst2|cola|mrpc|stsb|qnli|rte|mnli (empty = Qwen causal LM)
DATA_PATH="${DATA_PATH:-data/commonsense_170k.json}"   # Qwen only; env-overridable (e.g. DATA_PATH=data/boolq_alpaca.json)
MAX_SAMPLES="${MAX_SAMPLES:-1000}"       # "" = use all samples (override via env: MAX_SAMPLES=X sbatch ...)

# --- Method (env-overridable) ---
FINETUNING_TYPE="${FINETUNING_TYPE:-lora}"    # full | lora
LORA_RANK="${LORA_RANK:-1}"                   # rank of LoRA decomposition
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}" # LoRA scaling factor (default 2*rank)
LORA_DROPOUT=0.0                         # dropout on LoRA layers
LORA_TARGET="all-linear"                 # "all-linear" | comma-separated layer names

# --- Training ---
OPTIMIZER="adamw"                        # adamw | sgd
NUM_EPOCHS=3.0                           # total training epochs
LEARNING_RATE="${LEARNING_RATE:-2.0e-4}" # peak learning rate (env-overridable)
WEIGHT_DECAY=0.01                        # L2 regularization
WARMUP_RATIO=0.02                        # fraction of steps for LR warmup
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}" # batch size per GPU
GRAD_ACCUM="${GRAD_ACCUM:-4}"             # gradient accumulation steps (effective_bs = batch * gpus * accum)
LR_SCHEDULER="cosine"                    # cosine | linear | constant
MAX_GRAD_NORM=1.0                        # gradient clipping threshold
SGD_MOMENTUM=0.9                         # only used if OPTIMIZER=sgd
CUTOFF_LEN=512                           # max token sequence length

# --- Seeds ---
DATA_SEED=42                             # fixed: controls data subset & train/val split (same across methods)
SEED=$RANDOM                             # random: controls weight init & batch order (varies for error bars)

# --- Logging & infrastructure ---
LOGGING_STEPS=20
NUM_GPUS=1                               # must match --gres gpu count above

# EVAL_STEPS / SAVE_STEPS scale with dataset size so there are ~3 evals total
# (roughly one per epoch) regardless of how many samples are in the run.
# Overridable by env if you want more or fewer checkpoints.
if [ -n "${MAX_SAMPLES:-}" ]; then
    _train_size=$(( MAX_SAMPLES * 80 / 100 ))
    _effective_bs=$(( PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM ))
    _steps_per_epoch=$(( (_train_size + _effective_bs - 1) / _effective_bs ))
    [ $_steps_per_epoch -lt 1 ] && _steps_per_epoch=1
    EVAL_STEPS="${EVAL_STEPS:-$_steps_per_epoch}"
    SAVE_STEPS="${SAVE_STEPS:-$_steps_per_epoch}"
else
    EVAL_STEPS="${EVAL_STEPS:-200}"
    SAVE_STEPS="${SAVE_STEPS:-200}"
fi

# Reference configs:
#   Qwen 0.5B LoRA: MODEL_NAME="Qwen/Qwen2.5-0.5B"  GLUE_TASK=""      LR=2e-4  EP=3   R=1   A=2   BS=8  GA=4
#   Qwen 3B LoRA:   MODEL_NAME="Qwen/Qwen2.5-3B"     GLUE_TASK=""      LR=2e-5  EP=3   R=64  A=128 BS=1  GA=8
#   RoBERTa SST-2:  MODEL_NAME="roberta-base"          GLUE_TASK="sst2"  LR=5e-4  EP=60  R=8   A=16  BS=16 GA=1  WD=0.1 WR=0.06 DROPOUT=0.05

#############################################
# DO NOT CHANGE BELOW
#############################################

set -euo pipefail

# --- Ensure we run from the project root ---
cd "$SLURM_SUBMIT_DIR"

# --- Derive short model name: "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b", "roberta-base" -> "roberta-base" ---
MODEL_SHORT=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# --- Install UV if needed ---
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Sync dependencies (recreate venv to avoid cross-node staleness on HPC) ---
# Use uv-managed Python so dev headers (Python.h) are available (some compute
# nodes' system Python lacks them, which breaks triton JIT if it ever runs).
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON=3.12
# Per-job venv on node-local scratch. NFS destroys us on small-file ops:
# some nodes take 30-50 min just to rm -rf the venv. Local /tmp is ~instant,
# and uv sync is also faster because wheel installs don't hammer NFS.
VENV_SCRATCH="${TMPDIR:-/tmp}"
VENV_DIR="${VENV_SCRATCH}/lora-venv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
# Flush Python stdout line-by-line so tqdm/print show up in the log while
# training runs (default is block-buffered when redirected to file).
export PYTHONUNBUFFERED=1
echo "Syncing Python environment into $VENV_DIR ..."
rm -rf "$VENV_DIR"
# deepspeed pulls in triton's eager-compiled ops; only install it when actually
# needed for multi-GPU, so single-GPU runs don't require a C compiler on the node.
if [ "$NUM_GPUS" -gt 1 ] && [ -z "$GLUE_TASK" ]; then
    uv sync --extra multi_gpu
else
    uv sync
fi

# --- Build output directory ---
# Format: saves/<model>/<task>/<method>/<optimizer>_lr<lr>_ep<epochs>_bs<bs>_seed<seed>
EFFECTIVE_BS=$((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))
SAMPLES_TAG="${MAX_SAMPLES:-all}"

if [ -n "$GLUE_TASK" ]; then
    TASK_DIR="${GLUE_TASK}"
else
    DATASET_SHORT=$(basename "$DATA_PATH" .json)
    TASK_DIR="${DATASET_SHORT}_${SAMPLES_TAG}"
fi

if [ "$FINETUNING_TYPE" = "lora" ]; then
    METHOD_DIR="lora_r${LORA_RANK}_a${LORA_ALPHA}"
else
    METHOD_DIR="full"
fi

OUTPUT_DIR="saves/${MODEL_SHORT}/${TASK_DIR}/${METHOD_DIR}/${OPTIMIZER}_lr${LEARNING_RATE}_ep${NUM_EPOCHS}_bs${EFFECTIVE_BS}_seed${SEED}"

# --- Move job log into the run directory on exit (success or failure); clean venv ---
trap '
    set +e
    mkdir -p "$OUTPUT_DIR" 2>/dev/null
    mv "job-train-${SLURM_JOB_ID}.out" "${OUTPUT_DIR}/train.log" 2>/dev/null
    rm -rf "$VENV_DIR" 2>/dev/null
' EXIT

# --- Build command ---
CMD="uv run python src/train.py \
    --model_name $MODEL_NAME \
    --finetuning_type $FINETUNING_TYPE \
    --cutoff_len $CUTOFF_LEN \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler $LR_SCHEDULER \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --per_device_batch $PER_DEVICE_BATCH \
    --grad_accum $GRAD_ACCUM \
    --optimizer $OPTIMIZER \
    --sgd_momentum $SGD_MOMENTUM \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --data_seed $DATA_SEED"

# Add GLUE task or data path
if [ -n "$GLUE_TASK" ]; then
    CMD="$CMD --glue_task $GLUE_TASK"
else
    CMD="$CMD --data_path $DATA_PATH"
fi

# Add LoRA args
if [ "$FINETUNING_TYPE" = "lora" ]; then
    CMD="$CMD \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target $LORA_TARGET"
fi

# Add max_samples if set
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# --- DeepSpeed for multi-GPU (Qwen only) ---
if [ "$NUM_GPUS" -gt 1 ] && [ -z "$GLUE_TASK" ]; then
    cat > /tmp/ds_config_${SLURM_JOB_ID}.json << 'DSEOF'
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
DSEOF
    CMD="$CMD --deepspeed /tmp/ds_config_${SLURM_JOB_ID}.json"
    CMD=$(echo "$CMD" | sed "s|uv run python src/train.py|uv run torchrun --nproc_per_node=$NUM_GPUS src/train.py|")
fi

# --- Print config summary ---
echo "========================================"
echo "JOB CONFIGURATION"
echo "========================================"
echo "Model:          $MODEL_NAME"
if [ -n "$GLUE_TASK" ]; then
    echo "Pipeline:       classification (GLUE)"
    echo "Task:           $GLUE_TASK"
else
    echo "Pipeline:       causal LM (commonsense)"
    echo "Dataset:        $DATA_PATH"
    echo "Max samples:    ${MAX_SAMPLES:-all}"
fi
echo "Method:         $FINETUNING_TYPE"
if [ "$FINETUNING_TYPE" = "lora" ]; then
    echo "LoRA rank:      $LORA_RANK"
    echo "LoRA alpha:     $LORA_ALPHA"
fi
echo "Optimizer:      $OPTIMIZER"
echo "LR:             $LEARNING_RATE"
echo "Epochs:         $NUM_EPOCHS"
echo "Output:         $OUTPUT_DIR"
echo "Data seed:      $DATA_SEED"
echo "Train seed:     $SEED"
echo "========================================"

# --- Preflight checks ---
if [ ! -f src/train.py ]; then
    echo "ERROR: src/train.py not found in $PWD" >&2
    echo "Submit this job from the project root: sbatch jobs/job_train.sh" >&2
    exit 1
fi

# --- Run ---
eval $CMD

echo "Done. Output saved to: $OUTPUT_DIR"
