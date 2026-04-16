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
MODEL_NAME="Qwen/Qwen2.5-0.5B"          # Qwen/Qwen2.5-{0.5B,1.5B,3B,7B} | roberta-{base,large}
GLUE_TASK=""                             # sst2|cola|mrpc|stsb|qnli|rte|mnli (empty = Qwen causal LM)
DATA_PATH="data/commonsense_170k.json"   # Qwen only (ignored if GLUE_TASK is set)
MAX_SAMPLES="1000"                       # "" = use all samples

# --- Method ---
FINETUNING_TYPE="lora"                   # full | lora
LORA_RANK=1                              # rank of LoRA decomposition
LORA_ALPHA=2                             # LoRA scaling factor (typically 2*rank)
LORA_DROPOUT=0.0                         # dropout on LoRA layers
LORA_TARGET="all-linear"                 # "all-linear" | comma-separated layer names

# --- Training ---
OPTIMIZER="adamw"                        # adamw | sgd
NUM_EPOCHS=3.0                           # total training epochs
LEARNING_RATE=2.0e-4                     # peak learning rate
WEIGHT_DECAY=0.01                        # L2 regularization
WARMUP_RATIO=0.02                        # fraction of steps for LR warmup
PER_DEVICE_BATCH=8                       # batch size per GPU
GRAD_ACCUM=4                             # gradient accumulation steps (effective_bs = batch * gpus * accum)
LR_SCHEDULER="cosine"                    # cosine | linear | constant
MAX_GRAD_NORM=1.0                        # gradient clipping threshold
SGD_MOMENTUM=0.9                         # only used if OPTIMIZER=sgd
CUTOFF_LEN=512                           # max token sequence length

# --- Seeds ---
DATA_SEED=42                             # fixed: controls data subset & train/val split (same across methods)
SEED=$RANDOM                             # random: controls weight init & batch order (varies for error bars)

# --- Logging & infrastructure ---
LOGGING_STEPS=20
EVAL_STEPS=50                            # Qwen only; RoBERTa evals every epoch
SAVE_STEPS=50                            # Qwen only; RoBERTa saves every epoch
NUM_GPUS=1                               # must match --gres gpu count above

# Reference configs:
#   Qwen 0.5B LoRA: MODEL_NAME="Qwen/Qwen2.5-0.5B"  GLUE_TASK=""      LR=2e-4  EP=3   R=1   A=2   BS=8  GA=4
#   Qwen 3B LoRA:   MODEL_NAME="Qwen/Qwen2.5-3B"     GLUE_TASK=""      LR=2e-5  EP=3   R=64  A=128 BS=1  GA=8
#   RoBERTa SST-2:  MODEL_NAME="roberta-base"          GLUE_TASK="sst2"  LR=5e-4  EP=60  R=8   A=16  BS=16 GA=1  WD=0.1 WR=0.06 DROPOUT=0.05

#############################################
# DO NOT CHANGE BELOW
#############################################

set -euo pipefail

# --- Derive short model name: "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b", "roberta-base" -> "roberta-base" ---
MODEL_SHORT=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# --- Install UV if needed ---
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Sync dependencies ---
echo "Syncing Python environment..."
uv sync

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

# --- Move job log into the run directory on exit (success or failure) ---
trap 'mkdir -p "$OUTPUT_DIR" 2>/dev/null; mv "job-train-${SLURM_JOB_ID}.out" "${OUTPUT_DIR}/train.log" 2>/dev/null || true' EXIT

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

# --- Run ---
eval $CMD

echo "Done. Output saved to: $OUTPUT_DIR"
