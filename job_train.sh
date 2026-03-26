#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --output=logs/job-train-%j.out
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --time=1-00:00:00
#SBATCH --account=hai_1260
# NOTE: --gres GPU count must match NUM_GPUS below

#############################################
# CHANGE ONLY THESE
#############################################

# --- Model (cached locally) ---
# Options: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B, meta-llama/Llama-2-7b-hf
MODEL_NAME="Qwen/Qwen2.5-7B"
MODEL_SHORT="qwen-7b"

# --- Dataset (files in data/) ---
# Options: sst2, boolq, arc_easy, arc_challenge, commonsense_qa
DATASET="sst2"
CUTOFF_LEN=128

# --- Method ---
FINETUNING_TYPE="full"        # "full" or "lora"
LORA_RANK=64                  # only used if FINETUNING_TYPE=lora
LORA_TARGET="all"             # only used if FINETUNING_TYPE=lora

# --- Training hyperparameters ---
NUM_EPOCHS=1.0
LEARNING_RATE=1.0e-5
WEIGHT_DECAY=0.0
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.05
PER_DEVICE_BATCH=1
GRAD_ACCUM=4

# --- Logging & saving ---
LOGGING_STEPS=20
EVAL_STEPS=40
SAVE_STEPS=40

# --- Infrastructure ---
NUM_GPUS=4

#############################################
# DO NOT CHANGE BELOW
#############################################

# Derive output directory with hyperparameters
if [ "$FINETUNING_TYPE" = "lora" ]; then
  OUTPUT_DIR="new_saves/${MODEL_SHORT}/${DATASET}/lora_r${LORA_RANK}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_${LR_SCHEDULER}_ep${NUM_EPOCHS}"
else
  OUTPUT_DIR="new_saves/${MODEL_SHORT}/${DATASET}/full_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_${LR_SCHEDULER}_ep${NUM_EPOCHS}"
fi

export NPROC_PER_NODE=$NUM_GPUS
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

mkdir -p logs

module purge
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5 CUDA/13

VENV_DIR="/p/project1/hai_1260/zindari1/lora-vs-fft/venv"
source "$VENV_DIR/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- DeepSpeed config (ZeRO Stage 3) ---
cat > /tmp/ds_config.json << 'DSEOF'
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": "auto"
  },
  "fp16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
DSEOF

echo '========================================'
echo 'TRAINING CONFIGURATION'
echo '========================================'
echo "Model:          $MODEL_NAME"
echo "Dataset:        $DATASET"
echo "Method:         $FINETUNING_TYPE"
if [ "$FINETUNING_TYPE" = "lora" ]; then
  echo "LoRA rank:      $LORA_RANK"
  echo "LoRA target:    $LORA_TARGET"
fi
echo "Output:         $OUTPUT_DIR"
echo "Epochs:         $NUM_EPOCHS"
echo "LR:             $LEARNING_RATE"
echo "Weight decay:   $WEIGHT_DECAY"
echo "LR scheduler:   $LR_SCHEDULER"
echo "Warmup ratio:   $WARMUP_RATIO"
echo "Batch size:     $PER_DEVICE_BATCH x $NUM_GPUS GPUs x $GRAD_ACCUM accum = $(($PER_DEVICE_BATCH * $NUM_GPUS * $GRAD_ACCUM))"
echo "DeepSpeed:      ZeRO Stage 3"
echo '========================================'

srun torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "data/${DATASET}_alpaca.json" \
    --output_dir "$OUTPUT_DIR" \
    --finetuning_type "$FINETUNING_TYPE" \
    --lora_rank $LORA_RANK \
    --lora_target "$LORA_TARGET" \
    --cutoff_len $CUTOFF_LEN \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler "$LR_SCHEDULER" \
    --warmup_ratio $WARMUP_RATIO \
    --per_device_batch $PER_DEVICE_BATCH \
    --grad_accum $GRAD_ACCUM \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --deepspeed /tmp/ds_config.json

echo "Done."
