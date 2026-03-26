#!/bin/bash
#SBATCH --job-name=svd-fullft
#SBATCH --output=logs/job-svd-%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --account=hai_1260

#############################################
# CHANGE ONLY THESE
#############################################

# Options: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B, meta-llama/Llama-2-7b-hf
BASE_MODEL_NAME="Qwen/Qwen2.5-7B"
MODEL_SHORT="qwen-7b"
# Options: sst2, boolq, arc_easy, arc_challenge, commonsense_qa
DATASET="sst2"

#############################################
# DO NOT CHANGE BELOW
#############################################

mkdir -p logs

module purge
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5 CUDA/13

VENV_DIR="/p/project1/hai_1260/zindari1/lora-vs-fft/venv"
source "$VENV_DIR/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Save base model weights locally for safetensors loading
BASE_MODEL_DIR="base_models/${MODEL_SHORT}"
if [ ! -d "$BASE_MODEL_DIR" ] || ! ls "$BASE_MODEL_DIR"/model*.safetensors 1>/dev/null 2>&1; then
  echo "=== SAVING BASE MODEL WEIGHTS ==="
  python3 -c "
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('${BASE_MODEL_NAME}', torch_dtype=torch.float16)
model.save_pretrained('${BASE_MODEL_DIR}')
print('Base model saved to ${BASE_MODEL_DIR}')
"
fi

# Find the last checkpoint
FT_DIR="new_saves/${MODEL_SHORT}/${DATASET}/full"
LAST_CKPT=$(ls -d ${FT_DIR}/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)
FT_PATH="${FT_DIR}/checkpoint-${LAST_CKPT}"

OUTPUT_DIR="plots/${MODEL_SHORT}/${DATASET}/full"
mkdir -p "$OUTPUT_DIR"

echo '========================================'
echo 'SVD ANALYSIS — FULL FT'
echo '========================================'
echo "Base model:     $BASE_MODEL_NAME"
echo "FT checkpoint:  $FT_PATH"
echo "Output:         $OUTPUT_DIR"
echo '========================================'

srun python3 svd_full_ft.py \
    --base_model_path "$BASE_MODEL_DIR" \
    --ft_model_path "$FT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$BASE_MODEL_NAME" \
    --dataset "$DATASET"

echo "Done."
