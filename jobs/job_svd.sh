#!/bin/bash
#SBATCH --job-name=svd-fullft
#SBATCH --output=logs/job-svd-%j.out
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

#############################################
# CHANGE ONLY THESE
#############################################

# Options: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B
BASE_MODEL_NAME="Qwen/Qwen2.5-3B"
MODEL_SHORT="qwen-3b"

# Path to a single checkpoint
FT_CHECKPOINT="new_saves/qwen-3b-real/arcc/full_kgp3jx/checkpoint-84"

# Options: sst2, boolq, arc_easy, arc_challenge, commonsense_qa
DATASET="arc_challenge"

#############################################
# DO NOT CHANGE BELOW
#############################################

mkdir -p logs

JOBTMPDIR=/tmp/job-$SLURM_JOB_ID
mkdir -p "$JOBTMPDIR"

srun \
--container-image=projects.cispa.saarland:5005#c01alzi/lora-ft:v45 \
--container-mounts="$PWD":/workspace,"$JOBTMPDIR":/tmp \
bash -c "
set -e
cd /workspace

# Save base model weights locally for safetensors loading
BASE_MODEL_DIR=base_models/${MODEL_SHORT}
if [ -d \"\$BASE_MODEL_DIR\" ] && ls \$BASE_MODEL_DIR/model*.safetensors 1>/dev/null 2>&1; then
  echo \"=== BASE MODEL ALREADY CACHED at \$BASE_MODEL_DIR ===\"
else
  echo '=== SAVING BASE MODEL WEIGHTS ==='
  mkdir -p \$BASE_MODEL_DIR
  python3 -c \"
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL_NAME', torch_dtype=torch.float16)
model.save_pretrained('\$BASE_MODEL_DIR')
print('Base model saved to \$BASE_MODEL_DIR')
\"
fi

echo '========================================'
echo 'SVD ANALYSIS — FULL FT'
echo '========================================'
echo 'Base model:     $BASE_MODEL_NAME'
echo 'FT checkpoint:  $FT_CHECKPOINT'
echo '========================================'

python3 src/svd_full_ft.py \
    --base_model_path \$BASE_MODEL_DIR \
    --ft_checkpoint_path $FT_CHECKPOINT \
    --model_name '$BASE_MODEL_NAME' \
    --dataset '$DATASET'
"

echo "Done."
