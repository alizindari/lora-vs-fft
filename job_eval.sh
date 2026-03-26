#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --output=logs/job-eval-%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --account=hai_1260

#############################################
# CHANGE ONLY THESE
#############################################

FT_TYPE="lora"                          # "lora" or "full"
# Options: Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B, meta-llama/Llama-2-7b-hf
BASE_MODEL_NAME="Qwen/Qwen2.5-7B"

# LoRA only
LORA_PATH="new_saves/qwen-7b/sst2/lora_64/checkpoint-200"

# Full FT only
FULL_MODEL_PATH="new_saves/qwen-7b/sst2/full/checkpoint-200"

TASK="sst2"
BATCH_SIZE=8
NUM_FEWSHOT=0

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

MERGED_DIR="/tmp/merged_model_${SLURM_JOB_ID}"

echo '========================================'
echo 'EVALUATION CONFIGURATION'
echo '========================================'
echo "Task:           $TASK"
echo "Model:          $BASE_MODEL_NAME"
echo "Method:         $FT_TYPE"

if [ "$FT_TYPE" = "lora" ]; then
  echo "LoRA path:      $LORA_PATH"
  echo '========================================'

  echo "=== MERGING LORA ADAPTER ==="
  srun python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained('${BASE_MODEL_NAME}', torch_dtype=torch.float16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('${BASE_MODEL_NAME}')
model = PeftModel.from_pretrained(model, '${LORA_PATH}')
model = model.merge_and_unload()
model.save_pretrained('${MERGED_DIR}')
tokenizer.save_pretrained('${MERGED_DIR}')
print('Merged model saved to ${MERGED_DIR}')
"
  EVAL_MODEL="$MERGED_DIR"
else
  echo "Checkpoint:     $FULL_MODEL_PATH"
  echo '========================================'
  EVAL_MODEL="$FULL_MODEL_PATH"
fi

echo "=== RUNNING LM-EVAL ==="
srun lm_eval \
    --model hf \
    --model_args "pretrained=$EVAL_MODEL" \
    --tasks "$TASK" \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --num_fewshot $NUM_FEWSHOT \
    --output_path "new_saves/eval_results/${TASK}_${SLURM_JOB_ID}"

echo "Done."
