#!/bin/bash
#SBATCH --job-name=ft-eval
#SBATCH --output=job-eval-%j.out
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00

#############################################
# CHANGE ONLY THESE
#############################################

# --- What to evaluate (env-overridable: FT_TYPE=X BASE_MODEL=Y CHECKPOINT_PATH=Z sbatch ...) ---
FT_TYPE="${FT_TYPE:-lora}"                           # full | lora | base (base = no fine-tuning)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"          # HF model name (required for lora and base)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-saves/qwen2.5-3b/commonsense_170k_all/lora_r64_a128/adamw_lr2.0e-5_ep3.0_bs8_seed42/checkpoint-500}"
                                         # path to fine-tuned checkpoint (ignored if base)

# --- Task ---
GLUE_TASK=""                             # sst2|cola|mrpc|stsb|qnli|rte|mnli (empty = Qwen lm-eval)
TASKS="${TASKS:-arc_easy,arc_challenge,boolq,commonsense_qa,piqa,winogrande,hellaswag,openbookqa}"
                                         # lm-eval benchmarks; env-overridable (e.g. TASKS=boolq)

# --- Eval settings ---
BATCH_SIZE=8                             # eval batch size
NUM_FEWSHOT=0                            # few-shot examples for lm-eval (Qwen only)

#############################################
# DO NOT CHANGE BELOW
#############################################

set -euo pipefail

# --- Ensure we run from the project root ---
cd "$SLURM_SUBMIT_DIR"

# --- Install UV if needed ---
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Sync dependencies ---
# uv-managed Python (dev headers for triton JIT), per-job venv on node-local
# scratch (NFS is ~50x slower on small-file ops), unbuffered Python stdout.
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON=3.12
VENV_SCRATCH="${TMPDIR:-/tmp}"
VENV_DIR="${VENV_SCRATCH}/lora-venv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export PYTHONUNBUFFERED=1
echo "Syncing Python environment into $VENV_DIR ..."
rm -rf "$VENV_DIR"
uv sync

# --- Output path: save results inside checkpoint/run directory ---
if [ "$FT_TYPE" = "base" ]; then
    BASE_SHORT=$(echo "$BASE_MODEL" | tr '/' '-')
    OUTPUT_JSON="saves/base_${BASE_SHORT}/eval_results.json"
else
    OUTPUT_JSON="${CHECKPOINT_PATH}/eval_results.json"
fi

LOG_DIR=$(dirname "$OUTPUT_JSON")

# --- Move job log into the run directory on exit (success or failure) ---
trap '
    set +e
    mkdir -p "$LOG_DIR" 2>/dev/null
    mv "job-eval-${SLURM_JOB_ID}.out" "${LOG_DIR}/eval.log" 2>/dev/null
' EXIT

# --- Build command ---
CMD="uv run python src/eval.py --batch_size $BATCH_SIZE --output_path $OUTPUT_JSON"

# Add GLUE flag if set
if [ -n "$GLUE_TASK" ]; then
    CMD="$CMD --glue_task $GLUE_TASK"
else
    CMD="$CMD --tasks $TASKS --num_fewshot $NUM_FEWSHOT"
fi

# Add model source
if [ "$FT_TYPE" = "lora" ]; then
    CMD="$CMD --base_model $BASE_MODEL --lora_path $CHECKPOINT_PATH"
elif [ "$FT_TYPE" = "full" ]; then
    CMD="$CMD --model_path $CHECKPOINT_PATH"
elif [ "$FT_TYPE" = "base" ]; then
    CMD="$CMD --base_model $BASE_MODEL"
else
    echo "ERROR: FT_TYPE must be 'full', 'lora', or 'base'"
    exit 1
fi

# --- Print config ---
echo "========================================"
echo "EVALUATION CONFIGURATION"
echo "========================================"
echo "Type:           $FT_TYPE"
if [ "$FT_TYPE" != "base" ]; then
    echo "Checkpoint:     $CHECKPOINT_PATH"
fi
if [ "$FT_TYPE" != "full" ]; then
    echo "Base model:     $BASE_MODEL"
fi
if [ -n "$GLUE_TASK" ]; then
    echo "Pipeline:       classification (GLUE)"
    echo "Task:           $GLUE_TASK"
else
    echo "Pipeline:       causal LM (lm-eval)"
    echo "Tasks:          $TASKS"
fi
echo "Output:         $OUTPUT_JSON"
echo "========================================"

# --- Run ---
eval $CMD

echo "Done."
