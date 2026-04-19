#!/bin/bash
#SBATCH --job-name=ft-plot
#SBATCH --output=job-plot-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=2

#############################################
# CHANGE ONLY THESE (env-overridable)
#############################################

SAVES_ROOT="${SAVES_ROOT:-saves/qwen2.5-0.5b}"
DATASET="${DATASET:-boolq_alpaca}"      # matches <dataset>_<n> dirs under SAVES_ROOT
TASK="${TASK:-boolq}"                   # lm-eval task name to plot accuracy for
OUTPUT="${OUTPUT:-figures/${DATASET}_${TASK}_sample_size_sweep.png}"
TITLE="${TITLE:-}"
YLABEL="${YLABEL:-}"
STYLE="${STYLE:-accuracy}"              # 'accuracy' or 'paper'
BASELINE_PATH="${BASELINE_PATH:-saves/base_Qwen-Qwen2.5-0.5B/eval_results.json}"

#############################################
# DO NOT CHANGE BELOW
#############################################

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- Install UV if needed ---
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Per-job venv on node-local scratch (NFS is slow on small-file ops) ---
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON=3.12
VENV_DIR="${TMPDIR:-/tmp}/lora-venv-${SLURM_JOB_ID}"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export PYTHONUNBUFFERED=1

trap '
    set +e
    rm -rf "$VENV_DIR" 2>/dev/null
' EXIT

echo "Syncing Python environment into $VENV_DIR ..."
rm -rf "$VENV_DIR"
uv sync

# --- Build command ---
CMD="uv run python src/plot_sweep.py \
    --saves_root $SAVES_ROOT \
    --dataset $DATASET \
    --task $TASK \
    --output $OUTPUT \
    --style $STYLE"

[ -n "$BASELINE_PATH" ] && CMD="$CMD --baseline_path $BASELINE_PATH"

[ -n "$TITLE" ]  && CMD="$CMD --title \"$TITLE\""
[ -n "$YLABEL" ] && CMD="$CMD --ylabel \"$YLABEL\""

echo "========================================"
echo "PLOT CONFIGURATION"
echo "========================================"
echo "Saves root:  $SAVES_ROOT"
echo "Dataset:     $DATASET"
echo "Task:        $TASK"
echo "Output:      $OUTPUT"
echo "========================================"

eval $CMD
echo "Done."
