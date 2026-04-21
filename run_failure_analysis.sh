#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

set -euo pipefail

# ─── Usage ───────────────────────────────────────────────────────────────────
# sbatch --job-name=<name> -o logs/<name>-%j.out -e logs/<name>-%j.err \
#   run_failure_analysis.sh <name> <modality> <action_rep> [extra train flags...]
#
# Examples:
#   sbatch run_failure_analysis.sh action_only action_only native
#   sbatch run_failure_analysis.sh scene_obs   scene_mlp   native --scene_dim 48
#   sbatch run_failure_analysis.sh fusion      scene_token native --scene_dim 48

NAME="${1:?Usage: run_failure_analysis.sh <name> <modality> <action_rep> [opts]}"
MODALITY="${2:?}"
ACTION_REP="${3:?}"
shift 3
EXTRA_TRAIN_FLAGS="$*"    # any remaining flags forwarded to train_transformer.py

PROJECT_DIR="${HOME}/11777"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results/failure_analysis figures

# ─── Environment ─────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

echo "======================================================================"
echo "  Job: ${SLURM_JOB_ID:-local}  Name: ${NAME}"
echo "  Modality: ${MODALITY}  Action rep: ${ACTION_REP}"
echo "  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "======================================================================"

TAG="${NAME}_j${SLURM_JOB_ID:-local}"
CKPT_BASE="./checkpoints/${TAG}"
LOG_PATH="./results/${TAG}_log.json"
OUT_DIR="./results/failure_analysis/${NAME}"

COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64 \
        --weighted_loss --min_class_count 30 --num_workers 4"

# ─── Step 1: Train ───────────────────────────────────────────────────────────
echo ""
echo ">>> [1/3] Training ${NAME} ..."
python train_transformer.py \
    --modality "$MODALITY" \
    --action_rep "$ACTION_REP" \
    --save_path "${CKPT_BASE}.pth" \
    --log_path  "$LOG_PATH" \
    $COMMON $EXTRA_TRAIN_FLAGS

# ─── Step 2: Evaluate best checkpoint ────────────────────────────────────────
echo ""
echo ">>> [2/3] Evaluating best checkpoint ..."
BEST_CKPT="${CKPT_BASE}_best.pth"
if [[ ! -f "$BEST_CKPT" ]]; then
    echo "[warn] _best.pth not found, falling back to final checkpoint"
    BEST_CKPT="${CKPT_BASE}.pth"
fi
python test_transformer.py \
    --model_path "$BEST_CKPT" \
    --save_cm "./figures/${TAG}_best_cm.png" \
    --save_metrics "./results/${TAG}_best_metrics.json" \
    --save_preds "./results/${TAG}_best_preds.json"

# ─── Step 3: Failure analysis ────────────────────────────────────────────────
echo ""
echo ">>> [3/3] Running failure analysis for ${NAME} ..."
python analyze_failures.py \
    --model_path "$BEST_CKPT" \
    --out_dir    "$OUT_DIR" \
    --top_k 5    \
    --num_workers 4

echo ""
echo "======================================================================"
echo "  DONE: ${NAME}"
echo "  Checkpoint : ${BEST_CKPT}"
echo "  Failures   : ${OUT_DIR}/top_failures.json"
echo "  CM         : ${OUT_DIR}/confusion_matrix.png"
echo "======================================================================"
