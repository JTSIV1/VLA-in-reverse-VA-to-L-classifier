#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

# --- Generic single-experiment runner ---
# Usage: sbatch --job-name=<name> -o logs/<name>-%j.out -e logs/<name>-%j.err \
#          run_experiment.sh <name> <modality> <action_rep> [--fast_tokenizer_path PATH] [--debug N]
#
# Examples:
#   sbatch run_experiment.sh action_only action_only native
#   sbatch run_experiment.sh full_fast_v512 full fast --fast_tokenizer_path ./checkpoints/fast_tokenizer_v512

NAME="${1:?Usage: run_experiment.sh <name> <modality> <action_rep> [options]}"
MODALITY="${2:?Missing modality (full|action_only|vision_only)}"
ACTION_REP="${3:?Missing action_rep (native|fast)}"
shift 3

# Parse optional flags
FAST_TOK_FLAG=""
DEBUG_FLAG=""
CROSS_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast_tokenizer_path)
            FAST_TOK_FLAG="--fast_tokenizer_path $2"
            shift 2
            ;;
        --cross_layers)
            CROSS_FLAG="--cross_layers $2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug ${2:-32}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64"

# Build tag: {NAME}_j{JOB_ID}[_debug]
TAG="${NAME}_j${SLURM_JOB_ID:-local}"
if [[ -n "$DEBUG_FLAG" ]]; then
    TAG="${TAG}_debug"
fi

echo "===== Experiment: ${NAME} ====="
echo "Tag: ${TAG}"
echo "Modality: ${MODALITY} | Action rep: ${ACTION_REP}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- Train ---
python train_transformer.py \
    --modality "$MODALITY" --action_rep "$ACTION_REP" \
    --save_path "./checkpoints/${TAG}.pth" \
    --log_path "./results/${TAG}_log.json" \
    $COMMON $FAST_TOK_FLAG $CROSS_FLAG $DEBUG_FLAG

# --- Test ---
python test_transformer.py \
    --model_path "./checkpoints/${TAG}.pth" \
    --save_cm "./figures/${TAG}_cm.png" \
    --save_metrics "./results/${TAG}_metrics.json" \
    $FAST_TOK_FLAG $DEBUG_FLAG

echo ">>> ${NAME} done — checkpoint: checkpoints/${TAG}.pth | results: results/${TAG}_metrics.json"
