#!/bin/bash
#SBATCH --job-name=verb-clf
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# --- Usage: sbatch run.sh [--debug N] ---
# Pass --debug N to smoke test with only N samples
DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_FLAG="--debug ${2:-32}"
    echo "*** DEBUG MODE: ${2:-32} samples ***"
fi

# --- Setup ---
PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

# Build tag: full_native_j{JOB_ID}[_debug]
TAG="full_native_j${SLURM_JOB_ID:-local}"
if [[ -n "$DEBUG_FLAG" ]]; then
    TAG="${TAG}_debug"
fi

echo "===== Job Info ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Tag:       $TAG"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python:    $(python --version)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "==================="

# --- Train ---
echo ""
echo "===== Starting Training ====="
python train_transformer.py \
    --epochs 30 \
    --batch_size 16 \
    --lr 5e-4 \
    --max_seq_len 64 \
    --save_path "./checkpoints/${TAG}.pth" \
    --log_path "./results/${TAG}_log.json" \
    $DEBUG_FLAG

# --- Test ---
echo ""
echo "===== Starting Testing ====="
python test_transformer.py \
    --model_path "./checkpoints/${TAG}.pth" \
    --max_seq_len 64 \
    --save_cm "./figures/${TAG}_cm.png" \
    --save_metrics "./results/${TAG}_metrics.json" \
    $DEBUG_FLAG

echo ""
echo "===== Done ====="
