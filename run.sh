#!/bin/bash
#SBATCH --job-name=verb-clf
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
mkdir -p logs checkpoints

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

echo "===== Job Info ====="
echo "Job ID:    $SLURM_JOB_ID"
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
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --max_seq_len 64 \
    --save_path ./checkpoints/model.pth \
    $DEBUG_FLAG

# --- Test ---
echo ""
echo "===== Starting Testing ====="
python test_transformer.py \
    --model_path ./checkpoints/model.pth \
    --max_seq_len 64 \
    --save_cm ./checkpoints/confusion_matrix.png \
    $DEBUG_FLAG

echo ""
echo "===== Done ====="
