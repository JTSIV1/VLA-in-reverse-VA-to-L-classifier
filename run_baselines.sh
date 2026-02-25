#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/baselines-%j.out
#SBATCH --error=logs/baselines-%j.err

set -euo pipefail

# --- Usage: sbatch run_baselines.sh [--debug N] ---
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

COMMON="--epochs 10 --batch_size 16 --lr 1e-4 --max_seq_len 64"

echo "===== Job Info ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python:    $(python --version)"
echo "==================="

# =====================================================================
# Step 0: Fit FAST tokenizer on training data
# =====================================================================
echo ""
echo "===== Fitting FAST Tokenizer ====="
python fast_tokenizer.py --save_path ./checkpoints/fast_tokenizer $DEBUG_FLAG

# =====================================================================
# Step 1: Action-only (native continuous actions)
# =====================================================================
echo ""
echo "===== [1/4] Action-Only (native) ====="
python train_transformer.py \
    --modality action_only --action_rep native \
    --save_path ./checkpoints/action_only.pth \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path ./checkpoints/action_only.pth \
    --save_cm ./checkpoints/action_only_cm.png \
    --save_metrics ./checkpoints/action_only_metrics.json \
    $DEBUG_FLAG

echo ">>> action_only done — results in checkpoints/action_only_metrics.json"

# =====================================================================
# Step 2: Vision-only (first + last frame, no actions)
# =====================================================================
echo ""
echo "===== [2/4] Vision-Only ====="
python train_transformer.py \
    --modality vision_only \
    --save_path ./checkpoints/vision_only.pth \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path ./checkpoints/vision_only.pth \
    --save_cm ./checkpoints/vision_only_cm.png \
    --save_metrics ./checkpoints/vision_only_metrics.json \
    $DEBUG_FLAG

echo ">>> vision_only done — results in checkpoints/vision_only_metrics.json"

# =====================================================================
# Step 3: Full multimodal + FAST tokens
# =====================================================================
echo ""
echo "===== [3/4] Full + FAST ====="
python train_transformer.py \
    --modality full --action_rep fast \
    --save_path ./checkpoints/full_fast.pth \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path ./checkpoints/full_fast.pth \
    --save_cm ./checkpoints/full_fast_cm.png \
    --save_metrics ./checkpoints/full_fast_metrics.json \
    $DEBUG_FLAG

echo ">>> full_fast done — results in checkpoints/full_fast_metrics.json"

# =====================================================================
# Step 4: Action-only + FAST tokens
# =====================================================================
echo ""
echo "===== [4/4] Action-Only + FAST ====="
python train_transformer.py \
    --modality action_only --action_rep fast \
    --save_path ./checkpoints/action_only_fast.pth \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path ./checkpoints/action_only_fast.pth \
    --save_cm ./checkpoints/action_only_fast_cm.png \
    --save_metrics ./checkpoints/action_only_fast_metrics.json \
    $DEBUG_FLAG

echo ">>> action_only_fast done — results in checkpoints/action_only_fast_metrics.json"

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "===== All Baselines Complete ====="
echo "Results saved in ./checkpoints/*_metrics.json"
echo "Confusion matrices in ./checkpoints/*_cm.png"
echo "===== Done ====="
