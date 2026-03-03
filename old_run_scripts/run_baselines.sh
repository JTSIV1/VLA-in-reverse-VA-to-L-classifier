#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
mkdir -p logs checkpoints results figures

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64"

# Build tag suffix: _j{JOB_ID}[_debug]
TAG_SUFFIX="_j${SLURM_JOB_ID:-local}"
if [[ -n "$DEBUG_FLAG" ]]; then
    TAG_SUFFIX="${TAG_SUFFIX}_debug"
fi

echo "===== Job Info ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Tag suffix: $TAG_SUFFIX"
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
    --save_path "./checkpoints/action_only${TAG_SUFFIX}.pth" \
    --log_path "./results/action_only${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/action_only${TAG_SUFFIX}.pth" \
    --save_cm "./figures/action_only${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/action_only${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> action_only done"

# =====================================================================
# Step 2: Vision-only (first + last frame, no actions)
# =====================================================================
echo ""
echo "===== [2/4] Vision-Only ====="
python train_transformer.py \
    --modality vision_only \
    --save_path "./checkpoints/vision_only${TAG_SUFFIX}.pth" \
    --log_path "./results/vision_only${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/vision_only${TAG_SUFFIX}.pth" \
    --save_cm "./figures/vision_only${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/vision_only${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> vision_only done"

# =====================================================================
# Step 3: Full multimodal + FAST tokens
# =====================================================================
echo ""
echo "===== [3/4] Full + FAST ====="
python train_transformer.py \
    --modality full --action_rep fast \
    --save_path "./checkpoints/full_fast${TAG_SUFFIX}.pth" \
    --log_path "./results/full_fast${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/full_fast${TAG_SUFFIX}.pth" \
    --save_cm "./figures/full_fast${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/full_fast${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> full_fast done"

# =====================================================================
# Step 4: Action-only + FAST tokens
# =====================================================================
echo ""
echo "===== [4/4] Action-Only + FAST ====="
python train_transformer.py \
    --modality action_only --action_rep fast \
    --save_path "./checkpoints/action_only_fast${TAG_SUFFIX}.pth" \
    --log_path "./results/action_only_fast${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/action_only_fast${TAG_SUFFIX}.pth" \
    --save_cm "./figures/action_only_fast${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/action_only_fast${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> action_only_fast done"

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "===== All Baselines Complete ====="
echo "Checkpoints: ./checkpoints/*${TAG_SUFFIX}.pth"
echo "Results:     ./results/*${TAG_SUFFIX}_metrics.json"
echo "Figures:     ./figures/*${TAG_SUFFIX}_cm.png"
echo "===== Done ====="
