#!/bin/bash
#SBATCH --job-name=encoders
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/encoders-%j.out
#SBATCH --error=logs/encoders-%j.err

set -euo pipefail

# --- Usage: sbatch run_encoders.sh [--debug N] ---
DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_FLAG="--debug ${2:-32}"
    echo "*** DEBUG MODE: ${2:-32} samples ***"
fi

# --- Setup ---
PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

# Install encoder-specific deps (no-ops if already installed)
pip install timm --quiet                  # required for --image_encoder dinov2
pip install r3m --quiet                   # required for --image_encoder r3m

COMMON="--modality full --action_rep native --epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64"

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
# Step 1: Scratch (lower bound — learned from scratch)
# =====================================================================
echo ""
echo "===== [1/4] Scratch PatchEmbed ====="
python train_transformer.py \
    --image_encoder scratch \
    --save_path "./checkpoints/enc_scratch${TAG_SUFFIX}.pth" \
    --log_path  "./results/enc_scratch${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/enc_scratch${TAG_SUFFIX}.pth" \
    --save_cm    "./figures/enc_scratch${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/enc_scratch${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> enc_scratch done"

# =====================================================================
# Step 2: ResNet-18 (ImageNet pretrained, classical baseline)
# =====================================================================
echo ""
echo "===== [2/4] ResNet-18 ====="
python train_transformer.py \
    --image_encoder resnet18 \
    --save_path "./checkpoints/enc_resnet18${TAG_SUFFIX}.pth" \
    --log_path  "./results/enc_resnet18${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/enc_resnet18${TAG_SUFFIX}.pth" \
    --save_cm    "./figures/enc_resnet18${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/enc_resnet18${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> enc_resnet18 done"

# =====================================================================
# Step 3: DINOv2 (modern strong baseline)
# =====================================================================
echo ""
echo "===== [3/4] DINOv2 ====="
python train_transformer.py \
    --image_encoder dinov2 \
    --save_path "./checkpoints/enc_dinov2${TAG_SUFFIX}.pth" \
    --log_path  "./results/enc_dinov2${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/enc_dinov2${TAG_SUFFIX}.pth" \
    --save_cm    "./figures/enc_dinov2${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/enc_dinov2${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> enc_dinov2 done"

# =====================================================================
# Step 4: R3M (robotics-specific representation)
# Requires: pip install r3m
# =====================================================================
echo ""
echo "===== [4/4] R3M ====="
python train_transformer.py \
    --image_encoder r3m \
    --save_path "./checkpoints/enc_r3m${TAG_SUFFIX}.pth" \
    --log_path  "./results/enc_r3m${TAG_SUFFIX}_log.json" \
    $COMMON $DEBUG_FLAG

python test_transformer.py \
    --model_path "./checkpoints/enc_r3m${TAG_SUFFIX}.pth" \
    --save_cm    "./figures/enc_r3m${TAG_SUFFIX}_cm.png" \
    --save_metrics "./results/enc_r3m${TAG_SUFFIX}_metrics.json" \
    $DEBUG_FLAG

echo ">>> enc_r3m done"

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "===== All Encoder Ablations Complete ====="
echo "Checkpoints: ./checkpoints/enc_*${TAG_SUFFIX}.pth"
echo "Results:     ./results/enc_*${TAG_SUFFIX}_metrics.json"
echo "Figures:     ./figures/enc_*${TAG_SUFFIX}_cm.png"
echo "===== Done ====="
