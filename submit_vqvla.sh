#!/bin/bash
# Submit VQ-VLA pretrained action tokenizer experiment (action_only).
#
# Step 1: Download pretrained weights from HuggingFace (CPU job, ~1.4 GB).
# Step 2: Train classifier with frozen VQ-VLA tokenizer (GPU job).
#
# Both use the standing convention: --min_class_count 30 --weighted_loss.
# VQ-VLA produces 4 tokens per trajectory (fixed by pretrained architecture).

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
CKPT_DIR="${PROJECT_DIR}/checkpoints/vqvla_pretrained"
CKPT_PATH="${CKPT_DIR}/action_tokenizer_weight/all_data_vq.pth"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

echo "=== VQ-VLA experiment ==="
echo "Pretrained weights will be saved to: ${CKPT_PATH}"

# --- Step 1: Download pretrained weights (CPU job) ---
DL_JID=$(sbatch \
    --job-name="vqvla_download" \
    --partition=cpu \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    -o "logs/vqvla_download-%j.out" \
    -e "logs/vqvla_download-%j.err" \
    --parsable \
    --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
python -c \"
from vqvae_tokenizer import download_pretrained_vqvla
download_pretrained_vqvla('${CKPT_DIR}')
\"
")
echo "Download job: ${DL_JID}"

# --- Step 2: Train classifier (GPU, depends on download) ---
TRAIN_JID=$(sbatch \
    --dependency=afterok:${DL_JID} \
    --job-name="ao_vqvla" \
    -o "logs/ao_vqvla-%j.out" \
    -e "logs/ao_vqvla-%j.err" \
    --parsable \
    run_experiment.sh \
        ao_vqvla action_only vqvla \
        --vqvla_checkpoint_path "${CKPT_PATH}" \
        --min_class_count 30 --weighted_loss)
echo "Training job: ${TRAIN_JID} (depends on ${DL_JID})"

echo ""
echo "Monitor: squeue -u wenjiel2"
echo "Results: results/ao_vqvla_j*_best_metrics.json"
