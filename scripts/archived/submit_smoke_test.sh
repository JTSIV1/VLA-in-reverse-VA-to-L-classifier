#!/bin/bash
# Smoke test: train each tokenizer for 2 epochs with verb head to verify
# the refactored train_tokenizer.py and eval_tokenizer.py work correctly.
#
# Also prints latent shapes and token distributions per tokenizer.
#
# Usage:
#   bash scripts/submit_smoke_test.sh

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs checkpoints/smoke_test

EPOCHS=2
BS=16
TIME="00:30:00"
SAVE_BASE="checkpoints/smoke_test"

submit_job() {
    local JOB_NAME="$1"
    local TOK="$2"
    local EXTRA_ARGS="${3:-}"

    sbatch --job-name="smoke_${JOB_NAME}" \
        --partition=general --gres=gpu:1 --time="$TIME" \
        --mem=32G --cpus-per-task=4 \
        --output="logs/smoke_${JOB_NAME}_%j.out" \
        --error="logs/smoke_${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/train_tokenizer.py \
    --tokenizer $TOK \
    --tag smoke_${JOB_NAME} \
    --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_BASE \
    --aux_head verb --aux_lambda 0.5 \
    --min_class_count 30 --max_chunks 8 \
    $EXTRA_ARGS
"
    echo "  Submitted: smoke_${JOB_NAME}"
}

echo "=== Smoke Test: tokenizer refactor ==="

# VQ-BeT (chunk_size=4 from YAML)
submit_job "vq_bet" "vq_bet"

# OAT (chunk_size=32 from YAML)
submit_job "oat" "oat"

# QueST (chunk_size=32 from YAML)
submit_job "quest" "quest"

echo ""
echo "=== 3 jobs submitted ==="
echo "Monitor: squeue -u $(whoami)"
echo "Logs: logs/smoke_*"
