#!/bin/bash
# Submit BridgeV2 action tokenizer sweep (OAT, QueST, VQ-BeT).
# 9 jobs total: 3 OAT + 3 QueST + 3 VQ-BeT.
#
# Usage:
#   bash scripts/submit_bridge_tokenizer_sweep.sh          # all 9 jobs
#   bash scripts/submit_bridge_tokenizer_sweep.sh oat      # OAT only
#   bash scripts/submit_bridge_tokenizer_sweep.sh quest    # QueST only
#   bash scripts/submit_bridge_tokenizer_sweep.sh vq_bet   # VQ-BeT only

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs checkpoints/bridge_sweep

MODE="${1:-all}"
SAVE_BASE="checkpoints/bridge_sweep"
TIME="12:00:00"

submit_oat() {
    local TAG="$1"
    local HORIZON="$2"
    local NREG="$3"
    local FSQ="$4"  # space-separated FSQ levels
    local JOB_NAME="br_oat_${TAG}"

    sbatch --job-name="$JOB_NAME" \
        --partition=general --gres=gpu:1 --time="$TIME" \
        --mem=32G --cpus-per-task=8 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/train_tokenizer.py \
    --tokenizer oat --dataset bridge \
    --epochs 500 --batch_size 256 --lr 5e-5 \
    --horizon $HORIZON --num_registers $NREG \
    --fsq_levels $FSQ \
    --max_chunks_per_epoch 100000 \
    --save_dir ${SAVE_BASE}/oat_${TAG} --tag ${TAG}
"
    echo "  Submitted: $JOB_NAME"
}

submit_quest() {
    local TAG="$1"
    local HORIZON="$2"
    local DS="$3"
    local FSQ="$4"
    local JOB_NAME="br_quest_${TAG}"

    sbatch --job-name="$JOB_NAME" \
        --partition=general --gres=gpu:1 --time="$TIME" \
        --mem=32G --cpus-per-task=8 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/train_tokenizer.py \
    --tokenizer quest --dataset bridge \
    --epochs 300 --batch_size 128 --lr 1e-4 \
    --horizon $HORIZON --downsample_factor $DS \
    --fsq_levels $FSQ \
    --max_chunks_per_epoch 100000 \
    --save_dir ${SAVE_BASE}/quest_${TAG} --tag ${TAG}
"
    echo "  Submitted: $JOB_NAME"
}

submit_vqbet() {
    local TAG="$1"
    local CHUNK="$2"
    local NEMBED="$3"
    local NGROUPS="$4"
    local LATENT="$5"
    local JOB_NAME="br_vqbet_${TAG}"

    sbatch --job-name="$JOB_NAME" \
        --partition=general --gres=gpu:1 --time="$TIME" \
        --mem=32G --cpus-per-task=8 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/train_tokenizer.py \
    --tokenizer vq_bet --dataset bridge \
    --epochs 200 --batch_size 256 --lr 1e-4 \
    --chunk_size $CHUNK --num_codes $NEMBED --vq_groups $NGROUPS \
    --latent_dim $LATENT \
    --max_chunks_per_epoch 100000 \
    --save_dir ${SAVE_BASE}/vqbet_${TAG} --tag ${TAG}
"
    echo "  Submitted: $JOB_NAME"
}

# === OAT (5 configs) ===
if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "=== OAT ==="
    # O1: horizon=16, 4 registers, FSQ=[8,5,5] → 200 codes
    submit_oat "h16_r4_v200" 16 4 "8 5 5"
    # O2: horizon=32, 8 registers, FSQ=[5,5,5] → 125 codes
    submit_oat "h32_r8_v125" 32 8 "5 5 5"
    # O3: horizon=32, 8 registers, FSQ=[8,8,8] → 512 codes
    submit_oat "h32_r8_v512" 32 8 "8 8 8"
    # O4: horizon=16, 4 registers, FSQ=[8,5,5,5] → 1000 codes
    submit_oat "h16_r4_v1000" 16 4 "8 5 5 5"
    # O5: horizon=32, 8 registers, FSQ=[8,8,8,8] → 4096 codes
    submit_oat "h32_r8_v4096" 32 8 "8 8 8 8"
fi

# === QueST (5 configs) ===
if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "=== QueST ==="
    # Q1: horizon=16, ds=4 → 4 tokens, FSQ=[8,5,5] → 200 codes
    submit_quest "h16_ds4_v200" 16 4 "8 5 5"
    # Q2: horizon=32, ds=4 → 8 tokens, FSQ=[5,5,5] → 125 codes
    submit_quest "h32_ds4_v125" 32 4 "5 5 5"
    # Q3: horizon=32, ds=4 → 8 tokens, FSQ=[8,8,8] → 512 codes
    submit_quest "h32_ds4_v512" 32 4 "8 8 8"
    # Q4: horizon=16, ds=4 → 4 tokens, FSQ=[8,5,5,5] → 1000 codes
    submit_quest "h16_ds4_v1000" 16 4 "8 5 5 5"
    # Q5: horizon=32, ds=4 → 8 tokens, FSQ=[8,8,8,8] → 4096 codes
    submit_quest "h32_ds4_v4096" 32 4 "8 8 8 8"
fi

# === VQ-BeT (3 configs) ===
if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "=== VQ-BeT ==="
    # B1: chunk=5, n_embed=16, groups=2, latent=256
    submit_vqbet "c5_e16_g2_l256" 5 16 2 256
    # B2: chunk=5, n_embed=16, groups=2, latent=512
    submit_vqbet "c5_e16_g2_l512" 5 16 2 512
    # B3: chunk=10, n_embed=16, groups=2, latent=256
    submit_vqbet "c10_e16_g2_l256" 10 16 2 256
fi

echo "=== Done ==="
