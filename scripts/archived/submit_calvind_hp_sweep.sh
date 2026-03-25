#!/bin/bash
# Submit CALVIN-D tokenizer hyperparameter sweep (18 configs).
#
# Sweep axes: codebook size, tokens per chunk, horizon/chunk length.
# All vanilla (no aux heads) — lambda sweep comes after picking winners.
#
# Usage:
#   bash scripts/submit_calvind_hp_sweep.sh           # all 18 jobs
#   bash scripts/submit_calvind_hp_sweep.sh vq_bet    # just VQ-BeT (6 jobs)
#   bash scripts/submit_calvind_hp_sweep.sh oat       # just OAT (6 jobs)
#   bash scripts/submit_calvind_hp_sweep.sh quest     # just QueST (6 jobs)

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs checkpoints/calvind_hp_sweep

MODE="${1:-all}"
SAVE_BASE="checkpoints/calvind_hp_sweep"
EPOCHS=200
BS=64
TIME="08:00:00"

submit_job() {
    local JOB_NAME="$1"
    local TOK="$2"
    local TAG="$3"
    local EXTRA_ARGS="$4"

    sbatch --job-name="hp_${JOB_NAME}" \
        --partition=general --gres=gpu:1 --time="$TIME" \
        --mem=32G --cpus-per-task=8 \
        --output="logs/hp_${JOB_NAME}_%j.out" \
        --error="logs/hp_${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/train_tokenizer.py \
    --tokenizer $TOK \
    --tag $TAG \
    --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_BASE \
    $EXTRA_ARGS
"
    echo "  Submitted: hp_${JOB_NAME}"
}

# =====================================================================
# VQ-BeT: sweep chunk_size × (n_embed, groups)
# Fixed: latent_dim=512, hidden_dim=128, num_mlp_layers=1
# =====================================================================
if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "=== VQ-BeT (6 configs) ==="

    # V1: chunk=5, n_embed=16, groups=2 → 2 tok, 256 combos [BASELINE]
    submit_job "vb_c5_e16_g2" "vq_bet" "c5_e16_g2" \
        "--chunk_size 5 --latent_dim 512 --num_codes 16 --vq_groups 2"

    # V2: chunk=5, n_embed=16, groups=4 → 4 tok, 65K combos
    submit_job "vb_c5_e16_g4" "vq_bet" "c5_e16_g4" \
        "--chunk_size 5 --latent_dim 512 --num_codes 16 --vq_groups 4"

    # V3: chunk=5, n_embed=64, groups=2 → 2 tok, 4096 combos
    submit_job "vb_c5_e64_g2" "vq_bet" "c5_e64_g2" \
        "--chunk_size 5 --latent_dim 512 --num_codes 64 --vq_groups 2"

    # V4: chunk=10, n_embed=16, groups=2 → 2 tok, 256 combos
    submit_job "vb_c10_e16_g2" "vq_bet" "c10_e16_g2" \
        "--chunk_size 10 --latent_dim 512 --num_codes 16 --vq_groups 2"

    # V5: chunk=10, n_embed=16, groups=4 → 4 tok, 65K combos
    submit_job "vb_c10_e16_g4" "vq_bet" "c10_e16_g4" \
        "--chunk_size 10 --latent_dim 512 --num_codes 16 --vq_groups 4"

    # V6: chunk=10, n_embed=64, groups=2 → 2 tok, 4096 combos
    submit_job "vb_c10_e64_g2" "vq_bet" "c10_e64_g2" \
        "--chunk_size 10 --latent_dim 512 --num_codes 64 --vq_groups 2"
fi

# =====================================================================
# OAT: sweep horizon × (fsq_levels, num_registers)
# Fixed: emb_dim=256, enc_depth=2, dec_depth=4, head_dim=64
# =====================================================================
if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "=== OAT (6 configs) ==="

    # O1: h=32, FSQ[8,5,5,5]=1000, regs=8 → 8 tok [BASELINE]
    submit_job "oat_h32_f1000_r8" "oat" "h32_f1000_r8" \
        "--horizon 32 --fsq_levels 8 5 5 5 --num_registers 8"

    # O2: h=32, FSQ[4,4,4,4]=256, regs=8 → 8 tok
    submit_job "oat_h32_f256_r8" "oat" "h32_f256_r8" \
        "--horizon 32 --fsq_levels 4 4 4 4 --num_registers 8"

    # O3: h=32, FSQ[4,4,4,4]=256, regs=4 → 4 tok
    submit_job "oat_h32_f256_r4" "oat" "h32_f256_r4" \
        "--horizon 32 --fsq_levels 4 4 4 4 --num_registers 4"

    # O4: h=32, FSQ[4,4,4]=64, regs=4 → 4 tok
    submit_job "oat_h32_f64_r4" "oat" "h32_f64_r4" \
        "--horizon 32 --fsq_levels 4 4 4 --num_registers 4"

    # O5: h=16, FSQ[4,4,4,4]=256, regs=4 → 4 tok
    submit_job "oat_h16_f256_r4" "oat" "h16_f256_r4" \
        "--horizon 16 --fsq_levels 4 4 4 4 --num_registers 4"

    # O6: h=16, FSQ[4,4,4,4]=256, regs=8 → 8 tok
    submit_job "oat_h16_f256_r8" "oat" "h16_f256_r8" \
        "--horizon 16 --fsq_levels 4 4 4 4 --num_registers 8"
fi

# =====================================================================
# QueST: sweep horizon × (fsq_levels, downsample_factor)
# Fixed: encoder_dim=256, decoder_dim=256, enc_layers=2, dec_layers=4
# =====================================================================
if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "=== QueST (6 configs) ==="

    # Q1: h=32, FSQ[8,5,5,5]=1000, ds=4 → 8 tok [BASELINE]
    submit_job "quest_h32_f1000_d4" "quest" "h32_f1000_d4" \
        "--horizon 32 --fsq_levels 8 5 5 5 --downsample_factor 4"

    # Q2: h=32, FSQ[4,4,4,4]=256, ds=4 → 8 tok
    submit_job "quest_h32_f256_d4" "quest" "h32_f256_d4" \
        "--horizon 32 --fsq_levels 4 4 4 4 --downsample_factor 4"

    # Q3: h=32, FSQ[4,4,4,4]=256, ds=8 → 4 tok
    submit_job "quest_h32_f256_d8" "quest" "h32_f256_d8" \
        "--horizon 32 --fsq_levels 4 4 4 4 --downsample_factor 8"

    # Q4: h=32, FSQ[4,4,4]=64, ds=4 → 8 tok
    submit_job "quest_h32_f64_d4" "quest" "h32_f64_d4" \
        "--horizon 32 --fsq_levels 4 4 4 --downsample_factor 4"

    # Q5: h=16, FSQ[4,4,4,4]=256, ds=4 → 4 tok
    submit_job "quest_h16_f256_d4" "quest" "h16_f256_d4" \
        "--horizon 16 --fsq_levels 4 4 4 4 --downsample_factor 4"

    # Q6: h=16, FSQ[4,4,4,4]=256, ds=2 → 8 tok
    submit_job "quest_h16_f256_d2" "quest" "h16_f256_d2" \
        "--horizon 16 --fsq_levels 4 4 4 4 --downsample_factor 2"
fi

echo ""
echo "=== Summary ==="
echo "Checkpoints: ${SAVE_BASE}/"
echo "Monitor: squeue -u $(whoami)"
echo ""
echo "Sweep grid:"
echo "  VQ-BeT: chunk_size={5,10} × (n_embed,groups)={(16,2),(16,4),(64,2)}"
echo "  OAT:    horizon={16,32} × fsq={64,256,1000} × regs={4,8}"
echo "  QueST:  horizon={16,32} × fsq={64,256,1000} × ds={2,4,8}"
