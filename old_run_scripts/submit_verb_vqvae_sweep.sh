#!/bin/bash
# Verb-decodable VQ-VAE sweep
#
# Loss scales at init: recon+vq ≈ 0.20, verb CE ≈ 3.0 (21 classes).
# lambda=0.01 → verb contributes ~0.03 (recon-dominated)
# lambda=0.05 → verb contributes ~0.15 (roughly balanced)
# lambda=0.1  → verb contributes ~0.30 (verb starts to dominate)
# lambda=0.5  → verb contributes ~1.50 (verb-dominated)
# lambda=1.0  → verb contributes ~3.00 (strongly verb-dominated)
#
# Phase 1 (7 jobs): lambda sweep at fixed architecture
#   - Includes vanilla baseline (no --verb_decodable) for direct comparison
#   - Fixed: K=4, C=512 (best vanilla VQ-VAE setting)
#
# Phase 2 (4 jobs): classifier capacity at best lambda
#   - Uncomment after Phase 1 results; sweep cls_layers

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

CHUNK_SIZE=4
NUM_CODES=512

# ── Helper: submit a two-stage fit→train pipeline ──
submit_verb_vqvae() {
    local NAME="$1"
    local VQVAE_PATH="$2"
    local LAMBDA="$3"
    local EXTRA_FIT_ARGS="${4:-}"

    # Step 1: Fit verb-decodable VQ-VAE (GPU)
    FIT_JID=$(sbatch \
        --job-name="vqfit_${NAME}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=16G \
        --time=02:00:00 \
        -o "logs/vqfit_${NAME}-%j.out" \
        -e "logs/vqfit_${NAME}-%j.err" \
        --parsable \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
python vqvae_tokenizer.py \
    --verb_decodable \
    --chunk_size ${CHUNK_SIZE} \
    --num_codes ${NUM_CODES} \
    --save_path ${VQVAE_PATH} \
    --verb_loss_weight ${LAMBDA} \
    --min_class_count 30 \
    --epochs 200 \
    --lr 1e-3 \
    ${EXTRA_FIT_ARGS}
")
    echo "  [${NAME}] Fit: ${FIT_JID}"

    # Step 2: Downstream transformer (depends on fit)
    TRAIN_JID=$(sbatch \
        --dependency=afterok:${FIT_JID} \
        --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" \
        -e "logs/${NAME}-%j.err" \
        --parsable \
        old_run_scripts/run_experiment.sh \
            "${NAME}" action_only vq_vae \
            --vqvae_tokenizer_path "${VQVAE_PATH}" \
            --vqvae_chunk_size "${CHUNK_SIZE}" \
            --min_class_count 30 --weighted_loss)
    echo "  [${NAME}] Train: ${TRAIN_JID} (after ${FIT_JID})"
}

# ── Helper: submit vanilla VQ-VAE baseline ──
submit_vanilla() {
    local NAME="$1"
    local VQVAE_PATH="$2"

    FIT_JID=$(sbatch \
        --job-name="vqfit_${NAME}" \
        --partition=cpu \
        --cpus-per-task=4 \
        --mem=16G \
        --time=01:00:00 \
        -o "logs/vqfit_${NAME}-%j.out" \
        -e "logs/vqfit_${NAME}-%j.err" \
        --parsable \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
python vqvae_tokenizer.py \
    --chunk_size ${CHUNK_SIZE} \
    --num_codes ${NUM_CODES} \
    --save_path ${VQVAE_PATH} \
    --epochs 200 \
    --lr 1e-3
")
    echo "  [${NAME}] Fit (vanilla): ${FIT_JID}"

    TRAIN_JID=$(sbatch \
        --dependency=afterok:${FIT_JID} \
        --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" \
        -e "logs/${NAME}-%j.err" \
        --parsable \
        old_run_scripts/run_experiment.sh \
            "${NAME}" action_only vq_vae \
            --vqvae_tokenizer_path "${VQVAE_PATH}" \
            --vqvae_chunk_size "${CHUNK_SIZE}" \
            --min_class_count 30 --weighted_loss)
    echo "  [${NAME}] Train: ${TRAIN_JID} (after ${FIT_JID})"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: lambda sweep (fixed K=4, C=512, cls_layers=2, cls_d=128)
# ═══════════════════════════════════════════════════════════════════════════
echo "Phase 1: lambda sweep (K=${CHUNK_SIZE}, C=${NUM_CODES})"
echo "========================================================="

# Vanilla baseline (same 200-epoch budget for fair comparison)
submit_vanilla \
    "ao_vqvae_vanilla_k4_c512" \
    "./checkpoints/vqvae_vanilla_k4_c512"

# Lambda sweep
for LAMBDA in 0.01 0.05 0.1 0.5 1.0; do
    submit_verb_vqvae \
        "ao_vqvae_verb_l${LAMBDA}" \
        "./checkpoints/vqvae_verb_k4_c512_l${LAMBDA}" \
        "${LAMBDA}"
done

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: classifier capacity (uncomment after Phase 1)
# Uses best lambda from Phase 1; sweep cls_layers in {1, 3, 4}
# ═══════════════════════════════════════════════════════════════════════════
# BEST_LAMBDA=0.05  # ← update after Phase 1 results
#
# for CLS_LAYERS in 1 3 4; do
#     submit_verb_vqvae \
#         "ao_vqvae_verb_lBEST_cls${CLS_LAYERS}" \
#         "./checkpoints/vqvae_verb_k4_c512_lBEST_cls${CLS_LAYERS}" \
#         "${BEST_LAMBDA}" \
#         "--cls_layers ${CLS_LAYERS}"
# done

echo ""
echo "Phase 1: 6 pipelines submitted (1 vanilla + 5 verb-decodable)"
echo "Monitor: squeue -u wenjiel2"
echo "Results: results/ao_vqvae_*_best_metrics.json"
