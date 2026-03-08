#!/bin/bash
# Submit VQ-VAE chunk tokenizer sweep: K in {2,4,8} x num_codes in {256,512}
# Each combo: (1) CPU fitting job, (2) GPU training job dependent on fitting.
# All training jobs use the standing convention: --min_class_count 30 --weighted_loss

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

echo "Submitting VQ-VAE sweep: K={2,4,8} x num_codes={256,512}"
echo "========================================================="

for CHUNK_SIZE in 2 4 8; do
    for NUM_CODES in 256 512; do
        NAME="ao_vqvae_k${CHUNK_SIZE}_c${NUM_CODES}"
        VQVAE_PATH="./checkpoints/vqvae_k${CHUNK_SIZE}_c${NUM_CODES}"

        # --- Step 1: Fit VQ-VAE (CPU job) ---
        FIT_JID=$(sbatch \
            --job-name="vqvae_fit_k${CHUNK_SIZE}_c${NUM_CODES}" \
            --partition=cpu \
            --cpus-per-task=4 \
            --mem=16G \
            --time=01:00:00 \
            -o "logs/vqvae_fit_k${CHUNK_SIZE}_c${NUM_CODES}-%j.out" \
            -e "logs/vqvae_fit_k${CHUNK_SIZE}_c${NUM_CODES}-%j.err" \
            --parsable \
            --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
echo "Fitting VQ-VAE: chunk_size=${CHUNK_SIZE}, num_codes=${NUM_CODES}"
python vqvae_tokenizer.py \
    --chunk_size ${CHUNK_SIZE} \
    --num_codes ${NUM_CODES} \
    --save_path ${VQVAE_PATH} \
    --epochs 100
")
        echo "  [k=${CHUNK_SIZE}, c=${NUM_CODES}] Fitting job: ${FIT_JID}"

        # --- Step 2: Train transformer (GPU, depends on fitting) ---
        TRAIN_JID=$(sbatch \
            --dependency=afterok:${FIT_JID} \
            --job-name="${NAME}" \
            -o "logs/${NAME}-%j.out" \
            -e "logs/${NAME}-%j.err" \
            --parsable \
            run_experiment.sh \
                "${NAME}" action_only vq_vae \
                --vqvae_tokenizer_path "${VQVAE_PATH}" \
                --vqvae_chunk_size "${CHUNK_SIZE}" \
                --min_class_count 30 --weighted_loss)
        echo "  [k=${CHUNK_SIZE}, c=${NUM_CODES}] Training job: ${TRAIN_JID} (depends on ${FIT_JID})"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u wenjiel2"
echo "Results will be in results/ao_vqvae_k*_c*_best_metrics.json"
