#!/bin/bash
#SBATCH --job-name=cluster-%x
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j.err

# Per-combo worker script.  Called by run_cluster.sh via sbatch --export.
# Required env vars (passed via --export):
#   FEATURE_SOURCE : actions | images  (default: actions)
#   ACTION_REP     : native | fast | bin | quest | oat  (for actions)
#   IMAGE_ENCODER  : resnet18 | dinov2 | dinov2_s | dinov2_b | vc1 | r3m  (for images)
#   DELTA_PATCHES  : int  (default: 16, for images)
#   CLUSTER_METHOD : kmeans | agglomerative  (default: kmeans)
#   USE_PCA        : 1 | 0  (default: 1 — cluster on 99%-PCA features)

set -euo pipefail

PROJECT_DIR="/home/istepka/11777"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results/clustering

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

FEATURE_SOURCE="${FEATURE_SOURCE:-actions}"
CLUSTER_METHOD="${CLUSTER_METHOD:-kmeans}"
USE_PCA="${USE_PCA:-1}"
PCA_FLAG="--use_pca"
[[ "${USE_PCA}" == "0" ]] && PCA_FLAG="--no_use_pca"

echo "===== Cluster Job: FEATURE_SOURCE=${FEATURE_SOURCE} ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "======================================="

OUT_DIR="./results/clustering"   # base dir; subdirs created by cluster_analysis.py

if [[ "${FEATURE_SOURCE}" == "images" ]]; then
    IMAGE_ENCODER="${IMAGE_ENCODER:-resnet18}"
    DELTA_PATCHES="${DELTA_PATCHES:-16}"
    BATCH_SIZE="${BATCH_SIZE:-64}"

    echo ">>> Running image cluster analysis (encoder=${IMAGE_ENCODER}, delta=${DELTA_PATCHES}, method=${CLUSTER_METHOD}, pca=${USE_PCA}) ..."
    python -u cluster_analysis.py \
        --feature_source images \
        --image_encoder "${IMAGE_ENCODER}" \
        --delta_patches "${DELTA_PATCHES}" \
        --batch_size "${BATCH_SIZE}" \
        --out_dir "${OUT_DIR}" \
        --cluster_method "${CLUSTER_METHOD}" \
        ${PCA_FLAG}

elif [[ "${ACTION_REP:-native}" == "fast" ]]; then
    echo ">>> Fitting FAST tokenizer (vocab=${VOCAB_SIZE}, scale=${SCALE}) ..."
    python fast_tokenizer.py \
        --save_path "${TOKENIZER_PATH}" \
        --vocab_size "${VOCAB_SIZE}" \
        --scale "${SCALE}"

    echo ">>> Running cluster analysis (fast, vocab=${VOCAB_SIZE}, scale=${SCALE}, method=${CLUSTER_METHOD}, pca=${USE_PCA}) ..."
    python -u cluster_analysis.py \
        --feature_source actions \
        --max_len 64 \
        --out_dir "${OUT_DIR}" \
        --action_rep fast \
        --vocab_size "${VOCAB_SIZE}" \
        --scale "${SCALE}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --cluster_method "${CLUSTER_METHOD}" \
        ${PCA_FLAG}
else
    ACTION_REP="${ACTION_REP:-native}"
    echo ">>> Running cluster analysis (${ACTION_REP}, method=${CLUSTER_METHOD}, pca=${USE_PCA}) ..."
    python -u cluster_analysis.py \
        --feature_source actions \
        --max_len 64 \
        --out_dir "${OUT_DIR}" \
        --action_rep "${ACTION_REP}" \
        --cluster_method "${CLUSTER_METHOD}" \
        ${PCA_FLAG}
fi

echo "===== Done ====="
