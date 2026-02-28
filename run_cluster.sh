#!/bin/bash
# Launcher: submits one SLURM job per tokenizer/encoder combo.
# Usage: bash run_cluster.sh
# Each job runs run_cluster_job.sh with its own env vars.
# Outputs go to results/clustering/ to keep things clean.
#
# ── SWITCH: set the clustering algorithm here ─────────────────────────────────
#   kmeans        → K-Means (k = #verbs)
#   agglomerative → Agglomerative Ward (k = #verbs, no spherical assumption)
CLUSTER_METHOD="${CLUSTER_METHOD:-kmeans}"
# ── SWITCH: which job groups to submit ────────────────────────────────────────
#   actions → action-based clustering only (17 jobs)
#   images  → image-based clustering only  (6 jobs)
#   all     → both action + image jobs      (23 jobs)
RUN_MODE="${RUN_MODE:-all}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/home/istepka/11777"
WORKER="${PROJECT_DIR}/run_cluster_job.sh"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results/clustering" "$CKPT_DIR"

echo ">>> Using cluster method: ${CLUSTER_METHOD}"

submit() {
    local name="$1"; shift
    echo "Submitting: $name"
    sbatch --job-name="clust-${name}" "$@" "$WORKER"
}

# GPU submit helper for image jobs
submit_gpu() {
    local name="$1"; shift
    echo "Submitting (GPU): $name"
    sbatch --job-name="clust-${name}" \
        --partition=general \
        --gres=gpu:1 \
        --mem=48G \
        --time=02:00:00 \
        "$@" "$WORKER"
}

if [[ "${RUN_MODE}" == "actions" || "${RUN_MODE}" == "all" ]]; then

# ── native ────────────────────────────────────────────────────────────────────
submit "act-native" \
    --export=ALL,FEATURE_SOURCE=actions,ACTION_REP=native,CLUSTER_METHOD=${CLUSTER_METHOD}

# ── bin ───────────────────────────────────────────────────────────────────────
submit "act-bin" \
    --export=ALL,FEATURE_SOURCE=actions,ACTION_REP=bin,CLUSTER_METHOD=${CLUSTER_METHOD}

# ── quest ─────────────────────────────────────────────────────────────────────
submit "act-quest" \
    --export=ALL,FEATURE_SOURCE=actions,ACTION_REP=quest,CLUSTER_METHOD=${CLUSTER_METHOD}

# ── oat ───────────────────────────────────────────────────────────────────────
submit "act-oat" \
    --export=ALL,FEATURE_SOURCE=actions,ACTION_REP=oat,CLUSTER_METHOD=${CLUSTER_METHOD}

# ── FAST: valid (vocab, scale) pairs ─────────────────────────────────────────
# scale=25 needs vocab>=512; scale=50 needs vocab>=1024.
# Invalid combos (vocab too small for token range) are excluded.
#
#   vocab=256  → scale ∈ {1, 10}
#   vocab=512  → scale ∈ {1, 10, 25}
#   vocab=1024 → scale ∈ {1, 10, 25, 50}
#   vocab=2048 → scale ∈ {1, 10, 25, 50}

for combo in \
    "256 1" "256 10" \
    "512 1" "512 10" "512 25" \
    "1024 1" "1024 10" "1024 25" "1024 50" \
    "2048 1" "2048 10" "2048 25" "2048 50"
do
    read -r vocab scale <<< "$combo"
    tok_path="${CKPT_DIR}/fast_tokenizer_v${vocab}_s${scale}"
    submit "act-fast_v${vocab}_s${scale}" \
        --export=ALL,FEATURE_SOURCE=actions,ACTION_REP=fast,VOCAB_SIZE=${vocab},SCALE=${scale},TOKENIZER_PATH=${tok_path},CLUSTER_METHOD=${CLUSTER_METHOD}
done

echo "Action jobs submitted (4 base + 13 FAST = 17 jobs)."

fi  # end actions

if [[ "${RUN_MODE}" == "images" || "${RUN_MODE}" == "all" ]]; then

DELTA_PATCHES="${DELTA_PATCHES:-16}"

# Encoders that support delta patches
for enc in resnet18 dinov2 dinov2_s dinov2_b vc1; do
    submit_gpu "img-${enc}" \
        --export=ALL,FEATURE_SOURCE=images,IMAGE_ENCODER=${enc},DELTA_PATCHES=${DELTA_PATCHES},CLUSTER_METHOD=${CLUSTER_METHOD}
done

# R3M: global vector — delta not meaningful, use full mode
submit_gpu "img-r3m" \
    --export=ALL,FEATURE_SOURCE=images,IMAGE_ENCODER=r3m,DELTA_PATCHES=0,CLUSTER_METHOD=${CLUSTER_METHOD}

echo "Image jobs submitted (6 encoders)."

fi  # end images

echo ""
echo "Done. RUN_MODE=${RUN_MODE}. Monitor with: squeue -u \$USER"
