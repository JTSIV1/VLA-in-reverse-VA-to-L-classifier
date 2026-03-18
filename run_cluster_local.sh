#!/bin/bash
# Launcher: submits one SLURM job per tokenizer combo.
# Usage: bash run_cluster_local.sh
# Each job runs run_cluster_job.sh with its own env vars.
# Results go to: results/clustering/<pca|raw>/<cluster_method>/
#
# ── SWITCHES ──────────────────────────────────────────────────────────────────
#   kmeans        → K-Means (k = #verbs)
#   agglomerative → Agglomerative Ward (k = #verbs, no spherical assumption)
CLUSTER_METHOD="${CLUSTER_METHOD:-kmeans}"
#
#   1 → cluster on 99%-PCA-projected features (recommended)
#   0 → cluster on raw StandardScaled features
USE_PCA="${USE_PCA:-1}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/home/istepka/11777"
WORKER="${PROJECT_DIR}/run_cluster_job.sh"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results/clustering" "$CKPT_DIR"

echo ">>> Cluster method: ${CLUSTER_METHOD}  |  Use PCA: ${USE_PCA}"

submit() {
    local name="$1"; shift
    echo "Submitting: $name"
    sbatch --job-name="clust-${name}" "$@" "$WORKER"
}

# ── native ────────────────────────────────────────────────────────────────────
submit "native" \
    --export=ALL,ACTION_REP=native,CLUSTER_METHOD=${CLUSTER_METHOD},USE_PCA=${USE_PCA}

# ── bin ───────────────────────────────────────────────────────────────────────
submit "bin" \
    --export=ALL,ACTION_REP=bin,CLUSTER_METHOD=${CLUSTER_METHOD},USE_PCA=${USE_PCA}

# ── quest ─────────────────────────────────────────────────────────────────────
submit "quest" \
    --export=ALL,ACTION_REP=quest,CLUSTER_METHOD=${CLUSTER_METHOD},USE_PCA=${USE_PCA}

# ── oat ───────────────────────────────────────────────────────────────────────
submit "oat" \
    --export=ALL,ACTION_REP=oat,CLUSTER_METHOD=${CLUSTER_METHOD},USE_PCA=${USE_PCA}

# ── FAST: valid (vocab, scale) pairs ─────────────────────────────────────────
# scale=25 needs vocab>=512; scale=50 needs vocab>=1024.
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
    submit "fast_v${vocab}_s${scale}" \
        --export=ALL,ACTION_REP=fast,VOCAB_SIZE=${vocab},SCALE=${scale},TOKENIZER_PATH=${tok_path},CLUSTER_METHOD=${CLUSTER_METHOD},USE_PCA=${USE_PCA}
done

echo "All jobs submitted (4 base + 13 FAST = 17 jobs). Monitor with: squeue -u \$USER"
