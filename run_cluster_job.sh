#!/bin/bash
#SBATCH --job-name=cluster-%x
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j.err

# Per-combo worker script.  Called by run_cluster_local.sh via sbatch --export.
# Required env vars (passed via --export):
#   ACTION_REP     : native | fast | bin | quest | oat
#   CLUSTER_METHOD : kmeans | agglomerative  (default: kmeans)
# Optional (fast only):
#   VOCAB_SIZE   : e.g. 256 or 2048
#   SCALE        : e.g. 1 10 25 50
#   TOKENIZER_PATH: path to pre-fitted tokenizer

set -euo pipefail

PROJECT_DIR="/home/istepka/11777"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results/clustering

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "===== Cluster Job: ACTION_REP=${ACTION_REP} ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "======================================="

OUT_DIR="./results/clustering"
CLUSTER_METHOD="${CLUSTER_METHOD:-kmeans}"

if [[ "${ACTION_REP}" == "fast" ]]; then
    echo ">>> Fitting FAST tokenizer (vocab=${VOCAB_SIZE}, scale=${SCALE}) ..."
    python fast_tokenizer.py \
        --save_path "${TOKENIZER_PATH}" \
        --vocab_size "${VOCAB_SIZE}" \
        --scale "${SCALE}"

    echo ">>> Running cluster analysis (fast, vocab=${VOCAB_SIZE}, scale=${SCALE}, method=${CLUSTER_METHOD}) ..."
    python -u cluster_analysis.py \
        --max_len 64 \
        --out_dir "${OUT_DIR}" \
        --action_rep fast \
        --vocab_size "${VOCAB_SIZE}" \
        --scale "${SCALE}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --cluster_method "${CLUSTER_METHOD}"
else
    echo ">>> Running cluster analysis (${ACTION_REP}, method=${CLUSTER_METHOD}) ..."
    python -u cluster_analysis.py \
        --max_len 64 \
        --out_dir "${OUT_DIR}" \
        --action_rep "${ACTION_REP}" \
        --cluster_method "${CLUSTER_METHOD}"
fi

echo "===== Done ====="
