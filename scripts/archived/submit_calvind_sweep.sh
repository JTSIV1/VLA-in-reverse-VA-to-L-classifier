#!/bin/bash
# Submit CALVIN-D tokenizer sweep jobs.
# Usage:
#   bash scripts/submit_calvind_sweep.sh vanilla     # just vanilla (3 jobs)
#   bash scripts/submit_calvind_sweep.sh verb        # vanilla + verb sweep (15 jobs)
#   bash scripts/submit_calvind_sweep.sh clip        # vanilla + clip sweep (15 jobs)
#   bash scripts/submit_calvind_sweep.sh all         # full sweep (27 jobs)

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs checkpoints/calvind_sweep

MODE="${1:-vanilla}"
SAVE_BASE="checkpoints/calvind_sweep"
EPOCHS=200
BS=64
TIME="08:00:00"

submit_job() {
    local TOK="$1"
    local TAG="$2"          # used for SLURM job name + --tag (vanilla only)
    local EXTRA_ARGS="$3"
    local PY_TAG="${4:-}"    # optional: explicit --tag for python; empty = no --tag
    local JOB_NAME="cd_${TOK}_${TAG}"

    # Tokenizer-specific args
    local TOK_ARGS=""
    if [[ "$TOK" == "vq_bet" ]]; then
        # Paper-aligned: large latent (512) + small codebook (16) + Nq=2 groups
        # VQ-BeT paper Table 13: latent=512, codebook=8-16, Nq=2, chunk=5
        TOK_ARGS="--chunk_size 5 --latent_dim 512 --num_codes 16 --vq_groups 2"
    fi

    local TAG_ARG=""
    if [[ -n "${PY_TAG:-}" ]]; then
        TAG_ARG="--tag $PY_TAG"
    fi

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
    --tokenizer $TOK \
    --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_BASE \
    $TAG_ARG $TOK_ARGS $EXTRA_ARGS
"
    echo "  Submitted: $JOB_NAME"
}

TOKENIZERS="vq_bet oat quest"

# Vanilla (always) — pass "vanilla" as --tag
echo "=== Submitting vanilla jobs ==="
for TOK in $TOKENIZERS; do
    submit_job "$TOK" "vanilla" "" "vanilla"
done

# Verb cls lambda sweep (no --tag; run_name auto-generates from lambda)
if [[ "$MODE" == "verb" || "$MODE" == "all" ]]; then
    echo "=== Submitting verb_cls_lambda sweep ==="
    for TOK in $TOKENIZERS; do
        for LAMBDA in 0.01 0.1 0.5 1.0; do
            submit_job "$TOK" "verb${LAMBDA}" "--verb_cls_lambda $LAMBDA"
        done
    done
fi

# CLIP lambda sweep (no --tag; run_name auto-generates from lambda)
if [[ "$MODE" == "clip" || "$MODE" == "all" ]]; then
    echo "=== Submitting clip_lambda sweep ==="
    for TOK in $TOKENIZERS; do
        for LAMBDA in 0.1 0.5 1.0 2.0; do
            submit_job "$TOK" "clip${LAMBDA}" "--clip_lambda $LAMBDA"
        done
    done
fi

echo "=== Done ==="
