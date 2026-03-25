#!/bin/bash
# Submit DROID tokenizer sweep jobs.
# Usage:
#   bash scripts/submit_droid_sweep.sh vanilla
#   bash scripts/submit_droid_sweep.sh verb
#   bash scripts/submit_droid_sweep.sh clip
#   bash scripts/submit_droid_sweep.sh all

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
cd "$ROOT_DIR"
mkdir -p logs checkpoints/droid_sweep

MODE="${1:-vanilla}"
SAVE_BASE="checkpoints/droid_sweep"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-64}"
TIME="${TIME:-4:00:00}"
MAX_CHUNKS="${MAX_CHUNKS:-200000}"
VAL_FRAC="${VAL_FRAC:-0.1}"
DROID_ACTIONS_DIR="${DROID_ACTIONS_DIR:-/data/user_data/wenjiel2/datasets/droid_actions}"
METADATA_CACHE="${METADATA_CACHE:-$ROOT_DIR/data/droid_tokenizer_metadata.csv}"

job_exists() {
    local JOB_NAME="$1"
    squeue -u "$USER" -h -o "%j" | grep -Fxq "$JOB_NAME"
}

run_completed() {
    local RUN_NAME="$1"
    [[ -f "$SAVE_BASE/$RUN_NAME/full.pth" ]]
}

submit_job() {
    local TOK="$1"
    local TAG="$2"
    local EXTRA_ARGS="$3"
    local PY_TAG="${4:-}"
    local JOB_NAME="dr_${TOK}_${TAG}"
    local RUN_NAME="$TOK"

    if [[ "$TAG" == "vanilla" ]]; then
        RUN_NAME+="_vanilla"
    elif [[ "$TAG" == verb* ]]; then
        RUN_NAME+="_${TAG}"
    elif [[ "$TAG" == clip* ]]; then
        RUN_NAME+="_${TAG}"
    fi

    if job_exists "$JOB_NAME"; then
        echo "  Skipping active job: $JOB_NAME"
        return
    fi
    if run_completed "$RUN_NAME"; then
        echo "  Skipping completed run: $RUN_NAME"
        return
    fi

    local TOK_ARGS=""
    if [[ "$TOK" == "vq_bet" ]]; then
        TOK_ARGS="--chunk_size 5 --latent_dim 512 --num_codes 16 --vq_groups 2"
    fi

    local TAG_ARG=""
    if [[ -n "${PY_TAG:-}" ]]; then
        TAG_ARG="--tag $PY_TAG"
    fi

    sbatch \
        --job-name="$JOB_NAME" \
        --partition=shire-general \
        --gres=gpu:1 \
        --time="$TIME" \
        --mem=48G \
        --cpus-per-task=8 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source $ROOT_DIR/venv/bin/activate
cd $ROOT_DIR
python -u tokenization/train_tokenizer.py \
    --dataset droid \
    --droid_actions_dir $DROID_ACTIONS_DIR \
    --droid_metadata_cache $METADATA_CACHE \
    --val_fraction $VAL_FRAC \
    --tokenizer $TOK \
    --epochs $EPOCHS \
    --batch_size $BS \
    --max_chunks_per_epoch $MAX_CHUNKS \
    --save_dir $SAVE_BASE \
    $TAG_ARG $TOK_ARGS $EXTRA_ARGS
"

    echo "  Submitted: $JOB_NAME"
}

TOKENIZERS="vq_bet oat quest"

echo "=== Submitting DROID vanilla jobs ==="
for TOK in $TOKENIZERS; do
    submit_job "$TOK" "vanilla" "" "vanilla"
done

if [[ "$MODE" == "verb" || "$MODE" == "all" ]]; then
    echo "=== Submitting DROID verb sweep ==="
    for TOK in $TOKENIZERS; do
        for LAMBDA in 0.01 0.1 0.5 1.0; do
            submit_job "$TOK" "verb${LAMBDA}" "--verb_cls_lambda $LAMBDA"
        done
    done
fi

if [[ "$MODE" == "clip" || "$MODE" == "all" ]]; then
    echo "=== Submitting DROID CLIP sweep ==="
    for TOK in $TOKENIZERS; do
        for LAMBDA in 0.1 0.5 1.0 2.0; do
            submit_job "$TOK" "clip${LAMBDA}" "--clip_lambda $LAMBDA"
        done
    done
fi

echo "=== Done ==="