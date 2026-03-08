#!/bin/bash
set -euo pipefail

# Submit all baseline experiments in parallel.
#
# Experiments:
#   - action_only (native)          — immediate
#   - vision_only                   — immediate
#   - full_fast_vN (4 vocab sizes)  — immediate (tokenizers pre-fitted)
#   - action_only_fast_vN (4 vocab) — immediate (tokenizers pre-fitted)
#
# Usage: bash submit_baselines.sh [--debug N]

DEBUG_FLAG=""
DEBUG_ARGS=""
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_FLAG="--debug ${2:-32}"
    DEBUG_ARGS="--debug ${2:-32}"
fi

mkdir -p logs checkpoints results figures

SBATCH_EXP="sbatch --parsable"
VOCAB_SIZES="256 512 1024 2048"

echo "=== Submitting baseline experiments ==="

# 1. Action-only (native) — no dependency
AO_JOB=$($SBATCH_EXP \
    --job-name=action_only \
    --output=logs/action_only-%j.out \
    --error=logs/action_only-%j.err \
    run_experiment.sh action_only action_only native $DEBUG_ARGS)
echo "[1] Action-only (native):  job $AO_JOB"

# 2. Vision-only — no dependency
VO_JOB=$($SBATCH_EXP \
    --job-name=vision_only \
    --output=logs/vision_only-%j.out \
    --error=logs/vision_only-%j.err \
    run_experiment.sh vision_only vision_only native $DEBUG_ARGS)
echo "[2] Vision-only:           job $VO_JOB"

# 3. FAST vocab size sweep — tokenizers already fitted, no dependency
COUNT=3
for VS in $VOCAB_SIZES; do
    # Full + FAST at this vocab size
    FF_JOB=$($SBATCH_EXP \
        --job-name="full_fast_v${VS}" \
        --output="logs/full_fast_v${VS}-%j.out" \
        --error="logs/full_fast_v${VS}-%j.err" \
        run_experiment.sh "full_fast_v${VS}" full fast \
        --fast_tokenizer_path "./checkpoints/fast_tokenizer_v${VS}" $DEBUG_ARGS)
    echo "[$COUNT] Full+FAST v${VS}:        job $FF_JOB"
    COUNT=$((COUNT + 1))

    # Action-only + FAST at this vocab size
    AF_JOB=$($SBATCH_EXP \
        --job-name="ao_fast_v${VS}" \
        --output="logs/action_only_fast_v${VS}-%j.out" \
        --error="logs/action_only_fast_v${VS}-%j.err" \
        run_experiment.sh "action_only_fast_v${VS}" action_only fast \
        --fast_tokenizer_path "./checkpoints/fast_tokenizer_v${VS}" $DEBUG_ARGS)
    echo "[$COUNT] Action+FAST v${VS}:      job $AF_JOB"
    COUNT=$((COUNT + 1))
done

echo ""
echo "=== ${COUNT} jobs submitted ==="
echo "Monitor: squeue -u \$USER"
