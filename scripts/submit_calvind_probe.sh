#!/bin/bash
# Submit Stage 2: Verb decodability probes on all calvind_sweep checkpoints.
# Runs a frozen MLP probe on each tokenizer variant's latent representations.
#
# Usage:
#   bash scripts/submit_calvind_probe.sh [vq_bet|oat|quest|all]

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs results/decodability

FILTER="${1:-all}"
CKPT_BASE="checkpoints/calvind_sweep"

submit_probe() {
    local TOK="$1"
    local SUFFIX="$2"
    local DIR="${CKPT_BASE}/${TOK}_${SUFFIX}"
    local TAG="${TOK}_${SUFFIX}"
    local JOB_NAME="cd_probe_${TOK}_${SUFFIX}"

    if [ ! -d "$DIR" ]; then
        echo "  SKIP (no dir): $DIR"
        return
    fi

    sbatch --job-name="$JOB_NAME" \
        --partition=general --gres=gpu:1 --time="02:00:00" \
        --mem=16G --cpus-per-task=4 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python -u tokenization/probe_decodability.py \
    --ckpt_dir $DIR \
    --tag $TAG \
    --epochs 300 --patience 30
"
    echo "  Submitted: $JOB_NAME"
}

# Conditions to probe
VANILLA_TAG="vanilla"
VERB_TAGS="verb0.01_verb0.01 verb0.1_verb0.1 verb0.5_verb0.5 verb1.0_verb1.0"
CLIP_TAGS="clip0.1_clip0.1 clip0.5_clip0.5 clip1.0_clip1.0 clip2.0_clip2.0"

ALL_TAGS="$VANILLA_TAG $VERB_TAGS $CLIP_TAGS"

if [[ "$FILTER" == "all" ]]; then
    TOKS="vq_bet oat quest"
else
    TOKS="$FILTER"
fi

for TOK in $TOKS; do
    echo "=== $TOK ==="
    for SUFFIX in $ALL_TAGS; do
        submit_probe "$TOK" "$SUFFIX"
    done
done

echo ""
echo "All probes submitted. Monitor with: squeue -u wenjiel2"
