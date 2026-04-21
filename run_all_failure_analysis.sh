#!/bin/bash
# Run failure analysis for all 12 variants in Table (intrinsic metrics).
# Each run: ~3-5 min on a single GPU (50 episodes).
#
# Usage:
#   bash run_all_failure_analysis.sh              # all variants
#   bash run_all_failure_analysis.sh quest_only   # only QueST variants
#   bash run_all_failure_analysis.sh vqbet_only   # only VQ-BeT variants

set -euo pipefail

POLICY_ROOT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvin_sweep/policy"
TOK_ROOT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvin_sweep/tokenizers"
OUT_ROOT="results/vla_failure_analysis"
SCRIPT="policy/scripts/analyze_vla_failures.py"

MAX_BATCHES=50
TOP_K=10

run_variant() {
    local condition="$1"
    local tok_type="$2"
    local tok_dir="$3"
    local policy_dir="$4"

    echo ""
    echo "================================================================"
    echo "  Running: ${condition}"
    echo "  Tokenizer: ${tok_type} @ ${tok_dir}"
    echo "  Policy: ${policy_dir}"
    echo "================================================================"

    python "${SCRIPT}" \
        --family scratch \
        --condition "${condition}" \
        --checkpoint_dir "${POLICY_ROOT}/${policy_dir}" \
        --sweep_tokenizer_type "${tok_type}" \
        --sweep_checkpoint_path "${TOK_ROOT}/${tok_dir}/full.pth" \
        --out_dir "${OUT_ROOT}/${condition}" \
        --top_k "${TOP_K}" \
        --max_batches "${MAX_BATCHES}" \
        --episodic
}

FILTER="${1:-all}"

# ─────────────────────────────────────────────────────────
# VQ-BeT 5/16/4 variants
# ─────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "vqbet_only" ]]; then
    # vanilla
    run_variant "vqbet_5_16_4_vanilla" \
        "vq_bet" "vq_bet_5_16_4" "minivla_vq_bet_5_16_4"

    # verb (ours)
    run_variant "vqbet_5_16_4_verb" \
        "vq_bet" "vq_bet_5_16_4_verb0.1" "minivla_vq_bet_5_16_4_verb0.1"

    # CLIP (ours)
    run_variant "vqbet_5_16_4_clip" \
        "vq_bet" "vq_bet_5_16_4_clip0.1" "minivla_vq_bet_5_16_4_clip0.1"
fi

# ─────────────────────────────────────────────────────────
# QueST 16/4444/2 variants
# ─────────────────────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "quest_only" ]]; then
    # vanilla
    run_variant "quest_16_4444_2_vanilla" \
        "quest" "quest_16_4444_2" "minivla_quest_16_4444_2"

    # verb (ours)
    run_variant "quest_16_4444_2_verb" \
        "quest" "quest_16_4444_2_verb0.1" "minivla_quest_16_4444_2_verb0.1"

    # CLIP (ours)
    run_variant "quest_16_4444_2_clip" \
        "quest" "quest_16_4444_2_clip0.1" "minivla_quest_16_4444_2_clip0.1"

    # ─────────────────────────────────────────────────────
    # QueST 16/4444/4 variants
    # ─────────────────────────────────────────────────────
    # vanilla
    run_variant "quest_16_4444_4_vanilla" \
        "quest" "quest_16_4444_4" "minivla_quest_16_4444_4"

    # verb (ours)
    run_variant "quest_16_4444_4_verb" \
        "quest" "quest_16_4444_4_verb0.1" "minivla_quest_16_4444_4_verb0.1"

    # CLIP (ours)
    run_variant "quest_16_4444_4_clip" \
        "quest" "quest_16_4444_4_clip0.1" "minivla_quest_16_4444_4_clip0.1"

    # ─────────────────────────────────────────────────────
    # QueST 32/8555/4 variants
    # ─────────────────────────────────────────────────────
    # vanilla
    run_variant "quest_32_8555_4_vanilla" \
        "quest" "quest_32_8555_4" "minivla_quest_32_8555_4"

    # verb (ours)
    run_variant "quest_32_8555_4_verb" \
        "quest" "quest_32_8555_4_verb0.1" "minivla_quest_32_8555_4_verb0.1"

    # CLIP (ours)
    run_variant "quest_32_8555_4_clip" \
        "quest" "quest_32_8555_4_clip0.1" "minivla_quest_32_8555_4_clip0.1"
fi

echo ""
echo "All failure analyses complete!"
echo "Results in: ${OUT_ROOT}/"
