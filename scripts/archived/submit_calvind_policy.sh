#!/bin/bash
# Submit OpenVLA-mini policy fine-tuning with CALVIN-D sweep tokenizers.
#
# 9 conditions (3 tokenizers x {vanilla, verb-winner, clip-winner}):
#   VQ-BeT:  vanilla / verb λ=0.1 / clip λ=0.1
#   OAT:     vanilla / verb λ=0.1 / clip λ=0.1
#   QueST:   vanilla / verb λ=0.01 / clip λ=0.1
#
# Also submits the bin-based baseline (standard ActionTokenizer).
#
# Usage:
#   bash scripts/submit_calvind_policy.sh          # all 10 jobs
#   bash scripts/submit_calvind_policy.sh vq_bet   # just VQ-BeT (3 jobs)
#   bash scripts/submit_calvind_policy.sh baseline  # just bin baseline (1 job)

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/calvind_policy"
ADAPTER_DIR="${PROJECT_DIR}/runs/calvind_policy_adapter_tmp"
CKPT_BASE="${PROJECT_DIR}/checkpoints/calvind_sweep"

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}" "${ADAPTER_DIR}"

MODE="${1:-all}"

submit_sweep() {
    local TOK_TYPE="$1"    # vq_bet, oat, quest
    local CKPT_NAME="$2"   # e.g., vq_bet_vanilla, oat_verb0.1
    local TAG="$3"          # run_id_note

    local CKPT_PATH="${CKPT_BASE}/${CKPT_NAME}/full.pth"
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "  SKIP: checkpoint not found: $CKPT_PATH"
        return
    fi

    sbatch \
        --job-name="pol_${TAG}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${PROJECT_DIR}/logs/pol_${TAG}_%j.out" \
        -e "${PROJECT_DIR}/logs/pol_${TAG}_%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true
cd ${OPENVLA_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ${DATA_DIR} \
    --dataset_name calvin_dataset \
    --run_root_dir ${RUN_DIR} \
    --adapter_tmp_dir ${ADAPTER_DIR} \
    --lora_rank 32 \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --max_steps 50000 \
    --save_steps 5000 \
    --val_steps 1000 \
    --num_val_batches 50 \
    --early_stopping_patience 10 \
    --image_aug True \
    --shuffle_buffer_size 50000 \
    --sweep_tokenizer_type ${TOK_TYPE} \
    --sweep_checkpoint_path ${CKPT_PATH} \
    --run_id_note ${TAG}
"
    echo "  Submitted: pol_${TAG}"
}

submit_baseline() {
    sbatch \
        --job-name="pol_bin_baseline" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${PROJECT_DIR}/logs/pol_bin_baseline_%j.out" \
        -e "${PROJECT_DIR}/logs/pol_bin_baseline_%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true
cd ${OPENVLA_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ${DATA_DIR} \
    --dataset_name calvin_dataset \
    --run_root_dir ${RUN_DIR} \
    --adapter_tmp_dir ${ADAPTER_DIR} \
    --lora_rank 32 \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --max_steps 50000 \
    --save_steps 5000 \
    --val_steps 1000 \
    --num_val_batches 50 \
    --early_stopping_patience 10 \
    --image_aug True \
    --shuffle_buffer_size 50000 \
    --run_id_note bin_baseline
"
    echo "  Submitted: pol_bin_baseline"
}

echo "=== CALVIN-D Policy Training ==="

# Bin-based baseline
if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
    echo "--- Bin baseline ---"
    submit_baseline
fi

# VQ-BeT conditions
if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit_sweep "vq_bet" "vq_bet_vanilla"  "vqbet_vanilla"
    submit_sweep "vq_bet" "vq_bet_verb0.1"  "vqbet_verb01"
    submit_sweep "vq_bet" "vq_bet_clip0.1"  "vqbet_clip01"
fi

# OAT conditions
if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit_sweep "oat" "oat_vanilla"  "oat_vanilla"
    submit_sweep "oat" "oat_verb0.1"  "oat_verb01"
    submit_sweep "oat" "oat_clip0.1"  "oat_clip01"
fi

# QueST conditions
if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit_sweep "quest" "quest_vanilla"  "quest_vanilla"
    submit_sweep "quest" "quest_verb0.01" "quest_verb001"
    submit_sweep "quest" "quest_clip0.1"  "quest_clip01"
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u wenjiel2"
echo "Checkpoints: ${RUN_DIR}/"
