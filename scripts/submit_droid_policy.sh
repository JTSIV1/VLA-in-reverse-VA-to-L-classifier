#!/bin/bash
# Submit OpenVLA fine-tuning with the DROID base-sweep winners.
#
# 10 conditions (3 tokenizers x {vanilla, verb winner, clip winner}) plus the
# standard bin-tokenizer baseline.

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="${DATA_DIR:-/data/user_data/wenjiel2/datasets/droid_rlds_cache}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/droid_policy}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT_DIR}/runs/droid_policy_adapter_tmp}"
CKPT_BASE="${CKPT_BASE:-${ROOT_DIR}/checkpoints/droid_sweep}"
PARTITION="${PARTITION:-general}"
TIME="${TIME:-24:00:00}"
MODE="${1:-all}"

mkdir -p "${ROOT_DIR}/logs" "${RUN_DIR}" "${ADAPTER_DIR}"

submit_baseline() {
    sbatch \
        --job-name="dr_pol_bin_baseline" \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="${TIME}" \
        -o "${ROOT_DIR}/logs/dr_pol_bin_baseline_%j.out" \
        -e "${ROOT_DIR}/logs/dr_pol_bin_baseline_%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true
cd ${OPENVLA_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ${DATA_DIR} \
    --dataset_name droid_dataset \
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
    --run_id_note droid_bin_baseline
"
    echo "  Submitted: dr_pol_bin_baseline"
}

submit_sweep() {
    local tok_type="$1"
    local ckpt_name="$2"
    local tag="$3"
    local ckpt_path="${CKPT_BASE}/${ckpt_name}/full.pth"

    if [[ ! -f "$ckpt_path" ]]; then
        echo "  SKIP: checkpoint not found: $ckpt_path"
        return
    fi

    sbatch \
        --job-name="dr_pol_${tag}" \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="${TIME}" \
        -o "${ROOT_DIR}/logs/dr_pol_${tag}_%j.out" \
        -e "${ROOT_DIR}/logs/dr_pol_${tag}_%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true
cd ${OPENVLA_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ${DATA_DIR} \
    --dataset_name droid_dataset \
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
    --sweep_tokenizer_type ${tok_type} \
    --sweep_checkpoint_path ${ckpt_path} \
    --run_id_note ${tag}
"
    echo "  Submitted: dr_pol_${tag}"
}

echo "=== DROID OpenVLA Policy Training ==="

if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
    echo "--- Bin baseline ---"
    submit_baseline
fi

if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit_sweep "vq_bet" "vq_bet_vanilla"  "droid_vqbet_vanilla"
    submit_sweep "vq_bet" "vq_bet_verb0.01" "droid_vqbet_verb001"
    submit_sweep "vq_bet" "vq_bet_clip0.5"  "droid_vqbet_clip05"
fi

if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit_sweep "oat" "oat_vanilla"  "droid_oat_vanilla"
    submit_sweep "oat" "oat_verb0.01" "droid_oat_verb001"
    submit_sweep "oat" "oat_clip0.1"  "droid_oat_clip01"
fi

if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit_sweep "quest" "quest_vanilla"  "droid_quest_vanilla"
    submit_sweep "quest" "quest_verb0.01" "droid_quest_verb001"
    submit_sweep "quest" "quest_clip0.1"  "droid_quest_clip01"
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: ${RUN_DIR}/"