#!/bin/bash
# Submit OpenVLA fine-tuning with the top-3 DROID HP-sweep configs per tokenizer.
#
# Defaults mirror the CALVIN shortlist. After the DROID HP sweep completes, update
# the checkpoint names below if different configs win.

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="${DATA_DIR:-/data/user_data/wenjiel2/datasets/droid_rlds_cache}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/droid_hp_policy}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT_DIR}/runs/droid_hp_policy_adapter_tmp}"
CKPT_BASE="${CKPT_BASE:-${ROOT_DIR}/checkpoints/droid_hp_sweep}"
PARTITION="${PARTITION:-general}"
TIME="${TIME:-24:00:00}"
MODE="${1:-all}"

mkdir -p "${ROOT_DIR}/logs" "${RUN_DIR}" "${ADAPTER_DIR}"

submit() {
    local tok_type="$1"
    local ckpt_name="$2"
    local tag="$3"
    local ckpt_path="${CKPT_BASE}/${ckpt_name}/full.pth"

    if [[ ! -f "$ckpt_path" ]]; then
        echo "  SKIP: checkpoint not found: $ckpt_path"
        return
    fi

    sbatch \
        --job-name="dr_hp_pol_${tag}" \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="${TIME}" \
        -o "${ROOT_DIR}/logs/dr_hp_pol_${tag}_%j.out" \
        -e "${ROOT_DIR}/logs/dr_hp_pol_${tag}_%j.err" \
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
    echo "  Submitted: dr_hp_pol_${tag}"
}

echo "=== DROID HP Policy Training ==="

if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit "vq_bet" "vq_bet_c5_e16_g4"  "droid_vb_c5e16g4"
    submit "vq_bet" "vq_bet_c5_e64_g2"  "droid_vb_c5e64g2"
    submit "vq_bet" "vq_bet_c10_e16_g4" "droid_vb_c10e16g4"
fi

if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit "oat" "oat_h32_f256_r8"  "droid_oat_h32f256r8"
    submit "oat" "oat_h32_f1000_r8" "droid_oat_h32f1000r8"
    submit "oat" "oat_h32_f256_r4"  "droid_oat_h32f256r4"
fi

if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit "quest" "quest_h16_f256_d2"  "droid_quest_h16f256d2"
    submit "quest" "quest_h32_f1000_d4" "droid_quest_h32f1000d4"
    submit "quest" "quest_h16_f256_d4"  "droid_quest_h16f256d4"
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: ${RUN_DIR}/"