#!/bin/bash
# Train MiniVLA from scratch on DROID with shortlisted HP-sweep tokenizers.
#
# This mirrors scripts/submit_calvind_scratch.sh. The default VLA_CONFIG assumes
# openvla-mini has a DROID config registered; override it if your external config
# name differs.

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="${DATA_DIR:-/data/user_data/wenjiel2/datasets/droid_rlds_cache}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/droid_scratch}"
CKPT_BASE="${CKPT_BASE:-${ROOT_DIR}/checkpoints/droid_hp_sweep}"
VLA_CONFIG="${VLA_CONFIG:-prism-qwen25-dinosiglip-224px+0_5b+mx-droid-bin}"
BASE_VLM="${BASE_VLM:-/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498}"
PARTITION="${PARTITION:-general}"
TIME="${TIME:-24:00:00}"
MODE="${1:-all}"

mkdir -p "${ROOT_DIR}/logs" "${RUN_DIR}"

submit() {
    local tag="$1"
    local action_tok="$2"

    sbatch \
        --job-name="dr_sc_${tag}" \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="${TIME}" \
        -o "${ROOT_DIR}/logs/dr_sc_${tag}_%j.out" \
        -e "${ROOT_DIR}/logs/dr_sc_${tag}_%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT=${DATA_DIR}
cd ${OPENVLA_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
    --vla.type ${VLA_CONFIG} \
    --vla.base_vlm ${BASE_VLM} \
    --data_root_dir ${DATA_DIR} \
    --run_root_dir ${RUN_DIR} \
    --image_aug True \
    --run_id_note ${tag} \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 16 \
    --vla.per_device_batch_size 16 \
    --vla.action_tokenizer '${action_tok}'
"
    echo "  Submitted: dr_sc_${tag}"
}

echo "=== DROID MiniVLA from Scratch ==="

if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
    echo "--- Bin baseline ---"
    submit "bin_baseline" "extra_action_tokenizer"
fi

if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit "vb_c5e16g4"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_c5_e16_g4/full.pth"
    submit "vb_c5e64g2"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_c5_e64_g2/full.pth"
    submit "vb_c10e16g4" "sweep:vq_bet:${CKPT_BASE}/vq_bet_c10_e16_g4/full.pth"
fi

if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit "oat_h32f256r8"  "sweep:oat:${CKPT_BASE}/oat_h32_f256_r8/full.pth"
    submit "oat_h32f1000r8" "sweep:oat:${CKPT_BASE}/oat_h32_f1000_r8/full.pth"
    submit "oat_h32f256r4"  "sweep:oat:${CKPT_BASE}/oat_h32_f256_r4/full.pth"
fi

if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit "quest_h16f256d2"  "sweep:quest:${CKPT_BASE}/quest_h16_f256_d2/full.pth"
    submit "quest_h32f1000d4" "sweep:quest:${CKPT_BASE}/quest_h32_f1000_d4/full.pth"
    submit "quest_h16f256d4"  "sweep:quest:${CKPT_BASE}/quest_h16_f256_d4/full.pth"
fi

if [[ "$MODE" == "all" || "$MODE" == "aux" ]]; then
    echo "--- Aux-trained ---"
    submit "vb_c5e16g4_verb01"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_verb0.1_c5_e16_g4/full.pth"
    submit "vb_c5e16g4_clip01"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_clip0.1_c5_e16_g4/full.pth"
    submit "oat_h32f256r8_verb01" "sweep:oat:${CKPT_BASE}/oat_verb0.1_h32_f256_r8/full.pth"
    submit "oat_h32f256r8_clip01" "sweep:oat:${CKPT_BASE}/oat_clip0.1_h32_f256_r8/full.pth"
    submit "quest_h16f256d2_verb01" "sweep:quest:${CKPT_BASE}/quest_verb0.1_h16_f256_d2/full.pth"
    submit "quest_h16f256d2_clip01" "sweep:quest:${CKPT_BASE}/quest_clip0.1_h16_f256_d2/full.pth"
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: ${RUN_DIR}/"