#!/bin/bash
# Train MiniVLA from scratch on CALVIN-D with top-3 HP sweep tokenizers.
#
# Uses train.py (FSDP full training) with the base VLM (Qwen2.5-0.5B + DINOv2/SigLIP).
# No OXE pretraining — learns CALVIN actions from scratch.
#
# 9 conditions + bin baseline = 10 jobs total.
#
# Usage:
#   bash scripts/submit_calvind_scratch.sh           # all 10 jobs
#   bash scripts/submit_calvind_scratch.sh vq_bet    # just VQ-BeT (3 jobs)
#   bash scripts/submit_calvind_scratch.sh baseline   # just bin baseline

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/calvind_scratch"
CKPT_BASE="${PROJECT_DIR}/checkpoints/calvind_hp_sweep"

# VLA config ID for CALVIN-D with bin tokenizer (1 GPU, batch=32, 50K steps)
VLA_CONFIG="prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin"

# Local path to pretrained base VLM (Stage 1 checkpoint from Stanford-ILIAD HF)
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"

MODE="${1:-all}"

submit() {
    local TAG="$1"
    local ACTION_TOK="$2"   # e.g., "extra_action_tokenizer" or "sweep:vq_bet:/path"

    sbatch \
        --job-name="sc_${TAG}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${PROJECT_DIR}/logs/sc_${TAG}_%j.out" \
        -e "${PROJECT_DIR}/logs/sc_${TAG}_%j.err" \
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
    --run_id_note ${TAG} \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 16 \
    --vla.per_device_batch_size 16 \
    --vla.action_tokenizer '${ACTION_TOK}'
"
    echo "  Submitted: sc_${TAG}"
}

echo "=== CALVIN-D MiniVLA from Scratch ==="

# Bin baseline
if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
    echo "--- Bin baseline ---"
    submit "bin_baseline" "extra_action_tokenizer"
fi

# VQ-BeT top 3
if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit "vb_c5e16g4"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_c5_e16_g4/full.pth"
    submit "vb_c5e64g2"  "sweep:vq_bet:${CKPT_BASE}/vq_bet_c5_e64_g2/full.pth"
    submit "vb_c10e16g4" "sweep:vq_bet:${CKPT_BASE}/vq_bet_c10_e16_g4/full.pth"
fi

# OAT top 3
if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit "oat_h32f256r8"  "sweep:oat:${CKPT_BASE}/oat_h32_f256_r8/full.pth"
    submit "oat_h32f1000r8" "sweep:oat:${CKPT_BASE}/oat_h32_f1000_r8/full.pth"
    submit "oat_h32f256r4"  "sweep:oat:${CKPT_BASE}/oat_h32_f256_r4/full.pth"
fi

# QueST top 3
if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit "quest_h16f256d2"  "sweep:quest:${CKPT_BASE}/quest_h16_f256_d2/full.pth"
    submit "quest_h32f1000d4" "sweep:quest:${CKPT_BASE}/quest_h32_f1000_d4/full.pth"
    submit "quest_h16f256d4"  "sweep:quest:${CKPT_BASE}/quest_h16_f256_d4/full.pth"
fi

# Aux-trained winners (verb/clip λ=0.1 on best VQ-BeT and QueST configs)
if [[ "$MODE" == "all" || "$MODE" == "aux" ]]; then
    echo "--- Aux-trained ---"
    submit "vb_c5e16g4_verb01"    "sweep:vq_bet:${CKPT_BASE}/vq_bet_verb0.1_c5e16g4_verb01/full.pth"
    submit "vb_c5e16g4_clip01"    "sweep:vq_bet:${CKPT_BASE}/vq_bet_clip0.1_c5e16g4_clip01/full.pth"
    submit "quest_h16d2_verb01"   "sweep:quest:${CKPT_BASE}/quest_verb0.1_h16d2_verb01/full.pth"
    submit "quest_h16d2_clip01"   "sweep:quest:${CKPT_BASE}/quest_clip0.1_h16d2_clip01/full.pth"
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: ${RUN_DIR}/"
