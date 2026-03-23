#!/bin/bash
# Submit OpenVLA-mini policy fine-tuning with top-3 HP sweep tokenizers per type.
#
# 9 conditions (3 tokenizers × top-3 by val recon loss):
#   VQ-BeT: c5/e16/g4 (0.0085), c5/e64/g2 (0.0089), c10/e16/g4 (0.0119)
#   OAT:    h32/f256/r8 (0.0452), h32/f1000/r8 (0.0461), h32/f256/r4 (0.0484)
#   QueST:  h16/f256/d2 (0.0121), h32/f1000/d4 (0.0138), h16/f256/d4 (0.0158)
#
# Usage:
#   bash scripts/submit_calvind_hp_policy.sh           # all 9 jobs
#   bash scripts/submit_calvind_hp_policy.sh vq_bet    # just VQ-BeT (3 jobs)
#   bash scripts/submit_calvind_hp_policy.sh oat       # just OAT (3 jobs)
#   bash scripts/submit_calvind_hp_policy.sh quest     # just QueST (3 jobs)

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/calvind_hp_policy"
ADAPTER_DIR="${PROJECT_DIR}/runs/calvind_hp_policy_adapter_tmp"
CKPT_BASE="${PROJECT_DIR}/checkpoints/calvind_hp_sweep"

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}" "${ADAPTER_DIR}"

MODE="${1:-all}"

submit() {
    local TOK_TYPE="$1"    # vq_bet, oat, quest
    local CKPT_NAME="$2"   # directory name under calvind_hp_sweep/
    local TAG="$3"          # run_id_note

    local CKPT_PATH="${CKPT_BASE}/${CKPT_NAME}/full.pth"
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "  SKIP: checkpoint not found: $CKPT_PATH"
        return
    fi

    sbatch \
        --job-name="hp_pol_${TAG}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${PROJECT_DIR}/logs/hp_pol_${TAG}_%j.out" \
        -e "${PROJECT_DIR}/logs/hp_pol_${TAG}_%j.err" \
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
    echo "  Submitted: hp_pol_${TAG}"
}

echo "=== CALVIN-D HP Policy Training (top-3 per tokenizer) ==="

# VQ-BeT top 3
if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "--- VQ-BeT ---"
    submit "vq_bet" "vq_bet_c5_e16_g4"   "vb_c5e16g4"    # 4 tok, 0.0085
    submit "vq_bet" "vq_bet_c5_e64_g2"   "vb_c5e64g2"    # 2 tok, 0.0089
    submit "vq_bet" "vq_bet_c10_e16_g4"  "vb_c10e16g4"   # 4 tok, 0.0119
fi

# OAT top 3
if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "--- OAT ---"
    submit "oat" "oat_h32_f256_r8"   "oat_h32f256r8"    # 8 tok, 0.0452
    submit "oat" "oat_h32_f1000_r8"  "oat_h32f1000r8"   # 8 tok, 0.0461
    submit "oat" "oat_h32_f256_r4"   "oat_h32f256r4"    # 4 tok, 0.0484
fi

# QueST top 3
if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "--- QueST ---"
    submit "quest" "quest_h16_f256_d2"   "quest_h16f256d2"    # 8 tok, 0.0121
    submit "quest" "quest_h32_f1000_d4"  "quest_h32f1000d4"   # 8 tok, 0.0138
    submit "quest" "quest_h16_f256_d4"   "quest_h16f256d4"    # 4 tok, 0.0158
fi

echo ""
echo "=== Done ==="
echo "Monitor: squeue -u $(whoami)"
echo "Checkpoints: ${RUN_DIR}/"
