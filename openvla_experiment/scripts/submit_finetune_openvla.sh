#!/bin/bash
# Submit OpenVLA-mini fine-tuning jobs on CALVIN task_D_D.
#
# Conditions:
#   A. vanilla:   standard ActionTokenizer (256 bins, no verb-decodable tokenizer)
#   B. (future)   VQ-based verb-decodable tokenizer
#
# Prerequisite: CALVIN TFDS dataset must be built first:
#   bash openvla_experiment/scripts/build_calvin_tfds.sh
#
# Usage:
#   bash openvla_experiment/scripts/submit_finetune_openvla.sh

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/openvla"
ADAPTER_DIR="${PROJECT_DIR}/runs/openvla_adapter_tmp"

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}" "${ADAPTER_DIR}"

submit_ft() {
    local TAG="$1"
    local EXTRA="${2:-}"

    sbatch \
        --job-name="openvla_ft_${TAG}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${PROJECT_DIR}/logs/openvla_ft_${TAG}-%j.out" \
        -e "${PROJECT_DIR}/logs/openvla_ft_${TAG}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

# Install openvla-mini if not already installed
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
    --image_aug True \
    --shuffle_buffer_size 50000 \
    --run_id_note ${TAG} \
    ${EXTRA}
"
    echo "  Submitted: openvla_ft_${TAG}"
}

echo "Stage 2: OpenVLA-mini fine-tuning on CALVIN task_D_D"
echo "====================================================="

# Condition A: standard bin-based ActionTokenizer (baseline)
submit_ft "calvin_vanilla"

echo ""
echo "1 job submitted. Monitor: squeue -u wenjiel2"
echo "Checkpoints: ${RUN_DIR}/"
echo ""
echo "NOTE: This requires the CALVIN TFDS dataset to be built first."
echo "      If not done: bash openvla_experiment/scripts/build_calvin_tfds.sh"
