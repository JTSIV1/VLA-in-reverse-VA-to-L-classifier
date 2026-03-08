#!/bin/bash
# Submit attention analysis jobs for all 4 fine-tuned OpenVLA conditions.
#
# Requires attn_implementation="eager" (flash attn won't return weights).
# ~4-6h per condition for 300 val examples on L40S.
#
# Usage:
#   bash openvla_experiment/scripts/submit_attention_analysis.sh [MAX_EXAMPLES]

set -e

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
RUN_DIR="${PROJECT_DIR}/runs/openvla"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
SUFFIX="openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0"
MAX_EXAMPLES="${1:-300}"

mkdir -p "${LOG_DIR}"

submit_attn() {
    local COND="$1"
    local OPENVLA_CKPT="$2"
    local EXTRA="$3"

    sbatch \
        --job-name="attn_${COND}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=8:00:00 \
        -o "${LOG_DIR}/attn_${COND}-%j.out" \
        -e "${LOG_DIR}/attn_${COND}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PYTHONPATH=${PROJECT_DIR}:/data/user_data/wenjiel2/Code/openvla-mini:\${PYTHONPATH}
export PRISMATIC_DATA_ROOT=/data/user_data/wenjiel2/datasets
cd ${PROJECT_DIR}
python -u -m openvla_experiment.scripts.analyze_attention \
    --condition ${COND} \
    --checkpoint_dir ${OPENVLA_CKPT} \
    ${EXTRA} \
    --output_dir results/attention_analysis \
    --max_examples ${MAX_EXAMPLES}
"
}

# Bin baseline
JID_BIN=$(submit_attn \
    "bin" \
    "${RUN_DIR}/${SUFFIX}--calvin_bin--image_aug" \
    "" | awk '{print $NF}')
echo "bin attention:        job ${JID_BIN}"

# VQ vanilla
JID_VAN=$(submit_attn \
    "vq_vanilla" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_vanilla--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_vanilla" | awk '{print $NF}')
echo "vq_vanilla attention: job ${JID_VAN}"

# VQ verb λ=0.5
JID_VERB=$(submit_attn \
    "vq_verb" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_verb--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_verb_l0.5" | awk '{print $NF}')
echo "vq_verb attention:    job ${JID_VERB}"

# VQ verb λ=0.1
JID_VERB01=$(submit_attn \
    "vq_verb01" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_verb01--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_verb_l0.1" | awk '{print $NF}')
echo "vq_verb01 attention:  job ${JID_VERB01}"
