#!/bin/bash
# Submit CALVIN rollout evaluation jobs for all 4 fine-tuned OpenVLA conditions.
#
# Uses the calvin_venv conda env (has calvin_env + pybullet installed).
# Runs headless pybullet (no GUI needed on SLURM).
#
# Usage:
#   bash openvla_experiment/scripts/submit_rollout_eval.sh [--num_sequences 1000]
#
# Estimated wall time: ~18h for 1000 sequences with a 7B model on L40S.
# Use --num_sequences 500 to halve the time for a quick check.

set -e

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
RUN_DIR="${PROJECT_DIR}/runs/openvla"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
SUFFIX="openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0"
DATASET_PATH="/data/user_data/yashagar/task_D_D"
NUM_SEQUENCES="${1:-1000}"

mkdir -p "${LOG_DIR}"

submit_rollout() {
    local COND="$1"
    local OPENVLA_CKPT="$2"
    local EXTRA="$3"

    sbatch \
        --job-name="rollout_${COND}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        -o "${LOG_DIR}/rollout_${COND}-%j.out" \
        -e "${LOG_DIR}/rollout_${COND}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PYTHONPATH=${PROJECT_DIR}:/data/user_data/wenjiel2/Code/openvla-mini:/data/user_data/wenjiel2/Code/calvin/calvin_models:/data/user_data/wenjiel2/Code/calvin/calvin_env:\${PYTHONPATH}
export PRISMATIC_DATA_ROOT=/data/user_data/wenjiel2/datasets
export DISPLAY=''
cd ${PROJECT_DIR}
python -u -m openvla_experiment.scripts.evaluate_openvla_rollout \
    --condition ${COND} \
    --checkpoint_dir ${OPENVLA_CKPT} \
    ${EXTRA} \
    --dataset_path ${DATASET_PATH} \
    --output_dir results/rollout \
    --num_sequences ${NUM_SEQUENCES}
"
}

# Bin baseline (50k steps)
JID_BIN=$(submit_rollout \
    "bin" \
    "${RUN_DIR}/${SUFFIX}--calvin_bin--image_aug" \
    "" | awk '{print $NF}')
echo "bin rollout:       job ${JID_BIN}"

# VQ vanilla (45k steps)
JID_VAN=$(submit_rollout \
    "vq_vanilla" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_vanilla--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_vanilla" | awk '{print $NF}')
echo "vq_vanilla rollout: job ${JID_VAN}"

# VQ verb λ=0.5 (45k steps)
JID_VERB=$(submit_rollout \
    "vq_verb" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_verb--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_verb_l0.5" | awk '{print $NF}')
echo "vq_verb rollout:   job ${JID_VERB}"

# VQ verb λ=0.1 (45k steps)
JID_VERB01=$(submit_rollout \
    "vq_verb01" \
    "${RUN_DIR}/${SUFFIX}--calvin_vq_verb01--image_aug" \
    "--vqvla_checkpoint_dir ${CKPT_DIR}/vqvla_ft_verb_l0.1" | awk '{print $NF}')
echo "vq_verb01 rollout: job ${JID_VERB01}"
