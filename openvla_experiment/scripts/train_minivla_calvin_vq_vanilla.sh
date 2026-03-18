#!/bin/bash
#SBATCH --job-name=minivla_vq_vanilla
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH -o /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/logs/minivla_vq_vanilla-%j.out
#SBATCH -e /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/logs/minivla_vq_vanilla-%j.err

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/minivla"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

export WANDB_MODE=offline
export PRISMATIC_DATA_ROOT="${DATA_DIR}"

pip install -e "${OPENVLA_DIR}" --quiet 2>/dev/null || true

mkdir -p "${RUN_DIR}"

cd "${OPENVLA_DIR}"

# train.py reads .hf_token; create if missing (model is public, token not required)
touch -a "${OPENVLA_DIR}/.hf_token"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
    --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-vq-vanilla" \
    --data_root_dir "${DATA_DIR}" \
    --run_root_dir "${RUN_DIR}" \
    --image_aug True \
    --save_interval 5000 \
    --run_id_note calvin_d_vq_vanilla
