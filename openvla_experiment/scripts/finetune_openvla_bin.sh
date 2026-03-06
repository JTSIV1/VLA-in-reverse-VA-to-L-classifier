#!/bin/bash
#SBATCH --job-name=openvla_ft_bin
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH -o /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/logs/openvla_ft_bin-%j.out
#SBATCH -e /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/logs/openvla_ft_bin-%j.err

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/openvla"
ADAPTER_DIR="${PROJECT_DIR}/runs/openvla_adapter_tmp"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

export PRISMATIC_DATA_ROOT="${DATA_DIR}"

pip install -e "${OPENVLA_DIR}" --quiet 2>/dev/null || true

mkdir -p "${RUN_DIR}" "${ADAPTER_DIR}"

cd "${OPENVLA_DIR}"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "${DATA_DIR}" \
    --dataset_name calvin_dataset \
    --run_root_dir "${RUN_DIR}" \
    --adapter_tmp_dir "${ADAPTER_DIR}" \
    --lora_rank 32 \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --max_steps 50000 \
    --save_steps 5000 \
    --val_steps 1000 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    --image_aug True \
    --shuffle_buffer_size 50000 \
    --run_id_note calvin_bin
