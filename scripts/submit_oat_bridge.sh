#!/bin/bash
#SBATCH --job-name=oat_bridge
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/oat_bridge_%j.out
#SBATCH --error=logs/oat_bridge_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

TAG="oat_bridge"
JOBID=${SLURM_JOB_ID}

python train_oat_bridge.py \
    --shard_dir /data/user_data/wenjiel2/datasets/bridge_actions \
    --batch_size 256 \
    --epochs 500 \
    --lr 5e-5 \
    --horizon 32 \
    --num_registers 8 \
    --emb_dim 256 \
    --action_dim 7 \
    --val_fraction 0.1 \
    --num_workers 4 \
    --save_path checkpoints/${TAG}_j${JOBID}.pth \
    --log_path results/${TAG}_j${JOBID}_log.json
