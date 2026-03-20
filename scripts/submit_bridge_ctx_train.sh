#!/bin/bash
#SBATCH --job-name=bridge_ctx
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bridge_ctx_%j.out
#SBATCH --error=logs/bridge_ctx_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

TAG="bridge_ctx_subtask_d128"
JOBID=${SLURM_JOB_ID}

python train_bridge_ctx.py \
    --csv_path data/bridge_verb_segments.csv \
    --shard_dir /data/user_data/wenjiel2/datasets/bridge_actions \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --max_ep_len 64 \
    --max_segments 10 \
    --d_model 128 \
    --num_layers 4 \
    --min_class_count 30 \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --patience 15 \
    --val_fraction 0.15 \
    --num_workers 4 \
    --save_path checkpoints/${TAG}_j${JOBID}.pth \
    --log_path results/${TAG}_j${JOBID}_log.json
