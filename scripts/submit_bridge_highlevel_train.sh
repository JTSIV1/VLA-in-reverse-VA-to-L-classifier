#!/bin/bash
#SBATCH --job-name=bridge_hl
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bridge_hl_%j.out
#SBATCH --error=logs/bridge_hl_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

TAG="bridge_hl_ao_d256"
JOBID=${SLURM_JOB_ID}

python train_droid.py \
    --csv_path data/bridge_episodes_filtered.csv \
    --actions_dir /data/user_data/wenjiel2/datasets/bridge_actions \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --max_seq_len 64 \
    --d_model 256 \
    --num_layers 6 \
    --min_class_count 30 \
    --weighted_loss \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --patience 15 \
    --val_fraction 0.15 \
    --num_workers 4 \
    --save_path checkpoints/${TAG}_j${JOBID}.pth \
    --log_path results/${TAG}_j${JOBID}_log.json
