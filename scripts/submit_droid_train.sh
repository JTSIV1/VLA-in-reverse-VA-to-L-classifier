#!/bin/bash
#SBATCH --job-name=droid_ao
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/droid_ao_%j.out
#SBATCH --error=logs/droid_ao_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

TAG="droid_ao_native_d128_v3"
JOBID=${SLURM_JOB_ID}

python train_droid.py \
    --csv_path data/droid_episodes_filtered.csv \
    --actions_dir /data/user_data/wenjiel2/datasets/droid_actions \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --max_seq_len 512 \
    --d_model 128 \
    --num_layers 4 \
    --min_class_count 30 \
    --weighted_loss \
    --weight_decay 0.0 \
    --label_smoothing 0.0 \
    --patience 15 \
    --val_fraction 0.15 \
    --num_workers 4 \
    --save_path checkpoints/${TAG}_j${JOBID}.pth \
    --log_path results/${TAG}_j${JOBID}_log.json
