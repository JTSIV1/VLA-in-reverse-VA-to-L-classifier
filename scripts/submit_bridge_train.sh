#!/bin/bash
#SBATCH --job-name=bridge_ao
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bridge_ao_%j.out
#SBATCH --error=logs/bridge_ao_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

TAG="bridge_ao_subtask_d128_s20_nowt"
JOBID=${SLURM_JOB_ID}

python train_bridge.py \
    --csv_path data/bridge_verb_segments.csv \
    --actions_npz /data/user_data/wenjiel2/datasets/bridge_actions/segment_actions.npz \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --max_seq_len 20 \
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
