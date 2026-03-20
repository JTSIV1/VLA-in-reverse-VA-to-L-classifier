#!/bin/bash
#SBATCH --job-name=bridge_oat_cls
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bridge_oat_cls_%j.out
#SBATCH --error=logs/bridge_oat_cls_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

OAT_CKPT=checkpoints/oat_bridge_j6660959_best.pth
REP_TYPE=${1:-latent}  # pass "latent" or "discrete" as arg
TAG="bridge_oat_${REP_TYPE}_wt"
JOBID=${SLURM_JOB_ID}

python train_bridge_oat.py \
    --oat_ckpt ${OAT_CKPT} \
    --csv_path data/bridge_episodes_filtered.csv \
    --shard_dir /data/user_data/wenjiel2/datasets/bridge_actions \
    --rep_type ${REP_TYPE} \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --max_seq_len 32 \
    --d_model 128 \
    --num_layers 4 \
    --min_class_count 30 \
    --weighted_loss \
    --weight_decay 0.01 \
    --label_smoothing 0.1 \
    --patience 15 \
    --val_fraction 0.15 \
    --num_workers 4 \
    --save_path checkpoints/${TAG}_j${JOBID}.pth \
    --log_path results/${TAG}_j${JOBID}_log.json
