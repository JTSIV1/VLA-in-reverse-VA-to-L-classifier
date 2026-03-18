#!/bin/bash
#SBATCH --job-name=l1_ao
#SBATCH --partition=general
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --account=ybisk
#SBATCH --output=logs/l1_ao_%j.out
#SBATCH --error=logs/l1_ao_%j.err

set -e
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
mkdir -p logs

echo "=== L1 AO Transformer ==="
python3 scripts/train_l1_ao_transformer.py \
    --epochs 30 \
    --batch_size 64 \
    --max_seq_len 64 \
    --tag l1_ao \
    --resume
