#!/bin/bash
#SBATCH --job-name=droid_cm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/droid_cm_%j.out
#SBATCH --error=logs/droid_cm_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

python analysis/droid_confusion.py
