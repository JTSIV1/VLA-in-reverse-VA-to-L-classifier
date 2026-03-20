#!/bin/bash
#SBATCH --job-name=droid_dviz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/droid_dviz_%j.out
#SBATCH --error=logs/droid_dviz_%j.err

source ~/.bashrc
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python analysis/visualize_delta_patches.py
