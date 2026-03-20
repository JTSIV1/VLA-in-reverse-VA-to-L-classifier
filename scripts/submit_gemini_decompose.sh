#!/bin/bash
#SBATCH --job-name=gemini_d
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9
#SBATCH --output=logs/gemini_decompose_%a.out
#SBATCH --error=logs/gemini_decompose_%a.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

export SLURM_ARRAY_TASK_COUNT=10

python -u scripts/gemini_decompose_all.py
