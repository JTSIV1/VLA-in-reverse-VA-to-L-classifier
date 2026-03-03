#!/bin/bash
#SBATCH --job-name=cluster-analysis
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cluster-%j.out
#SBATCH --error=logs/cluster-%j.err

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

echo "===== Cluster Analysis ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "=========================="

python -u cluster_analysis.py --max_len 64 --out_dir ./checkpoints

echo "===== Done ====="
