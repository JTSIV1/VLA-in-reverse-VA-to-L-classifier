#!/bin/bash
#SBATCH --job-name=droid_frames
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-31
#SBATCH --output=logs/droid_frames_%a.out
#SBATCH --error=logs/droid_frames_%a.err

# Each array task processes 64 shards (2048 / 32 = 64)
SHARDS_PER_TASK=64
SHARD_START=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
SHARD_END=$((SHARD_START + SHARDS_PER_TASK))

echo "Task ${SLURM_ARRAY_TASK_ID}: shards ${SHARD_START}-${SHARD_END}"

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

python scripts/extract_droid_frames.py \
    --shard_start ${SHARD_START} \
    --shard_end ${SHARD_END} \
    --total_shards 2048 \
    --output_dir /data/user_data/wenjiel2/datasets/droid_frames
