#!/bin/bash
#SBATCH --job-name=bridge_dl
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-31
#SBATCH --output=logs/bridge_dl_%a.out
#SBATCH --error=logs/bridge_dl_%a.err

# Each task: download 32 TFDS shards, then extract actions from them
# 1024 shards / 32 tasks = 32 shards per task

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate openvla

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

SHARDS_PER_TASK=32
SHARD_START=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
SHARD_END=$((SHARD_START + SHARDS_PER_TASK))
TOTAL_SHARDS=1024
TFDS_DIR=/data/user_data/wenjiel2/datasets/bridge_v2
BASE_URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"

echo "Task ${SLURM_ARRAY_TASK_ID}: shards ${SHARD_START} to $((SHARD_END - 1))"

for i in $(seq ${SHARD_START} $((SHARD_END - 1))); do
    SHARD=$(printf "bridge_dataset-train.tfrecord-%05d-of-%05d" $i $TOTAL_SHARDS)
    DEST="${TFDS_DIR}/${SHARD}"

    # Download if needed
    if [ ! -f "${DEST}" ]; then
        echo "[${i}] Downloading ${SHARD}..."
        curl -sL "${BASE_URL}/${SHARD}" -o "${DEST}"
        if [ $? -ne 0 ]; then
            echo "[${i}] Download FAILED"
            rm -f "${DEST}"
            continue
        fi
    fi

    # Extract actions
    python scripts/extract_bridge_actions.py --shard_idx $i
    echo "[${i}] Extracted (keeping tfrecord for future frame access)"
done

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
