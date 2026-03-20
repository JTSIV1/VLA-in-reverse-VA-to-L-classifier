#!/bin/bash
#SBATCH --job-name=droid_dl
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-31
#SBATCH --output=logs/droid_dl_%a.out
#SBATCH --error=logs/droid_dl_%a.err

# Each array task downloads 64 shards (2048 / 32 = 64)
SHARDS_PER_TASK=64
SHARD_START=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
SHARD_END=$((SHARD_START + SHARDS_PER_TASK))
TOTAL_SHARDS=2048

OUTDIR=/data/user_data/wenjiel2/datasets/droid_rlds

mkdir -p ${OUTDIR}

echo "Task ${SLURM_ARRAY_TASK_ID}: downloading shards ${SHARD_START} to $((SHARD_END - 1))"

for i in $(seq ${SHARD_START} $((SHARD_END - 1))); do
    SHARD=$(printf "droid_101-train.tfrecord-%05d-of-%05d" $i $TOTAL_SHARDS)
    DEST="${OUTDIR}/${SHARD}"

    if [ -f "${DEST}" ]; then
        echo "[${i}] Already exists, skipping."
        continue
    fi

    echo "[${i}] Downloading ${SHARD}..."
    gsutil -q cp "gs://gresearch/robotics/droid/1.0.1/${SHARD}" "${DEST}"

    if [ $? -ne 0 ]; then
        echo "[${i}] FAILED"
        rm -f "${DEST}"
    else
        echo "[${i}] Done"
    fi
done

# Also grab dataset_info.json once (only task 0)
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    gsutil -q cp "gs://gresearch/robotics/droid/1.0.1/dataset_info.json" "${OUTDIR}/"
fi

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
