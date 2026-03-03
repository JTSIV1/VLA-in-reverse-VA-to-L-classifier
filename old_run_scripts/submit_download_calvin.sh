#!/bin/bash
#SBATCH --job-name=dl_calvin
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_calvin_%j.log

set -euo pipefail

DEST="/data/user_data/wenjiel2/datasets"
URL="http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip"
ZIPFILE="${DEST}/task_ABCD_D.zip"
MAX_RETRIES=20

cd "$DEST"

echo "=== Starting download at $(date) ==="
for i in $(seq 1 $MAX_RETRIES); do
    echo "--- Attempt $i / $MAX_RETRIES at $(date) ---"
    if wget --continue --progress=dot:giga --timeout=120 --tries=3 -O "$ZIPFILE" "$URL"; then
        echo "=== Download finished at $(date) ==="
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "=== Download FAILED after $MAX_RETRIES attempts ==="
        exit 1
    fi
    echo "--- wget exited, retrying in 30s... ---"
    sleep 30
done

echo "=== Starting unzip at $(date) ==="
unzip -o "$ZIPFILE" -d "$DEST"
echo "=== Unzip finished at $(date) ==="

echo "=== Cleaning up zip file ==="
rm "$ZIPFILE"

echo "=== Done at $(date) ==="
ls -lh "$DEST"
