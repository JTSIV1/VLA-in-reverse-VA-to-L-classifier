#!/bin/bash
# Submit SLURM job to build the CALVIN ABCD TFDS dataset.
# Converts raw CALVIN ABCD_D npz files -> RLDS/TFDS episode TFRecords.
# Output: /data/user_data/wenjiel2/datasets/calvin_rlds/calvin_abcd_dataset/1.0.0/
#
# Usage:
#   bash policy/scripts/build_calvin_abcd_tfds.sh

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OUTPUT_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"

sbatch \
    --job-name="calvin_abcd_tfds" \
    --partition=cpu \
    --cpus-per-task=16 \
    --mem=64G \
    --time=24:00:00 \
    -o "${PROJECT_DIR}/logs/calvin_abcd_tfds_build-%j.out" \
    -e "${PROJECT_DIR}/logs/calvin_abcd_tfds_build-%j.err" \
    --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
echo 'Building CALVIN ABCD TFDS dataset...'
python -m datasets.tfds_builders.calvin_abcd_dataset \
    --output_dir ${OUTPUT_DIR}
echo 'Done.'
"

echo "Submitted CALVIN ABCD TFDS build job. Monitor: squeue -u wenjiel2"
echo "Output will be in: ${OUTPUT_DIR}/calvin_abcd_dataset/1.0.0/"
