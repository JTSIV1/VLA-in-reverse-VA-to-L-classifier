#!/bin/bash
# Submit SLURM job to build the CALVIN TFDS dataset.
# This converts raw CALVIN npz files -> proper RLDS/TFDS episode TFRecords.
# Output: /data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/
#
# Must be run once before fine-tuning. Overwrites previous (incorrectly
# formatted) TFRecords from calvin_to_rlds.py.
#
# Usage:
#   bash openvla_experiment/scripts/build_calvin_tfds.sh

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OUTPUT_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"

sbatch \
    --job-name="calvin_tfds_build" \
    --partition=cpu \
    --cpus-per-task=8 \
    --mem=32G \
    --time=04:00:00 \
    -o "${PROJECT_DIR}/logs/calvin_tfds_build-%j.out" \
    -e "${PROJECT_DIR}/logs/calvin_tfds_build-%j.err" \
    --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
echo 'Building CALVIN TFDS dataset...'
python -m openvla_experiment.tfds_builders.calvin_dataset \
    --output_dir ${OUTPUT_DIR}
echo 'Done.'
"

echo "Submitted CALVIN TFDS build job. Monitor: squeue -u wenjiel2"
echo "Output will be in: ${OUTPUT_DIR}/calvin_dataset/1.0.0/"
