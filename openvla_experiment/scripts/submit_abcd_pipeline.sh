#!/bin/bash
# Submit the full ABCD pipeline:
#   1. Build CALVIN ABCD TFDS dataset (cpu job, ~4-8h)
#   2. Fine-tune OpenVLA-mini on ABCD for all 4 conditions (after build completes)
#   3. (Run rollout eval manually after fine-tuning finishes)
#
# Usage:
#   bash openvla_experiment/scripts/submit_abcd_pipeline.sh
#
# Rollout eval: after fine-tuning done, patch config.json with norm_stats then run:
#   bash openvla_experiment/scripts/submit_rollout_eval_abcd.sh

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"

# Step 1: Build TFDS dataset
BUILD_JOB=$(sbatch \
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
python -m openvla_experiment.tfds_builders.calvin_abcd_dataset \
    --output_dir /data/user_data/wenjiel2/datasets/calvin_rlds
echo 'ABCD TFDS build done.'
" | awk '{print $NF}')
echo "TFDS build job: ${BUILD_JOB}"

# Step 2: Fine-tune all 4 conditions (depends on build completing)
for script in \
    finetune_openvla_abcd_bin.sh \
    finetune_openvla_abcd_vq_vanilla.sh \
    finetune_openvla_abcd_vq_verb.sh \
    finetune_openvla_abcd_vq_verb_l0.1.sh; do
    JID=$(sbatch \
        --dependency=afterok:${BUILD_JOB} \
        "${PROJECT_DIR}/openvla_experiment/scripts/${script}" | awk '{print $NF}')
    echo "Fine-tune ${script}: job ${JID}"
done

echo ""
echo "Pipeline submitted. Steps:"
echo "  1. TFDS build (job ${BUILD_JOB}) — ~4-8h"
echo "  2. 4× fine-tune jobs (start after build) — ~30h each"
echo "  3. After fine-tuning: patch config.json with norm_stats, then submit rollout eval"
