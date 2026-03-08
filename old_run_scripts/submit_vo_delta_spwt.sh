#!/bin/bash
# Vision-only delta16 sparse+wt — mirrors the multimodal sp+wt recipe:
#   --min_class_count 30 (21 sparse classes) + --weighted_loss
# Needed for a fair apples-to-apples comparison with AO sp+wt and MM sp+wt.
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# --- DINOv2-S delta K=16, 2 frames, 21 sparse classes, weighted CE ---
JID1=$(sbatch --parsable --job-name=vision_dinov2s_delta16_sp_wt \
    -o logs/vision_dinov2s_delta16_sp_wt-%j.out \
    -e logs/vision_dinov2s_delta16_sp_wt-%j.err \
    run_experiment.sh vision_dinov2s_delta16_sp_wt vision_only native \
    --vision_encoder dinov2_s --delta_patches 16 \
    --min_class_count 30 --weighted_loss)

# --- VC-1 delta K=16, 2 frames, 21 sparse classes, weighted CE ---
JID2=$(sbatch --parsable --job-name=vision_vc1_delta16_sp_wt \
    -o logs/vision_vc1_delta16_sp_wt-%j.out \
    -e logs/vision_vc1_delta16_sp_wt-%j.err \
    run_experiment.sh vision_vc1_delta16_sp_wt vision_only native \
    --vision_encoder vc1 --delta_patches 16 \
    --min_class_count 30 --weighted_loss)

echo "Submitted:"
echo "  $JID1  vision_dinov2s_delta16_sp_wt  (delta K=16, 2f, 21 cls, weighted CE)"
echo "  $JID2  vision_vc1_delta16_sp_wt      (delta K=16, 2f, 21 cls, weighted CE)"
