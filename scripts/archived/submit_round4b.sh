#!/bin/bash
# Round 4b: Multimodal fusion + sparse filtering + weighted CE
# Supplements Round 4 by applying the optimized training recipe (sparse+wt CE)
# to the best architectures. Key question: does sparse+weighted CE close the macro F1 gap?
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# --- VC-1 delta16 late1 d256 + sparse + weighted (best R4 architecture) ---
JID1=$(sbatch --parsable --job-name=full_vc1_d16_late1_d256_sp_wt \
    --time=12:00:00 \
    -o logs/full_vc1_d16_late1_d256_sp_wt-%j.out \
    -e logs/full_vc1_d16_late1_d256_sp_wt-%j.err \
    run_experiment.sh full_vc1_d16_late1_d256_sp_wt full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 1 --d_model 256 \
    --min_class_count 30 --weighted_loss)

# --- DINOv2-S delta16 late2 d128 + sparse + weighted (best R4 DINOv2 architecture) ---
JID2=$(sbatch --parsable --job-name=full_dinov2s_d16_late2_sp_wt \
    -o logs/full_dinov2s_d16_late2_sp_wt-%j.out \
    -e logs/full_dinov2s_d16_late2_sp_wt-%j.err \
    run_experiment.sh full_dinov2s_d16_late2_sp_wt full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 2 \
    --min_class_count 30 --weighted_loss)

# --- VC-1 delta16 late2 d128 + sparse + weighted ---
JID3=$(sbatch --parsable --job-name=full_vc1_d16_late2_sp_wt \
    -o logs/full_vc1_d16_late2_sp_wt-%j.out \
    -e logs/full_vc1_d16_late2_sp_wt-%j.err \
    run_experiment.sh full_vc1_d16_late2_sp_wt full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 2 \
    --min_class_count 30 --weighted_loss)

echo "Round 4b jobs submitted:"
echo "  $JID1  full_vc1_d16_late1_d256_sp_wt  (VC-1 late1 d256 + sparse+wt)"
echo "  $JID2  full_dinov2s_d16_late2_sp_wt    (DINOv2 late2 d128 + sparse+wt)"
echo "  $JID3  full_vc1_d16_late2_sp_wt        (VC-1 late2 d128 + sparse+wt)"
