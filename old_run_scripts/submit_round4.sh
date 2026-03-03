#!/bin/bash
# Round 4: Multimodal late fusion — Action native + Vision delta patches
# Tests early vs late fusion with DINOv2-S and VC-1, at d_model=128 and 256
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# ========== d_model=128 (same as unimodal baselines) ==========

# --- DINOv2-S delta16 + native actions ---
JID1=$(sbatch --parsable --job-name=full_dinov2s_d16_early \
    -o logs/full_dinov2s_d16_early-%j.out -e logs/full_dinov2s_d16_early-%j.err \
    run_experiment.sh full_dinov2s_d16_early full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 4)

JID2=$(sbatch --parsable --job-name=full_dinov2s_d16_late1 \
    -o logs/full_dinov2s_d16_late1-%j.out -e logs/full_dinov2s_d16_late1-%j.err \
    run_experiment.sh full_dinov2s_d16_late1 full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 1)

JID3=$(sbatch --parsable --job-name=full_dinov2s_d16_late2 \
    -o logs/full_dinov2s_d16_late2-%j.out -e logs/full_dinov2s_d16_late2-%j.err \
    run_experiment.sh full_dinov2s_d16_late2 full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 2)

# --- VC-1 delta16 + native actions ---
JID4=$(sbatch --parsable --job-name=full_vc1_d16_early \
    -o logs/full_vc1_d16_early-%j.out -e logs/full_vc1_d16_early-%j.err \
    run_experiment.sh full_vc1_d16_early full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 4)

JID5=$(sbatch --parsable --job-name=full_vc1_d16_late1 \
    -o logs/full_vc1_d16_late1-%j.out -e logs/full_vc1_d16_late1-%j.err \
    run_experiment.sh full_vc1_d16_late1 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 1)

JID6=$(sbatch --parsable --job-name=full_vc1_d16_late2 \
    -o logs/full_vc1_d16_late2-%j.out -e logs/full_vc1_d16_late2-%j.err \
    run_experiment.sh full_vc1_d16_late2 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 2)

# ========== d_model=256 (larger model for fair multimodal capacity) ==========

JID7=$(sbatch --parsable --job-name=full_dinov2s_d16_early_d256 \
    -o logs/full_dinov2s_d16_early_d256-%j.out -e logs/full_dinov2s_d16_early_d256-%j.err \
    run_experiment.sh full_dinov2s_d16_early_d256 full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 4 --d_model 256)

JID8=$(sbatch --parsable --job-name=full_dinov2s_d16_late1_d256 \
    -o logs/full_dinov2s_d16_late1_d256-%j.out -e logs/full_dinov2s_d16_late1_d256-%j.err \
    run_experiment.sh full_dinov2s_d16_late1_d256 full native \
    --vision_encoder dinov2_s --delta_patches 16 --cross_layers 1 --d_model 256)

JID9=$(sbatch --parsable --job-name=full_vc1_d16_early_d256 \
    -o logs/full_vc1_d16_early_d256-%j.out -e logs/full_vc1_d16_early_d256-%j.err \
    run_experiment.sh full_vc1_d16_early_d256 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 4 --d_model 256)

JID10=$(sbatch --parsable --job-name=full_vc1_d16_late1_d256 \
    -o logs/full_vc1_d16_late1_d256-%j.out -e logs/full_vc1_d16_late1_d256-%j.err \
    run_experiment.sh full_vc1_d16_late1_d256 full native \
    --vision_encoder vc1 --delta_patches 16 --cross_layers 1 --d_model 256)

echo "Round 4 jobs submitted:"
echo "  --- d_model=128 ---"
echo "  $JID1  full_dinov2s_d16_early (early fusion, cross=4)"
echo "  $JID2  full_dinov2s_d16_late1 (late fusion, cross=1)"
echo "  $JID3  full_dinov2s_d16_late2 (late fusion, cross=2)"
echo "  $JID4  full_vc1_d16_early (early fusion, cross=4)"
echo "  $JID5  full_vc1_d16_late1 (late fusion, cross=1)"
echo "  $JID6  full_vc1_d16_late2 (late fusion, cross=2)"
echo "  --- d_model=256 ---"
echo "  $JID7  full_dinov2s_d16_early_d256 (early, d=256)"
echo "  $JID8  full_dinov2s_d16_late1_d256 (late1, d=256)"
echo "  $JID9  full_vc1_d16_early_d256 (early, d=256)"
echo "  $JID10 full_vc1_d16_late1_d256 (late1, d=256)"
