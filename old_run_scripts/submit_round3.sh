#!/bin/bash
# Round 3: DINOv2 + VC-1 patch-based vision encoders
# Experiments: frozen patch tokens (all + delta mode)
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# --- DINOv2-S frozen, all patches (7x7 pooled = 49/frame), 2 frames, 27 classes ---
JID1=$(sbatch --parsable --job-name=vision_dinov2s \
    -o logs/vision_dinov2s-%j.out -e logs/vision_dinov2s-%j.err \
    run_experiment.sh vision_dinov2s vision_only native \
    --vision_encoder dinov2_s)

# --- DINOv2-S frozen, all patches, 2 frames, 21 classes (sparse) ---
JID2=$(sbatch --parsable --job-name=vision_dinov2s_sparse \
    -o logs/vision_dinov2s_sparse-%j.out -e logs/vision_dinov2s_sparse-%j.err \
    run_experiment.sh vision_dinov2s_sparse vision_only native \
    --vision_encoder dinov2_s --min_class_count 30)

# --- VC-1 frozen, all patches (7x7 pooled = 49/frame), 2 frames, 27 classes ---
JID3=$(sbatch --parsable --job-name=vision_vc1 \
    -o logs/vision_vc1-%j.out -e logs/vision_vc1-%j.err \
    run_experiment.sh vision_vc1 vision_only native \
    --vision_encoder vc1)

# --- VC-1 frozen, all patches, 2 frames, 21 classes (sparse) ---
JID4=$(sbatch --parsable --job-name=vision_vc1_sparse \
    -o logs/vision_vc1_sparse-%j.out -e logs/vision_vc1_sparse-%j.err \
    run_experiment.sh vision_vc1_sparse vision_only native \
    --vision_encoder vc1 --min_class_count 30)

# --- DINOv2-S delta patches, 2 frames, K=16, 27 classes ---
JID5=$(sbatch --parsable --job-name=vision_dinov2s_delta16 \
    -o logs/vision_dinov2s_delta16-%j.out -e logs/vision_dinov2s_delta16-%j.err \
    run_experiment.sh vision_dinov2s_delta16 vision_only native \
    --vision_encoder dinov2_s --delta_patches 16)

# --- DINOv2-S delta patches, 8 frames, K=16, 27 classes ---
JID6=$(sbatch --parsable --job-name=vision_dinov2s_delta16_8f \
    -o logs/vision_dinov2s_delta16_8f-%j.out -e logs/vision_dinov2s_delta16_8f-%j.err \
    run_experiment.sh vision_dinov2s_delta16_8f vision_only native \
    --vision_encoder dinov2_s --delta_patches 16 --num_frames 8)

echo "Round 3 jobs submitted:"
echo "  $JID1  vision_dinov2s (frozen, all patches, 2f, 27cls)"
echo "  $JID2  vision_dinov2s_sparse (frozen, all patches, 2f, 21cls)"
echo "  $JID3  vision_vc1 (frozen, all patches, 2f, 27cls)"
echo "  $JID4  vision_vc1_sparse (frozen, all patches, 2f, 21cls)"
echo "  $JID5  vision_dinov2s_delta16 (delta K=16, 2f, 27cls)"
echo "  $JID6  vision_dinov2s_delta16_8f (delta K=16, 8f, 27cls)"
