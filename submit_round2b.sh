#!/bin/bash
# Round 2b: Additional experiments
# 1. R3M finetune + 8 frames (frozen 8f collapsed, finetune 2f reached 22.4% — try finetune 8f)
# 2. R3M finetune + sparse class filtering (drop classes with <30 train samples: 27→21 classes)
# 3. Action-only native + sparse class filtering (re-baseline with fewer classes)
# 4. R3M finetune + 8 frames + sparse class filtering (best combo)

set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# --- R3M finetune 8 frames ---
sbatch --job-name=vision_r3m_ft_8f \
    -o logs/vision_r3m_ft_8f-%j.out -e logs/vision_r3m_ft_8f-%j.err \
    run_experiment.sh vision_r3m_ft_8f vision_only native \
    --vision_encoder r3m --no_freeze_vision --num_frames 8

# --- R3M finetune 2f + sparse filtering (min 30 train samples) ---
sbatch --job-name=vision_r3m_ft_sparse \
    -o logs/vision_r3m_ft_sparse-%j.out -e logs/vision_r3m_ft_sparse-%j.err \
    run_experiment.sh vision_r3m_ft_sparse vision_only native \
    --vision_encoder r3m --no_freeze_vision --min_class_count 30

# --- Action-only native + sparse filtering (re-baseline) ---
sbatch --job-name=ao_native_sparse \
    -o logs/ao_native_sparse-%j.out -e logs/ao_native_sparse-%j.err \
    run_experiment.sh ao_native_sparse action_only native \
    --min_class_count 30

# --- R3M finetune 8f + sparse filtering (best combo) ---
sbatch --job-name=vision_r3m_ft_8f_sparse \
    -o logs/vision_r3m_ft_8f_sparse-%j.out -e logs/vision_r3m_ft_8f_sparse-%j.err \
    run_experiment.sh vision_r3m_ft_8f_sparse vision_only native \
    --vision_encoder r3m --no_freeze_vision --num_frames 8 --min_class_count 30

echo "Submitted 4 round 2b jobs"
