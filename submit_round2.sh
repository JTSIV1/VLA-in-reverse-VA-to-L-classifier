#!/bin/bash
# Submit Round 2 experiments: R3M vision-only + pretrained FAST+ action-only
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs

# --- Vision-only with R3M (frozen) ---
sbatch --job-name=vision_r3m_frozen \
    -o logs/vision_r3m_frozen-%j.out -e logs/vision_r3m_frozen-%j.err \
    run_experiment.sh vision_r3m_frozen vision_only native \
    --vision_encoder r3m

# --- Vision-only with R3M (fine-tuned) ---
sbatch --job-name=vision_r3m_finetune \
    -o logs/vision_r3m_finetune-%j.out -e logs/vision_r3m_finetune-%j.err \
    run_experiment.sh vision_r3m_finetune vision_only native \
    --vision_encoder r3m --no_freeze_vision

# --- Action-only with pretrained FAST+ (max_seq_len=192 for longer token seqs) ---
sbatch --job-name=ao_fast_pretrained \
    -o logs/ao_fast_pretrained-%j.out -e logs/ao_fast_pretrained-%j.err \
    run_experiment.sh ao_fast_pretrained action_only fast \
    --fast_tokenizer_path ./checkpoints/fast_pretrained --max_seq_len 192

echo "Submitted 3 round 2 experiments"
