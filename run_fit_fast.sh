#!/bin/bash
#SBATCH --job-name=fit-fast
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fit_fast-%j.out
#SBATCH --error=logs/fit_fast-%j.err

set -euo pipefail

# Fit FAST tokenizers on CALVIN training data for multiple vocab sizes.
# Usage: sbatch run_fit_fast.sh [--debug N]

DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_FLAG="--debug ${2:-32}"
fi

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

for VS in 256 512 1024 2048; do
    echo "=== Fitting FAST tokenizer (vocab_size=$VS) ==="
    python fast_tokenizer.py \
        --save_path "./checkpoints/fast_tokenizer_v${VS}" \
        --vocab_size "$VS" \
        $DEBUG_FLAG
done

echo "All FAST tokenizers fitted."
