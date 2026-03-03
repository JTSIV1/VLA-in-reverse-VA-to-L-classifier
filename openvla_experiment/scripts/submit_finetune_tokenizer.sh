#!/bin/bash
# Submit VQ-VLA tokenizer fine-tuning jobs (Stage 1 of OpenVLA experiment)
#
# Two conditions:
#   1a. Vanilla fine-tuning (lambda=0): domain adaptation only
#   1b. Verb-decodable fine-tuning (lambda>0): recon + vq + verb_CE
#
# Usage:
#   bash openvla_experiment/scripts/submit_finetune_tokenizer.sh

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

submit_ft() {
    local TAG="$1"
    local LAMBDA="$2"
    local EXTRA="${3:-}"

    sbatch \
        --job-name="vqvla_ft_${TAG}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=06:00:00 \
        -o "logs/vqvla_ft_${TAG}-%j.out" \
        -e "logs/vqvla_ft_${TAG}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT_DIR}
python -m openvla_experiment.scripts.finetune_tokenizer \
    --tag ${TAG} \
    --verb_loss_weight ${LAMBDA} \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_windows 16 \
    --min_class_count 30 \
    ${EXTRA}
"
    echo "  Submitted: vqvla_ft_${TAG} (lambda=${LAMBDA})"
}

echo "Stage 1: VQ-VLA tokenizer fine-tuning"
echo "======================================"

# 1a. Vanilla (domain adaptation control)
submit_ft "vanilla" 0.0

# 1b. Verb-decodable (lambda sweep)
for LAMBDA in 0.1 0.5 1.0; do
    submit_ft "verb_l${LAMBDA}" "${LAMBDA}"
done

echo ""
echo "4 jobs submitted. Monitor: squeue -u wenjiel2"
echo "Checkpoints: checkpoints/vqvla_ft_*/"
