#!/bin/bash
# Round 6: Multimodal Action Tokenizer Sweep
# Grid: {vc1, dinov2_s} × {native, vqvla, 12 FAST variants} = 28 experiments
#
# Base architecture: R4 best (delta_patches=16, cross_layers=2, d_model=128)
# Standing convention: --min_class_count 30 --weighted_loss
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

RUN_SCRIPT="old_run_scripts/run_experiment.sh"
VQVLA_CKPT="./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth"

# Shared flags: R4 best architecture + standing convention
BASE_FLAGS="--delta_patches 16 --cross_layers 2 --min_class_count 30 --weighted_loss"

# Vision encoders
VISIONS=("vc1" "dinov2_s")
VISION_SHORT=("vc1" "dv2")

# FAST tokenizer configs: (short_name  tokenizer_path)
FAST_CONFIGS=(
    "fs1v256    ./checkpoints/fast_tokenizer_s1_v256"
    "fs1v512    ./checkpoints/fast_tokenizer_s1_v512"
    "fs1v1024   ./checkpoints/fast_tokenizer_s1_v1024"
    "fs5v256    ./checkpoints/fast_tokenizer_s5_v256"
    "fs5v512    ./checkpoints/fast_tokenizer_s5_v512"
    "fs5v1024   ./checkpoints/fast_tokenizer_s5_v1024"
    "fs10v512   ./checkpoints/fast_tokenizer_s10_v512"
    "fs20v512   ./checkpoints/fast_tokenizer_s20_v512"
    "fs20v768   ./checkpoints/fast_tokenizer_s20_v768"
    "fs20v1024  ./checkpoints/fast_tokenizer_s20_v1024"
    "fs50v1024  ./checkpoints/fast_tokenizer_s50_v1024"
    "fs50v1536  ./checkpoints/fast_tokenizer_s50_v1536"
)

echo "=========================================="
echo " Round 6: Multimodal Action Tokenizer Sweep"
echo " Grid: {vc1, dinov2_s} x {native, vqvla, 12 FAST}"
echo " Architecture: delta16, late2, d128, sp+wt"
echo "=========================================="

JOB_COUNT=0

for vi in "${!VISIONS[@]}"; do
    VENC="${VISIONS[$vi]}"
    VSHORT="${VISION_SHORT[$vi]}"

    # --- Native actions ---
    NAME="r6_${VSHORT}_native"
    JID=$(sbatch --parsable --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
        "${RUN_SCRIPT}" "${NAME}" full native \
        --vision_encoder "${VENC}" ${BASE_FLAGS})
    echo "  ${JID}  ${NAME}"
    JOB_COUNT=$((JOB_COUNT + 1))

    # --- VQ-VLA ---
    NAME="r6_${VSHORT}_vqvla"
    JID=$(sbatch --parsable --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
        "${RUN_SCRIPT}" "${NAME}" full vqvla \
        --vision_encoder "${VENC}" --vqvla_checkpoint_path "${VQVLA_CKPT}" ${BASE_FLAGS})
    echo "  ${JID}  ${NAME}"
    JOB_COUNT=$((JOB_COUNT + 1))

    # --- FAST variants ---
    for cfg in "${FAST_CONFIGS[@]}"; do
        FSHORT=$(echo "$cfg" | awk '{print $1}')
        FPATH=$(echo "$cfg" | awk '{print $2}')
        NAME="r6_${VSHORT}_${FSHORT}"
        JID=$(sbatch --parsable --job-name="${NAME}" \
            -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
            "${RUN_SCRIPT}" "${NAME}" full fast \
            --vision_encoder "${VENC}" --fast_tokenizer_path "${FPATH}" ${BASE_FLAGS})
        echo "  ${JID}  ${NAME}"
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted ${JOB_COUNT} jobs. Monitor: squeue -u wenjiel2"
echo "Results: results/r6_*_best_metrics.json"
