#!/bin/bash
# Round 7: Scene Obs + Action Fusion
# Grid: {scene_token, scene_concat, scene_film} x {native, vqvla, fast} = 9 experiments
#
# Scene representation: delta_start (48-d): [scene_obs[start], scene_obs[end]-scene_obs[start]]
# Standing convention: --min_class_count 30 --weighted_loss
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

RUN_SCRIPT="old_run_scripts/run_experiment.sh"
VQVLA_CKPT="./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth"
FAST_TOK="./checkpoints/fast_tokenizer_s1_v256"

BASE_FLAGS="--min_class_count 30 --weighted_loss"

# Fusion strategies (modality names)
FUSIONS=("scene_token" "scene_concat" "scene_film")
FUSION_SHORT=("token" "concat" "film")

echo "=========================================="
echo " Round 7: Scene Obs + Action Fusion"
echo " Grid: {token, concat, film} x {native, vqvla, fast}"
echo " Scene: delta_start (48-d), sp+wt"
echo "=========================================="

JOB_COUNT=0

for fi in "${!FUSIONS[@]}"; do
    FMOD="${FUSIONS[$fi]}"
    FSHORT="${FUSION_SHORT[$fi]}"

    # --- Native actions ---
    NAME="r7_${FSHORT}_native"
    JID=$(sbatch --parsable --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
        "${RUN_SCRIPT}" "${NAME}" "${FMOD}" native \
        ${BASE_FLAGS})
    echo "  ${JID}  ${NAME}"
    JOB_COUNT=$((JOB_COUNT + 1))

    # --- VQ-VLA ---
    NAME="r7_${FSHORT}_vqvla"
    JID=$(sbatch --parsable --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
        "${RUN_SCRIPT}" "${NAME}" "${FMOD}" vqvla \
        --vqvla_checkpoint_path "${VQVLA_CKPT}" ${BASE_FLAGS})
    echo "  ${JID}  ${NAME}"
    JOB_COUNT=$((JOB_COUNT + 1))

    # --- FAST (best: s1/v256) ---
    NAME="r7_${FSHORT}_fast"
    JID=$(sbatch --parsable --job-name="${NAME}" \
        -o "logs/${NAME}-%j.out" -e "logs/${NAME}-%j.err" \
        "${RUN_SCRIPT}" "${NAME}" "${FMOD}" fast \
        --fast_tokenizer_path "${FAST_TOK}" ${BASE_FLAGS})
    echo "  ${JID}  ${NAME}"
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "Submitted ${JOB_COUNT} jobs. Monitor: squeue -u wenjiel2"
echo "Results: results/r7_*_best_metrics.json"
