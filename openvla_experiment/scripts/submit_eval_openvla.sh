#!/bin/bash
# Submit Stage 3 evaluation jobs for fine-tuned OpenVLA models.
#
# Runs two evaluations per condition:
#   A. Teacher-forcing NLL + action accuracy (GPU, needs checkpoint)
#   B. Verb decodability probe via tokenizer round-trip (CPU-friendly)
#
# Run after Stage 2 fine-tuning is complete.
#
# Usage:
#   bash openvla_experiment/scripts/submit_eval_openvla.sh
#
# To evaluate a specific condition only:
#   bash openvla_experiment/scripts/submit_eval_openvla.sh --condition bin

set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/openvla"
RESULTS_DIR="${PROJECT_DIR}/results/stage3"
CKPT_DIR="${PROJECT_DIR}/checkpoints"

mkdir -p "${PROJECT_DIR}/logs" "${RESULTS_DIR}"

# ── Checkpoint paths (set after Stage 2 training completes) ───────────────────
# These follow the naming convention from finetune.py:
#   openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--{TAG}--image_aug
BIN_CKPT="${RUN_DIR}/openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--calvin_bin--image_aug"
VQ_VAN_CKPT="${RUN_DIR}/openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--calvin_vq_vanilla--image_aug"
VQ_VERB_CKPT="${RUN_DIR}/openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--calvin_vq_verb--image_aug"
VQ_VERB01_CKPT="${RUN_DIR}/openvla-7b+calvin_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--calvin_vq_verb01--image_aug"

# ── VQ-VLA tokenizer checkpoints ─────────────────────────────────────────────
VQVLA_VANILLA="${CKPT_DIR}/vqvla_ft_vanilla"
VQVLA_VERB="${CKPT_DIR}/vqvla_ft_verb_l0.5"
VQVLA_VERB01="${CKPT_DIR}/vqvla_ft_verb_l0.1"

# ── Verb classifier checkpoint ────────────────────────────────────────────────
VERB_CLF="${CKPT_DIR}/ao_native_sparse_weighted_j6457852_best.pth"

# ── Submission helper ──────────────────────────────────────────────────────────
submit_nll() {
    local CONDITION="$1"
    local CHECKPOINT="$2"
    local EXTRA="${3:-}"

    sbatch \
        --job-name="eval_nll_${CONDITION}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=48G \
        --time=4:00:00 \
        -o "${PROJECT_DIR}/logs/eval_nll_${CONDITION}-%j.out" \
        -e "${PROJECT_DIR}/logs/eval_nll_${CONDITION}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true

cd ${PROJECT_DIR}

python -m openvla_experiment.scripts.evaluate_openvla \
    --condition ${CONDITION} \
    --eval_nll \
    --checkpoint_dir ${CHECKPOINT} \
    --data_root_dir ${DATA_DIR} \
    --max_nll_batches 500 \
    --output_dir ${RESULTS_DIR}/${CONDITION} \
    ${EXTRA}
"
    echo "  Submitted: eval_nll_${CONDITION}"
}

submit_verb_probe() {
    local CONDITION="$1"
    local EXTRA="${2:-}"

    sbatch \
        --job-name="eval_verb_${CONDITION}" \
        --partition=general \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=2:00:00 \
        -o "${PROJECT_DIR}/logs/eval_verb_${CONDITION}-%j.out" \
        -e "${PROJECT_DIR}/logs/eval_verb_${CONDITION}-%j.err" \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

export PRISMATIC_DATA_ROOT=${DATA_DIR}
pip install -e ${OPENVLA_DIR} --quiet 2>/dev/null || true

cd ${PROJECT_DIR}

python -m openvla_experiment.scripts.evaluate_openvla \
    --condition ${CONDITION} \
    --eval_verb_probe \
    --verb_classifier_ckpt ${VERB_CLF} \
    --min_class_count 30 \
    --output_dir ${RESULTS_DIR}/${CONDITION} \
    ${EXTRA}
"
    echo "  Submitted: eval_verb_${CONDITION}"
}

echo "Stage 3: OpenVLA evaluation"
echo "============================"
echo ""
echo "Checking checkpoint directories..."

for dir in "${BIN_CKPT}" "${VQ_VAN_CKPT}" "${VQ_VERB_CKPT}" "${VQ_VERB01_CKPT}"; do
    if [ -d "${dir}" ]; then
        echo "  [FOUND ] ${dir}"
    else
        echo "  [MISSING] ${dir} — NLL eval will be skipped for this condition"
    fi
done

echo ""
echo "Submitting verb probe jobs (no checkpoint required)..."
submit_verb_probe "bin"
submit_verb_probe "vq_vanilla" "--vqvla_checkpoint_dir ${VQVLA_VANILLA}"
submit_verb_probe "vq_verb"    "--vqvla_checkpoint_dir ${VQVLA_VERB}"
submit_verb_probe "vq_verb01"  "--vqvla_checkpoint_dir ${VQVLA_VERB01}"

echo ""
echo "Submitting NLL eval jobs (requires checkpoints)..."
if [ -d "${BIN_CKPT}" ]; then
    submit_nll "bin" "${BIN_CKPT}"
fi
if [ -d "${VQ_VAN_CKPT}" ]; then
    submit_nll "vq_vanilla" "${VQ_VAN_CKPT}" "--vqvla_checkpoint_dir ${VQVLA_VANILLA}"
fi
if [ -d "${VQ_VERB_CKPT}" ]; then
    submit_nll "vq_verb" "${VQ_VERB_CKPT}" "--vqvla_checkpoint_dir ${VQVLA_VERB}"
fi
if [ -d "${VQ_VERB01_CKPT}" ]; then
    submit_nll "vq_verb01" "${VQ_VERB01_CKPT}" "--vqvla_checkpoint_dir ${VQVLA_VERB01}"
fi

echo ""
echo "Results will be written to: ${RESULTS_DIR}/"
echo "Monitor: squeue -u wenjiel2"
