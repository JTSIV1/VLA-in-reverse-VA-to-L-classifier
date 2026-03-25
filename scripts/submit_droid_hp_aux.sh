#!/bin/bash
# Retrain DROID HP-sweep winner configs with auxiliary losses.
#
# Defaults mirror the CALVIN follow-up experiments. After the DROID HP sweep
# finishes, update the *_TAG / *_ARGS variables below if different configs win.
#
# Usage:
#   bash scripts/submit_droid_hp_aux.sh
#   bash scripts/submit_droid_hp_aux.sh vq_bet
#   bash scripts/submit_droid_hp_aux.sh oat
#   bash scripts/submit_droid_hp_aux.sh quest

set -euo pipefail

ROOT_DIR="/home/yashagar/multimodal/multimodal-project"
cd "$ROOT_DIR"
mkdir -p logs checkpoints/droid_hp_sweep

MODE="${1:-all}"
SAVE_BASE="checkpoints/droid_hp_sweep"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-64}"
TIME="${TIME:-6:00:00}"
MAX_CHUNKS="${MAX_CHUNKS:-200000}"
VAL_FRAC="${VAL_FRAC:-0.1}"
DROID_ACTIONS_DIR="${DROID_ACTIONS_DIR:-/data/user_data/wenjiel2/datasets/droid_actions}"
METADATA_CACHE="${METADATA_CACHE:-$ROOT_DIR/data/droid_tokenizer_metadata.csv}"

VQBET_TAG="${VQBET_TAG:-c5_e16_g4}"
VQBET_ARGS="${VQBET_ARGS:---chunk_size 5 --latent_dim 512 --num_codes 16 --vq_groups 4}"
OAT_TAG="${OAT_TAG:-h32_f256_r8}"
OAT_ARGS="${OAT_ARGS:---horizon 32 --fsq_levels 4 4 4 4 --num_registers 8}"
QUEST_TAG="${QUEST_TAG:-h16_f256_d2}"
QUEST_ARGS="${QUEST_ARGS:---horizon 16 --fsq_levels 4 4 4 4 --downsample_factor 2}"

job_exists() {
    local job_name="$1"
    squeue -u "$USER" -h -o "%j" | grep -Fxq "$job_name"
}

run_completed() {
    local run_name="$1"
    [[ -f "$SAVE_BASE/$run_name/full.pth" ]]
}

submit_job() {
    local job_name="$1"
    local tokenizer="$2"
    local extra_args="$3"
    local run_name="$4"

    if job_exists "$job_name"; then
        echo "  Skipping active job: $job_name"
        return
    fi
    if run_completed "$run_name"; then
        echo "  Skipping completed run: $run_name"
        return
    fi

    sbatch \
        --job-name="$job_name" \
        --partition=shire-general \
        --gres=gpu:1 \
        --time="$TIME" \
        --mem=48G \
        --cpus-per-task=8 \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --wrap="
source $ROOT_DIR/venv/bin/activate
cd $ROOT_DIR
python -u tokenization/train_tokenizer.py \
    --dataset droid \
    --droid_actions_dir $DROID_ACTIONS_DIR \
    --droid_metadata_cache $METADATA_CACHE \
    --val_fraction $VAL_FRAC \
    --tokenizer $tokenizer \
    --epochs $EPOCHS \
    --batch_size $BS \
    --max_chunks_per_epoch $MAX_CHUNKS \
    --save_dir $SAVE_BASE \
    $extra_args
"

    echo "  Submitted: $job_name"
}

if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "=== VQ-BeT HP winner aux runs ==="
    submit_job "dr_hpaux_vqbet_verb01" "vq_bet" "$VQBET_ARGS --tag $VQBET_TAG --verb_cls_lambda 0.1" "vq_bet_verb0.1_${VQBET_TAG}"
    submit_job "dr_hpaux_vqbet_clip01" "vq_bet" "$VQBET_ARGS --tag $VQBET_TAG --clip_lambda 0.1" "vq_bet_clip0.1_${VQBET_TAG}"
fi

if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "=== OAT HP winner aux runs ==="
    submit_job "dr_hpaux_oat_verb01" "oat" "$OAT_ARGS --tag $OAT_TAG --verb_cls_lambda 0.1" "oat_verb0.1_${OAT_TAG}"
    submit_job "dr_hpaux_oat_clip01" "oat" "$OAT_ARGS --tag $OAT_TAG --clip_lambda 0.1" "oat_clip0.1_${OAT_TAG}"
fi

if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "=== QueST HP winner aux runs ==="
    submit_job "dr_hpaux_quest_verb01" "quest" "$QUEST_ARGS --tag $QUEST_TAG --verb_cls_lambda 0.1" "quest_verb0.1_${QUEST_TAG}"
    submit_job "dr_hpaux_quest_clip01" "quest" "$QUEST_ARGS --tag $QUEST_TAG --clip_lambda 0.1" "quest_clip0.1_${QUEST_TAG}"
    submit_job "dr_hpaux_quest_verb01_prefsq" "quest" "$QUEST_ARGS --tag ${QUEST_TAG}_prefsq --verb_cls_lambda 0.1 --pre_fsq_aux" "quest_verb0.1_${QUEST_TAG}_prefsq"
    submit_job "dr_hpaux_quest_clip01_prefsq" "quest" "$QUEST_ARGS --tag ${QUEST_TAG}_prefsq --clip_lambda 0.1 --pre_fsq_aux" "quest_clip0.1_${QUEST_TAG}_prefsq"
    submit_job "dr_hpaux_quest_verb01_vq" "quest" "--horizon 16 --downsample_factor 2 --vq_type vq --vq_codebook_size 256 --vq_codebook_dim 512 --tag h16_vq_d2 --verb_cls_lambda 0.1" "quest_verb0.1_h16_vq_d2"
    submit_job "dr_hpaux_quest_clip01_vq" "quest" "--horizon 16 --downsample_factor 2 --vq_type vq --vq_codebook_size 256 --vq_codebook_dim 512 --tag h16_vq_d2 --clip_lambda 0.1" "quest_clip0.1_h16_vq_d2"
fi

echo ""
echo "=== Summary ==="
echo "Checkpoints: ${SAVE_BASE}/"
echo "Monitor: squeue -u $(whoami)"