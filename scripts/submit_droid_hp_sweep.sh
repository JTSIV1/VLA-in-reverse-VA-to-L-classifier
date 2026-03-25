#!/bin/bash
# Submit DROID tokenizer hyperparameter sweep jobs.
#
# Mirrors scripts/submit_calvind_hp_sweep.sh but targets the DROID action shards.
#
# Usage:
#   bash scripts/submit_droid_hp_sweep.sh
#   bash scripts/submit_droid_hp_sweep.sh vq_bet
#   bash scripts/submit_droid_hp_sweep.sh oat
#   bash scripts/submit_droid_hp_sweep.sh quest

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
    local tag="$3"
    local extra_args="$4"
    local run_name="${tokenizer}_${tag}"

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
    --tag $tag \
    --epochs $EPOCHS \
    --batch_size $BS \
    --max_chunks_per_epoch $MAX_CHUNKS \
    --save_dir $SAVE_BASE \
    $extra_args
"

    echo "  Submitted: $job_name"
}

if [[ "$MODE" == "all" || "$MODE" == "vq_bet" ]]; then
    echo "=== VQ-BeT (6 configs) ==="
    submit_job "dr_hp_vb_c5_e16_g2"  "vq_bet" "c5_e16_g2"  "--chunk_size 5  --latent_dim 512 --num_codes 16 --vq_groups 2"
    submit_job "dr_hp_vb_c5_e16_g4"  "vq_bet" "c5_e16_g4"  "--chunk_size 5  --latent_dim 512 --num_codes 16 --vq_groups 4"
    submit_job "dr_hp_vb_c5_e64_g2"  "vq_bet" "c5_e64_g2"  "--chunk_size 5  --latent_dim 512 --num_codes 64 --vq_groups 2"
    submit_job "dr_hp_vb_c10_e16_g2" "vq_bet" "c10_e16_g2" "--chunk_size 10 --latent_dim 512 --num_codes 16 --vq_groups 2"
    submit_job "dr_hp_vb_c10_e16_g4" "vq_bet" "c10_e16_g4" "--chunk_size 10 --latent_dim 512 --num_codes 16 --vq_groups 4"
    submit_job "dr_hp_vb_c10_e64_g2" "vq_bet" "c10_e64_g2" "--chunk_size 10 --latent_dim 512 --num_codes 64 --vq_groups 2"
fi

if [[ "$MODE" == "all" || "$MODE" == "oat" ]]; then
    echo "=== OAT (6 configs) ==="
    submit_job "dr_hp_oat_h32_f1000_r8" "oat" "h32_f1000_r8" "--horizon 32 --fsq_levels 8 5 5 5 --num_registers 8"
    submit_job "dr_hp_oat_h32_f256_r8"  "oat" "h32_f256_r8"  "--horizon 32 --fsq_levels 4 4 4 4 --num_registers 8"
    submit_job "dr_hp_oat_h32_f256_r4"  "oat" "h32_f256_r4"  "--horizon 32 --fsq_levels 4 4 4 4 --num_registers 4"
    submit_job "dr_hp_oat_h32_f64_r4"   "oat" "h32_f64_r4"   "--horizon 32 --fsq_levels 4 4 4 --num_registers 4"
    submit_job "dr_hp_oat_h16_f256_r4"  "oat" "h16_f256_r4"  "--horizon 16 --fsq_levels 4 4 4 4 --num_registers 4"
    submit_job "dr_hp_oat_h16_f256_r8"  "oat" "h16_f256_r8"  "--horizon 16 --fsq_levels 4 4 4 4 --num_registers 8"
fi

if [[ "$MODE" == "all" || "$MODE" == "quest" ]]; then
    echo "=== QueST (6 configs) ==="
    submit_job "dr_hp_quest_h32_f1000_d4" "quest" "h32_f1000_d4" "--horizon 32 --fsq_levels 8 5 5 5 --downsample_factor 4"
    submit_job "dr_hp_quest_h32_f256_d4"  "quest" "h32_f256_d4"  "--horizon 32 --fsq_levels 4 4 4 4 --downsample_factor 4"
    submit_job "dr_hp_quest_h32_f256_d8"  "quest" "h32_f256_d8"  "--horizon 32 --fsq_levels 4 4 4 4 --downsample_factor 8"
    submit_job "dr_hp_quest_h32_f64_d4"   "quest" "h32_f64_d4"   "--horizon 32 --fsq_levels 4 4 4 --downsample_factor 4"
    submit_job "dr_hp_quest_h16_f256_d4"  "quest" "h16_f256_d4"  "--horizon 16 --fsq_levels 4 4 4 4 --downsample_factor 4"
    submit_job "dr_hp_quest_h16_f256_d2"  "quest" "h16_f256_d2"  "--horizon 16 --fsq_levels 4 4 4 4 --downsample_factor 2"
fi

echo ""
echo "=== Summary ==="
echo "Checkpoints: ${SAVE_BASE}/"
echo "Monitor: squeue -u $(whoami)"