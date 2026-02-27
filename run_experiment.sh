#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

# --- Generic single-experiment runner ---
# Usage: sbatch --job-name=<name> -o logs/<name>-%j.out -e logs/<name>-%j.err \
#          run_experiment.sh <name> <modality> <action_rep> [--fast_tokenizer_path PATH] [--debug N]
#
# Examples:
#   sbatch run_experiment.sh action_only action_only native
#   sbatch run_experiment.sh full_fast_v512 full fast --fast_tokenizer_path ./checkpoints/fast_tokenizer_v512

NAME="${1:?Usage: run_experiment.sh <name> <modality> <action_rep> [options]}"
MODALITY="${2:?Missing modality (full|action_only|vision_only)}"
ACTION_REP="${3:?Missing action_rep (native|fast)}"
shift 3

# Parse optional flags
FAST_TOK_FLAG=""
DEBUG_FLAG=""
CROSS_FLAG=""
VISION_FLAG=""
FREEZE_FLAG=""
FRAMES_FLAG=""
WEIGHT_FLAG=""
MIN_CLASS_FLAG=""
DELTA_FLAG=""
D_MODEL_FLAG=""
NUM_LAYERS_FLAG=""
SEQ_LEN_OVERRIDE=""
VQVAE_TOK_FLAG=""
VQVAE_CHUNK_FLAG=""
VQVLA_TOK_FLAG=""
MODAL_DROPOUT_FLAG=""
AUX_LOSS_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast_tokenizer_path)
            FAST_TOK_FLAG="--fast_tokenizer_path $2"
            shift 2
            ;;
        --cross_layers)
            CROSS_FLAG="--cross_layers $2"
            shift 2
            ;;
        --vision_encoder)
            VISION_FLAG="--vision_encoder $2"
            shift 2
            ;;
        --no_freeze_vision)
            FREEZE_FLAG="--no_freeze_vision"
            shift
            ;;
        --max_seq_len)
            SEQ_LEN_OVERRIDE="$2"
            shift 2
            ;;
        --num_frames)
            FRAMES_FLAG="--num_frames $2"
            shift 2
            ;;
        --weighted_loss)
            WEIGHT_FLAG="--weighted_loss"
            shift
            ;;
        --min_class_count)
            MIN_CLASS_FLAG="--min_class_count $2"
            shift 2
            ;;
        --delta_patches)
            DELTA_FLAG="--delta_patches $2"
            shift 2
            ;;
        --d_model)
            D_MODEL_FLAG="--d_model $2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS_FLAG="--num_layers $2"
            shift 2
            ;;
        --vqvae_tokenizer_path)
            VQVAE_TOK_FLAG="--vqvae_tokenizer_path $2"
            shift 2
            ;;
        --vqvae_chunk_size)
            VQVAE_CHUNK_FLAG="--vqvae_chunk_size $2"
            shift 2
            ;;
        --vqvla_checkpoint_path)
            VQVLA_TOK_FLAG="--vqvla_checkpoint_path $2"
            shift 2
            ;;
        --modal_dropout)
            MODAL_DROPOUT_FLAG="--modal_dropout $2"
            shift 2
            ;;
        --aux_loss_weight)
            AUX_LOSS_FLAG="--aux_loss_weight $2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug ${2:-32}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

MAX_SEQ=${SEQ_LEN_OVERRIDE:-64}
COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len $MAX_SEQ"

# Build tag: {NAME}_j{JOB_ID}[_debug]
TAG="${NAME}_j${SLURM_JOB_ID:-local}"
if [[ -n "$DEBUG_FLAG" ]]; then
    TAG="${TAG}_debug"
fi

echo "===== Experiment: ${NAME} ====="
echo "Tag: ${TAG}"
echo "Modality: ${MODALITY} | Action rep: ${ACTION_REP}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- Train ---
python train_transformer.py \
    --modality "$MODALITY" --action_rep "$ACTION_REP" \
    --save_path "./checkpoints/${TAG}.pth" \
    --log_path "./results/${TAG}_log.json" \
    $COMMON $FAST_TOK_FLAG $VQVAE_TOK_FLAG $VQVAE_CHUNK_FLAG $VQVLA_TOK_FLAG $CROSS_FLAG $VISION_FLAG $FREEZE_FLAG $FRAMES_FLAG $WEIGHT_FLAG $MIN_CLASS_FLAG $DELTA_FLAG $D_MODEL_FLAG $NUM_LAYERS_FLAG $MODAL_DROPOUT_FLAG $AUX_LOSS_FLAG $DEBUG_FLAG

# --- Test (final epoch) ---
python test_transformer.py \
    --model_path "./checkpoints/${TAG}.pth" \
    --save_cm "./figures/${TAG}_cm.png" \
    --save_metrics "./results/${TAG}_metrics.json" \
    $FAST_TOK_FLAG $VQVAE_TOK_FLAG $VQVAE_CHUNK_FLAG $VQVLA_TOK_FLAG $DEBUG_FLAG

# --- Test (best val checkpoint, if exists) ---
if [[ -f "./checkpoints/${TAG}_best.pth" ]]; then
    echo ">>> Evaluating best-val checkpoint..."
    python test_transformer.py \
        --model_path "./checkpoints/${TAG}_best.pth" \
        --save_cm "./figures/${TAG}_best_cm.png" \
        --save_metrics "./results/${TAG}_best_metrics.json" \
        $FAST_TOK_FLAG $VQVAE_TOK_FLAG $VQVAE_CHUNK_FLAG $VQVLA_TOK_FLAG $DEBUG_FLAG
fi

echo ">>> ${NAME} done — checkpoint: checkpoints/${TAG}.pth | results: results/${TAG}_metrics.json"
