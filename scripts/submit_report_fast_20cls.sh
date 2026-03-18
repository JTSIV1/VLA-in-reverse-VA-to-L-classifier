#!/bin/bash
# Resubmit FAST tokenizer jobs with --min_class_count 30 (20 classes)
# Excludes collapse (<30 train) and unstack (<30 train)
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

CONDA_SH="/data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh"
COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64 --weighted_loss --min_class_count 30"
SBATCH_COMMON="--partition=general --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 --account=ybisk"

submit_fast() {
    local NAME="$1"
    local TOK_PATH="$2"
    local SEQ_LEN="$3"

    local SCRIPT=$(mktemp /tmp/report_${NAME}.XXXX.sh)
    cat > "$SCRIPT" << JOBEOF
#!/bin/bash
set -euo pipefail
cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate mmml
export PYTHONNOUSERSITE=1

TAG="${NAME}_j\${SLURM_JOB_ID}"
echo "Node: \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 train_transformer.py \
    --modality action_only --action_rep fast \
    --fast_tokenizer_path "$TOK_PATH" \
    --save_path "./checkpoints/\${TAG}.pth" \
    --log_path "./results/\${TAG}_log.json" \
    --epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len $SEQ_LEN \
    --weighted_loss --min_class_count 30

python3 test_transformer.py \
    --model_path "./checkpoints/\${TAG}.pth" \
    --save_cm "./figures/\${TAG}_cm.png" \
    --save_metrics "./results/\${TAG}_metrics.json" \
    --min_class_count 30

if [[ -f "./checkpoints/\${TAG}_best.pth" ]]; then
    python3 test_transformer.py \
        --model_path "./checkpoints/\${TAG}_best.pth" \
        --save_cm "./figures/\${TAG}_best_cm.png" \
        --save_metrics "./results/\${TAG}_best_metrics.json" \
        --save_preds "./results/\${TAG}_best_preds.json" \
        --min_class_count 30
fi

echo ">>> ${NAME} complete."
JOBEOF
    chmod +x "$SCRIPT"
    local JOB_ID=$(sbatch --parsable $SBATCH_COMMON \
        --job-name="$NAME" \
        -o "logs/${NAME}-%j.out" \
        -e "logs/${NAME}-%j.err" \
        "$SCRIPT")
    echo "$NAME: job $JOB_ID"
}

# FAST+ pretrained (DROID, s10/v2048) — needs longer seq_len
submit_fast tok_fastp_20cls ./checkpoints/fast_pretrained 192

# FAST s1/v256 (fitted on CALVIN, refitted for current tokenizers lib)
submit_fast tok_fast_s1v256_20cls ./checkpoints/fast_tokenizer_s1_v256_v2 64

echo ""
echo "Monitor: squeue -u \$USER"
