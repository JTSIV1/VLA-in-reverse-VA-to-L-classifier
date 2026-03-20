#!/bin/bash
# Submit tokenizer verb decodability jobs (Table 2)
# 22 classes, weighted loss, action_only modality
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

CONDA_SH="/data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh"
COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64"

submit_job() {
    local NAME="$1"
    shift
    local EXTRA_ARGS="$*"

    local SCRIPT=$(mktemp /tmp/${NAME}.XXXX.sh)
    cat > "$SCRIPT" << JOBEOF
#!/bin/bash
set -euo pipefail
cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate mmml

TAG="${NAME}_j\${SLURM_JOB_ID}"
echo "===== \${TAG} ====="
echo "Node: \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 train_transformer.py \\
    --modality action_only \\
    --save_path "./checkpoints/\${TAG}.pth" \\
    --log_path "./results/\${TAG}_log.json" \\
    $COMMON --weighted_loss $EXTRA_ARGS

echo ">>> Training done. Checkpoints:"
ls -lh checkpoints/\${TAG}*.pth 2>/dev/null

# Eval best checkpoint if it exists
if [[ -f "./checkpoints/\${TAG}_best.pth" ]]; then
    echo ">>> Evaluating best checkpoint..."
    python3 test_transformer.py \\
        --model_path "./checkpoints/\${TAG}_best.pth" \\
        --save_metrics "./results/\${TAG}_best_metrics.json" \\
        --save_preds "./results/\${TAG}_best_preds.json" \\
        --save_cm "./figures/\${TAG}_best_cm.png" \\
        $EXTRA_ARGS 2>&1 || echo ">>> Eval failed (torchvision issue?), but checkpoint is saved."
fi
echo ">>> \${TAG} complete."
JOBEOF
    chmod +x "$SCRIPT"

    local JOB_ID=$(sbatch --parsable \
        --partition=general --gres=gpu:1 --cpus-per-task=8 --mem=32G \
        --time=08:00:00 --account=ybisk \
        --job-name="$NAME" \
        -o "logs/${NAME}-%j.out" \
        -e "logs/${NAME}-%j.err" \
        "$SCRIPT")
    echo "  $NAME: job $JOB_ID"
}

echo "Submitting tokenizer jobs..."

# 1. VQ-VLA pretrained (Open X-Embodiment)
submit_job tok_vqvla_pre --action_rep vqvla

# 2. VQ-VLA finetuned vanilla on CALVIN (λ=0)
submit_job tok_vqvla_ft0 --action_rep vqvla \
    --vqvla_checkpoint_path ./checkpoints/vqvla_ft_vanilla/vqvla_weights.pth

# 3. FAST+ pretrained (DROID, s10/v2048)
submit_job tok_fastp --action_rep fast \
    --fast_tokenizer_path ./checkpoints/fast_pretrained \
    --max_seq_len 192

# 4. FAST s1/v256 fitted on CALVIN (refitted tokenizer)
submit_job tok_fast_s1v256 --action_rep fast \
    --fast_tokenizer_path ./checkpoints/fast_tokenizer_s1_v256_v2

echo ""
echo "Monitor: squeue -u \$USER"
echo "Checkpoints will be in: checkpoints/tok_*_j*.pth"
