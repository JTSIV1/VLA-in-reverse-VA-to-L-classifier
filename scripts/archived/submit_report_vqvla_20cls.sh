#!/bin/bash
# Submit VQ-VLA tokenizer jobs with --min_class_count 30 (20 classes)
# Uses mmml_tok env (has compatible torchvision + transformers for VQ-VLA loading)
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

CONDA_SH="/data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh"
SBATCH_COMMON="--partition=general --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 --account=ybisk"

submit_vqvla() {
    local NAME="$1"
    local CKPT_PATH="$2"

    local SCRIPT=$(mktemp /tmp/report_${NAME}.XXXX.sh)
    cat > "$SCRIPT" << JOBEOF
#!/bin/bash
set -euo pipefail
cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate mmml_tok
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"

TAG="${NAME}_j\${SLURM_JOB_ID}"
echo "Node: \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python: \$(which python3)"

python3 train_transformer.py \\
    --modality action_only --action_rep vqvla \\
    --vqvla_checkpoint_path "$CKPT_PATH" \\
    --save_path "./checkpoints/\${TAG}.pth" \\
    --log_path "./results/\${TAG}_log.json" \\
    --epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64 \\
    --weighted_loss --min_class_count 30

python3 test_transformer.py \\
    --model_path "./checkpoints/\${TAG}.pth" \\
    --save_cm "./figures/\${TAG}_cm.png" \\
    --save_metrics "./results/\${TAG}_metrics.json"

if [[ -f "./checkpoints/\${TAG}_best.pth" ]]; then
    python3 test_transformer.py \\
        --model_path "./checkpoints/\${TAG}_best.pth" \\
        --save_cm "./figures/\${TAG}_best_cm.png" \\
        --save_metrics "./results/\${TAG}_best_metrics.json" \\
        --save_preds "./results/\${TAG}_best_preds.json"
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

# VQ-VLA pretrained (Open X-Embodiment)
submit_vqvla tok_vqvla_pre_20cls \
    ./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth

# VQ-VLA finetuned vanilla on CALVIN (λ=0, reconstruction only)
submit_vqvla tok_vqvla_ft0_20cls \
    ./checkpoints/vqvla_ft_vanilla/vqvla_weights_wrapped.pth

echo ""
echo "Monitor: squeue -u \$USER"
