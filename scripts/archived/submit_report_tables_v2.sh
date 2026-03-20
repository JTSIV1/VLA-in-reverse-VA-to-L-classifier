#!/bin/bash
# Submit jobs for report tables: 22 classes, weighted loss
# Table 1: AO native + Scene MLP
# Table 2: AO native vs tokenizer codebooks (VQ-VLA pretrained, VQ-VLA ft vanilla, FAST+, FAST s1/v256)
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

CONDA_SH="/data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh"
COMMON="--epochs 30 --batch_size 16 --lr 5e-4 --max_seq_len 64"
SBATCH_COMMON="--partition=general --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 --account=ybisk"

submit_gpu_job() {
    local NAME="$1"
    local MODALITY="$2"
    local ACTION_REP="$3"
    shift 3
    local EXTRA_FLAGS="$*"

    local SCRIPT=$(mktemp /tmp/report_${NAME}.XXXX.sh)
    cat > "$SCRIPT" << JOBEOF
#!/bin/bash
set -euo pipefail
cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate mmml
mkdir -p logs checkpoints results figures

TAG="${NAME}_j\${SLURM_JOB_ID}"
echo "===== Experiment: ${NAME} ====="
echo "Tag: \${TAG}"
echo "Node: \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 train_transformer.py \\
    --modality "$MODALITY" --action_rep "$ACTION_REP" \\
    --save_path "./checkpoints/\${TAG}.pth" \\
    --log_path "./results/\${TAG}_log.json" \\
    $COMMON --weighted_loss $EXTRA_FLAGS

python3 test_transformer.py \\
    --model_path "./checkpoints/\${TAG}.pth" \\
    --save_cm "./figures/\${TAG}_cm.png" \\
    --save_metrics "./results/\${TAG}_metrics.json" \\
    $EXTRA_FLAGS

if [[ -f "./checkpoints/\${TAG}_best.pth" ]]; then
    echo ">>> Evaluating best-val checkpoint..."
    python3 test_transformer.py \\
        --model_path "./checkpoints/\${TAG}_best.pth" \\
        --save_cm "./figures/\${TAG}_best_cm.png" \\
        --save_metrics "./results/\${TAG}_best_metrics.json" \\
        --save_preds "./results/\${TAG}_best_preds.json" \\
        $EXTRA_FLAGS
fi

echo ">>> Done: \${TAG}"
JOBEOF
    chmod +x "$SCRIPT"
    local JOB_ID=$(sbatch --parsable $SBATCH_COMMON \
        --job-name="$NAME" \
        -o "logs/${NAME}-%j.out" \
        -e "logs/${NAME}-%j.err" \
        "$SCRIPT")
    echo "$NAME: job $JOB_ID"
    echo "$JOB_ID"
}

# --- Table 1 + Table 2 row 1: AO native (22 cls, weighted loss) ---
AO_JOB=$(submit_gpu_job report_ao_22cls action_only native)
AO_JID=$(echo "$AO_JOB" | tail -1)

# --- Table 2 row 2: VQ-VLA pretrained (Open X-Embodiment) ---
submit_gpu_job report_ao_vqvla_pre_22cls action_only vqvla

# --- Table 2 row 3: VQ-VLA finetuned vanilla on CALVIN (λ=0) ---
submit_gpu_job report_ao_vqvla_ft0_22cls action_only vqvla \
    --vqvla_checkpoint_path ./checkpoints/vqvla_ft_vanilla/vqvla_weights.pth

# --- Table 2 row 4: FAST+ pretrained (DROID, s10/v2048) ---
submit_gpu_job report_ao_fastp_22cls action_only fast \
    --fast_tokenizer_path ./checkpoints/fast_pretrained \
    --max_seq_len 192

# --- Table 2 row 5: FAST s1/v256 fitted on CALVIN ---
submit_gpu_job report_ao_fast_s1v256_22cls action_only fast \
    --fast_tokenizer_path ./checkpoints/fast_tokenizer_s1_v256

# --- Table 1: Scene MLP (CPU, depends on AO) ---
SCENE_SCRIPT=$(mktemp /tmp/report_scene.XXXX.sh)
cat > "$SCENE_SCRIPT" << 'SCENEEOF'
#!/bin/bash
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

# Find latest AO best checkpoint
CKPT=$(ls -t checkpoints/report_ao_22cls_j*_best.pth 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: No AO checkpoint found"
    exit 1
fi
echo "Using AO checkpoint: $CKPT"

# Generate preds_ao.json from AO checkpoint
python3 test_transformer.py \
    --model_path "$CKPT" \
    --save_preds results/preds_ao.json \
    --save_metrics results/report_ao_22cls_best_metrics.json \
    --save_cm figures/report_ao_22cls_best_cm.png

# Train scene MLP + complementarity analysis
python3 analysis/sklearn_scene_obs_preds.py

echo "Done."
SCENEEOF
chmod +x "$SCENE_SCRIPT"

SCENE_JOB=$(sbatch --parsable \
    --partition=cpu --cpus-per-task=8 --mem=16G --time=01:00:00 --account=ybisk \
    --dependency=afterok:$AO_JID \
    --job-name=report_scene_mlp \
    -o logs/report_scene_mlp-%j.out \
    -e logs/report_scene_mlp-%j.err \
    "$SCENE_SCRIPT")
echo "Scene MLP: job $SCENE_JOB (depends on AO job $AO_JID)"

echo ""
echo "All jobs submitted. Monitor: squeue -u \$USER"
