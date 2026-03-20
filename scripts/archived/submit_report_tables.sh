#!/bin/bash
# Submit jobs for report Table 1 data: AO transformer + Scene MLP
# 22 classes (all verbs), weighted loss, no min_class_count filtering
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"
mkdir -p logs

# --- Job 1: AO native transformer (GPU) ---
AO_JOB=$(sbatch --parsable \
  --job-name=report_ao_22cls \
  -o logs/report_ao_22cls-%j.out \
  -e logs/report_ao_22cls-%j.err \
  "$PROJECT_DIR/old_run_scripts/run_experiment.sh" \
  report_ao_22cls action_only native --weighted_loss)
echo "Submitted AO transformer: job $AO_JOB"

# --- Job 2: AO VQ-VLA transformer (GPU) ---
VQVLA_JOB=$(sbatch --parsable \
  --job-name=report_ao_vqvla_22cls \
  -o logs/report_ao_vqvla_22cls-%j.out \
  -e logs/report_ao_vqvla_22cls-%j.err \
  "$PROJECT_DIR/old_run_scripts/run_experiment.sh" \
  report_ao_vqvla_22cls action_only vqvla --weighted_loss)
echo "Submitted AO VQ-VLA: job $VQVLA_JOB"

# --- Job 3: AO FAST s1/v256 transformer (GPU) ---
FAST_JOB=$(sbatch --parsable \
  --job-name=report_ao_fast_22cls \
  -o logs/report_ao_fast_22cls-%j.out \
  -e logs/report_ao_fast_22cls-%j.err \
  "$PROJECT_DIR/old_run_scripts/run_experiment.sh" \
  report_ao_fast_22cls action_only fast \
  --weighted_loss \
  --fast_tokenizer_path ./checkpoints/fast_tokenizer_s1_v256)
echo "Submitted AO FAST s1/v256: job $FAST_JOB"

# --- Job 4: Scene MLP (CPU, depends on AO finishing for preds) ---
# Scene MLP runs after AO completes to use preds_ao.json for complementarity analysis
cat > /tmp/report_scene_mlp.sh << 'SCENE_EOF'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmml

# First, generate preds_ao.json from the AO best checkpoint
CKPT=$(ls -t checkpoints/report_ao_22cls_j*_best.pth 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: No AO checkpoint found. Was the AO job completed?"
    exit 1
fi
echo "Using AO checkpoint: $CKPT"

python test_transformer.py \
    --model_path "$CKPT" \
    --save_preds results/preds_ao.json \
    --save_metrics results/report_ao_22cls_best_metrics.json \
    --save_cm figures/report_ao_22cls_best_cm.png

# Then run scene MLP complementarity analysis
python analysis/sklearn_scene_obs_preds.py

echo "Done. Results in results/preds_scene.json and results/scene_obs_mlp_metrics.json"
SCENE_EOF

SCENE_JOB=$(sbatch --parsable \
  --dependency=afterok:$AO_JOB \
  --job-name=report_scene_mlp \
  -o logs/report_scene_mlp-%j.out \
  -e logs/report_scene_mlp-%j.err \
  /tmp/report_scene_mlp.sh)
echo "Submitted Scene MLP: job $SCENE_JOB (depends on AO job $AO_JOB)"

echo ""
echo "Summary:"
echo "  AO native:   $AO_JOB"
echo "  AO VQ-VLA:   $VQVLA_JOB"
echo "  AO FAST:     $FAST_JOB"
echo "  Scene MLP:   $SCENE_JOB (after AO)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "After all complete, per-class data in:"
echo "  results/report_ao_22cls_j*_best_metrics.json"
echo "  results/report_ao_vqvla_22cls_j*_best_metrics.json"
echo "  results/report_ao_fast_22cls_j*_best_metrics.json"
echo "  results/scene_obs_mlp_metrics.json"
