#!/bin/bash
# Submit all R8 experiments: retrain baselines + fusion stage ablation
# Usage: bash scripts/r8_submit_all.sh
set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
CKPT="${PROJECT}/checkpoints"
LOGS="${PROJECT}/logs"

submit_job() {
    local name=$1
    local script=$2
    sbatch --job-name="$name" \
           --partition=general \
           --gres=gpu:1 \
           --cpus-per-task=8 \
           --mem=32G \
           --time=8:00:00 \
           -o "${LOGS}/${name}-%j.out" \
           -e "${LOGS}/${name}-%j.err" \
           "$script"
}

COMMON="--min_class_count 30 --weighted_loss --epochs 30 --lr 5e-4"

# Write and submit each job script
for spec in \
    "r8_ao_native:--modality action_only --action_rep native" \
    "r8_scene_mlp:--modality scene_mlp --action_rep native" \
    "r8_token_cl1:--modality scene_token --action_rep native --cross_layers 1" \
    "r8_token_cl2:--modality scene_token --action_rep native --cross_layers 2" \
    "r8_token_cl4:--modality scene_token --action_rep native --cross_layers 4" \
; do
    name="${spec%%:*}"
    args="${spec#*:}"
    script="/tmp/${name}.sh"
    cat > "$script" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT}
python3 train_transformer.py ${COMMON} ${args} --save_path ${CKPT}/${name}.pth
SBATCH
    submit_job "$name" "$script"
    echo "Submitted $name"
done

echo "All R8 jobs submitted."
