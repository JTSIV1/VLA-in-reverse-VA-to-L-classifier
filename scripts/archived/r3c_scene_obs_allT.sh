#!/bin/bash
# R3c extension: scene_obs with full sequence (all T timesteps)
# Instead of 2f/8f uniform sampling, load every timestep like native actions
set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
CKPT="${PROJECT}/checkpoints"
LOGS="${PROJECT}/logs"
RESULTS="${PROJECT}/results"

COMMON="--min_class_count 30 --weighted_loss --epochs 30 --lr 5e-4 --max_seq_len 64"

NAME="scene_obs_allT"
SCRIPT="/tmp/${NAME}.sh"
cat > "$SCRIPT" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
# Ensure conda torch takes priority over ~/.local torch 2.8.0
export PYTHONPATH=\$(python -c "import site; print(site.getsitepackages()[0])"):\$PYTHONPATH
cd ${PROJECT}
python3 train_transformer.py ${COMMON} \
    --modality scene_obs \
    --action_rep native \
    --num_frames 0 \
    --save_path ${CKPT}/${NAME}.pth \
    --log_path ${RESULTS}/${NAME}_log.json
SBATCH

sbatch --job-name="$NAME" \
       --partition=general \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=8:00:00 \
       -o "${LOGS}/${NAME}-%j.out" \
       -e "${LOGS}/${NAME}-%j.err" \
       "$SCRIPT"
echo "Submitted $NAME"
