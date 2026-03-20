#!/bin/bash
# Submit sweep: best classifier per representation type
# All use weighted CE, label_smoothing=0.1, patience=15

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

OAT_CKPT=checkpoints/oat_bridge_j6660959_best.pth
CSV=data/bridge_episodes_filtered.csv
SHARD_DIR=/data/user_data/wenjiel2/datasets/bridge_actions

submit_raw() {
    local D=$1 L=$2
    local TAG="bridge_raw_d${D}_l${L}_wt"
    sbatch --job-name=${TAG} \
        --partition=general --gres=gpu:1 --time=8:00:00 --mem=64G --cpus-per-task=8 \
        --output=logs/${TAG}_%j.out --error=logs/${TAG}_%j.err \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python train_droid.py \
    --csv_path ${CSV} \
    --actions_dir ${SHARD_DIR} \
    --batch_size 64 --epochs 100 --lr 1e-4 \
    --max_seq_len 64 --d_model ${D} --num_layers ${L} \
    --min_class_count 30 --weighted_loss \
    --weight_decay 0.01 --label_smoothing 0.1 --patience 15 \
    --val_fraction 0.15 --num_workers 4 \
    --save_path checkpoints/${TAG}_j\${SLURM_JOB_ID}.pth \
    --log_path results/${TAG}_j\${SLURM_JOB_ID}_log.json
"
}

submit_oat() {
    local REP=$1 D=$2 L=$3
    local TAG="bridge_oat_${REP}_d${D}_l${L}_wt"
    sbatch --job-name=${TAG} \
        --partition=general --gres=gpu:1 --time=8:00:00 --mem=64G --cpus-per-task=8 \
        --output=logs/${TAG}_%j.out --error=logs/${TAG}_%j.err \
        --wrap="
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python train_bridge_oat.py \
    --oat_ckpt ${OAT_CKPT} \
    --csv_path ${CSV} \
    --shard_dir ${SHARD_DIR} \
    --rep_type ${REP} \
    --batch_size 64 --epochs 100 --lr 1e-4 \
    --max_seq_len 32 --d_model ${D} --num_layers ${L} \
    --min_class_count 30 --weighted_loss \
    --weight_decay 0.01 --label_smoothing 0.1 --patience 15 \
    --val_fraction 0.15 --num_workers 4 \
    --save_path checkpoints/${TAG}_j\${SLURM_JOB_ID}.pth \
    --log_path results/${TAG}_j\${SLURM_JOB_ID}_log.json
"
}

echo "=== Raw actions (7d x ~38 tokens) ==="
submit_raw 128 4
submit_raw 256 4
submit_raw 256 6

echo "=== OAT latent (4d x ~13 tokens) ==="
submit_oat latent 64 2
submit_oat latent 128 2
submit_oat latent 128 4

echo "=== OAT discrete (token IDs x ~13 tokens) ==="
submit_oat discrete 64 2
submit_oat discrete 128 2
submit_oat discrete 128 4
