#!/bin/bash
#SBATCH --job-name=tok_smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/tok_smoke_%j.out
#SBATCH --error=logs/tok_smoke_%j.err

source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

SAVE_DIR="/tmp/smoke_test_${SLURM_JOB_ID}"
EPOCHS=2
BS=8

echo "===== Smoke test: all tokenizers ====="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Save dir: $SAVE_DIR"
echo ""

# --- VQ-BeT ---
echo ">>> VQ-BeT (vanilla)"
python -u tokenization/train_tokenizer.py \
    --tokenizer vq_bet --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2 \
    --chunk_size 4 --latent_dim 64 --num_codes 512 --vq_groups 2
echo ""

# --- VQ-VAE ---
echo ">>> VQ-VAE (vanilla)"
python -u tokenization/train_tokenizer.py \
    --tokenizer vq_vae --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2
echo ""

# --- OAT ---
echo ">>> OAT (vanilla)"
python -u tokenization/train_tokenizer.py \
    --tokenizer oat --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2
echo ""

# --- QueST ---
echo ">>> QueST (vanilla)"
python -u tokenization/train_tokenizer.py \
    --tokenizer quest --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2
echo ""

# --- FAST ---
echo ">>> FAST (fit)"
python -u tokenization/train_tokenizer.py \
    --tokenizer fast --save_dir $SAVE_DIR \
    --fast_vocab_size 256 --num_workers 2
echo ""

# --- VQ-BeT with verb head ---
echo ">>> VQ-BeT + verb_cls_lambda=0.1"
python -u tokenization/train_tokenizer.py \
    --tokenizer vq_bet --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2 \
    --chunk_size 4 --latent_dim 64 --num_codes 512 --vq_groups 2 \
    --verb_cls_lambda 0.1 --tag verb01
echo ""

# --- VQ-BeT with CLIP head ---
echo ">>> VQ-BeT + clip_lambda=0.5"
python -u tokenization/train_tokenizer.py \
    --tokenizer vq_bet --epochs $EPOCHS --batch_size $BS \
    --save_dir $SAVE_DIR --num_workers 2 \
    --chunk_size 4 --latent_dim 64 --num_codes 512 --vq_groups 2 \
    --clip_lambda 0.5 --tag clip05
echo ""

echo "===== All smoke tests done ====="
ls -la $SAVE_DIR/*/
