#!/bin/bash
# Submit CLIP action-language tokenizer experiments
# Usage: bash scripts/clip_tokenizer_submit.sh
set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
CKPT="${PROJECT}/checkpoints/clip_tokenizer"
LOGS="${PROJECT}/logs"
mkdir -p "${CKPT}" "${LOGS}"

submit_job() {
    local name=$1
    local script=$2
    sbatch --job-name="$name" \
           --partition=general \
           --gres=gpu:1 \
           --cpus-per-task=8 \
           --mem=32G \
           --time=12:00:00 \
           -o "${LOGS}/${name}-%j.out" \
           -e "${LOGS}/${name}-%j.err" \
           "$script"
}

COMMON="--epochs 200 --batch_size 256 --lr 1e-3 --clip_lambda 1.0 --log_every 5"

# Experiment conditions:
#   Text encoder:    CLIP frozen | CLIP+LoRA | GPT-2 frozen | GPT-2+LoRA
#   VQ-VAE:          Tiny VQ-VLA from scratch (3M params)
#
# Phase 1: Text encoder ablation (all use Tiny VQ-VLA)
# Phase 2: Best text encoder + Full VQ-VLA LoRA (TBD after Phase 1)

for spec in \
    "clip_frozen:--text_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K --text_type clip --lora_r 0" \
    "clip_lora:--text_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K --text_type clip --lora_r 8" \
    "gpt2_frozen:--text_model gpt2 --text_type gpt2 --lora_r 0" \
    "gpt2_lora:--text_model gpt2 --text_type gpt2 --lora_r 8" \
; do
    name="${spec%%:*}"
    args="${spec#*:}"
    script="/tmp/clip_${name}.sh"
    cat > "$script" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT}
cd ${PROJECT}/tokenization && PYTHONUNBUFFERED=1 python3 _clip_launcher.py ${COMMON} ${args} --save_dir ${CKPT}/${name}
SBATCH
    submit_job "clip_${name}" "$script"
    echo "Submitted clip_${name}"
done

echo "All CLIP tokenizer jobs submitted."
