#!/bin/bash
# Submit VQ-VLA + CLIP contrastive fine-tuning experiments
# Usage: bash scripts/clip_vqvla_submit.sh
set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
LOGS="${PROJECT}/logs"
mkdir -p "${LOGS}"

submit_job() {
    local name=$1
    local script=$2
    sbatch --job-name="$name" \
           --partition=general \
           --gres=gpu:1 \
           --cpus-per-task=8 \
           --mem=48G \
           --time=12:00:00 \
           -o "${LOGS}/${name}-%j.out" \
           -e "${LOGS}/${name}-%j.err" \
           "$script"
}

COMMON="--epochs 50 --batch_size 32 --lr 1e-4 --clip_lambda 1.0 --patience 15"

# Conditions:
#   VQ-VLA tokenizer: full finetune (113M) vs LoRA r=8 (~2.4M)
#   Text encoder:     CLIP frozen vs GPT-2 frozen
#   Control:          vanilla (lambda=0, full finetune)

for spec in \
    "clip_full:--text_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K --text_type clip --lora_r 0 --vqvla_lora_r 0" \
    "gpt2_full:--text_model gpt2 --text_type gpt2 --lora_r 0 --vqvla_lora_r 0" \
    "clip_lora:--text_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K --text_type clip --lora_r 0 --vqvla_lora_r 8" \
    "gpt2_lora:--text_model gpt2 --text_type gpt2 --lora_r 0 --vqvla_lora_r 8" \
    "vanilla:--clip_lambda 0.0 --text_model gpt2 --text_type gpt2 --lora_r 0 --vqvla_lora_r 0" \
; do
    name="${spec%%:*}"
    args="${spec#*:}"
    script="/tmp/vqvla_clip_${name}.sh"
    cat > "$script" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd ${PROJECT}
PYTHONUNBUFFERED=1 python3 -m openvla_experiment.scripts.finetune_tokenizer_clip \
    --tag ${name} ${COMMON} ${args}
SBATCH
    submit_job "vqclip_${name}" "$script"
    echo "Submitted vqclip_${name}"
done

echo "All VQ-VLA CLIP jobs submitted."
