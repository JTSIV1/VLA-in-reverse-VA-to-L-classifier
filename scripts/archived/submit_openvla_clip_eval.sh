#!/bin/bash
# Submit OpenVLA-mini fine-tuning with CLIP-trained vs vanilla tokenizers
# This is the downstream evaluation: does CLIP contrastive loss improve
# the tokenizer for VLA action prediction?
#
# 6 conditions:
#   1. clip_full:    CLIP text enc, VQ-VLA full finetune
#   2. clip_lora:    CLIP text enc, VQ-VLA LoRA r=8
#   3. gpt2_full:    GPT-2 text enc, VQ-VLA full finetune
#   4. gpt2_lora:    GPT-2 text enc, VQ-VLA LoRA r=8
#   5. vanilla_full: No contrastive (λ=0), VQ-VLA full finetune
#   6. vanilla_lora: No contrastive (λ=0), VQ-VLA LoRA r=8
set -e

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
RUN_DIR="${PROJECT_DIR}/runs/openvla_clip_eval"
ADAPTER_DIR="${PROJECT_DIR}/runs/openvla_adapter_tmp"
LOGS="${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}" "${ADAPTER_DIR}" "${LOGS}"

for spec in \
    "clip_full:${PROJECT_DIR}/checkpoints/vqvla_clip_clip_full" \
    "clip_lora:${PROJECT_DIR}/checkpoints/vqvla_clip_clip_lora" \
    "gpt2_full:${PROJECT_DIR}/checkpoints/vqvla_clip_gpt2_full" \
    "gpt2_lora:${PROJECT_DIR}/checkpoints/vqvla_clip_gpt2_lora" \
    "vanilla_full:${PROJECT_DIR}/checkpoints/vqvla_clip_vanilla_full" \
    "vanilla_lora:${PROJECT_DIR}/checkpoints/vqvla_clip_vanilla_lora" \
; do
    name="${spec%%:*}"
    ckpt="${spec#*:}"
    script="/tmp/openvla_clip_${name}.sh"
    cat > "$script" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate openvla
export PRISMATIC_DATA_ROOT="${DATA_DIR}"
pip install -e "${OPENVLA_DIR}" --quiet 2>/dev/null || true
cd "${OPENVLA_DIR}"
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\
    --vla_path openvla/openvla-7b \\
    --data_root_dir "${DATA_DIR}" \\
    --dataset_name calvin_abcd_dataset \\
    --run_root_dir "${RUN_DIR}" \\
    --adapter_tmp_dir "${ADAPTER_DIR}/${name}" \\
    --lora_rank 32 \\
    --batch_size 8 \\
    --grad_accumulation_steps 2 \\
    --learning_rate 5e-4 \\
    --max_steps 50000 \\
    --save_steps 5000 \\
    --val_steps 1000 \\
    --warmup_steps 500 \\
    --max_grad_norm 1.0 \\
    --image_aug True \\
    --shuffle_buffer_size 50000 \\
    --run_id_note calvin_clip_${name} \\
    --vqvla_checkpoint_dir "${ckpt}"
SBATCH
    sbatch --job-name="ovla_${name}" \
           --partition=general \
           --gres=gpu:L40S:1 \
           --exclude=babel-m5-32,babel-y9-12 \
           --cpus-per-task=8 \
           --mem=64G \
           --time=30:00:00 \
           -o "${LOGS}/ovla_clip_${name}-%j.out" \
           -e "${LOGS}/ovla_clip_${name}-%j.err" \
           "$script"
    echo "Submitted ovla_${name}"
done

echo "All OpenVLA-mini CLIP eval jobs submitted."
