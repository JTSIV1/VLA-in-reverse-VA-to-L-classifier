#!/bin/bash
# ============================================================================
# LIBERO-Goal OAT 16_855_4 LoRA fine-tunes (d_fixed=896 fullproj path).
#
# Two policies, matching the Phase 3 scope in
# lab_notebooks/libero_para/plan.md:
#   - vanilla:   tokenizer = oat_16_855_4
#   - LATTiCE:   tokenizer = oat_16_855_4_vlm_clip0.1
#
# Tokenizers were trained on data/libero_goal_episodes.csv (500 demos);
# RLDS data was downloaded from openvla/modified_libero_rlds:libero_goal_no_noops.
# ============================================================================
set -euo pipefail

PROJECT_DIR="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
RLDS_DIR="/data/user_data/wenjiel2/datasets/libero_rlds"
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"
POLICY_DIR="$PROJECT_DIR/checkpoints/libero_sweep/policy"
TOK_DIR="$PROJECT_DIR/checkpoints/libero_sweep/tokenizers"
LOG_DIR="$PROJECT_DIR/logs"
EXCLUDE_NODES="babel-l5-[16,20],babel-m9-[16,20],babel-n9-20,babel-z5-28"

D_FIXED=896

# (tag, tokenizer_dir_name)
JOBS=(
    "libero_goal_oat_16_855_4_fullproj              oat_16_855_4"
    "libero_goal_oat_16_855_4_vlm_clip0.1_fullproj  oat_16_855_4_vlm_clip0.1"
)

mkdir -p "$LOG_DIR" "$POLICY_DIR"

submitted=0
for entry in "${JOBS[@]}"; do
    read -r tag tok_dir <<< "$entry"
    name="pol_${tag}"
    tok_ckpt="$TOK_DIR/$tok_dir"

    if [[ ! -f "$tok_ckpt/full.pth" ]]; then
        echo "SKIP $tag — tokenizer ckpt not found: $tok_ckpt/full.pth"
        continue
    fi

    run_dir="$POLICY_DIR/minivla_${tag}"
    if [[ -d "$run_dir/checkpoints" ]] && ls "$run_dir/checkpoints/"*.pt &>/dev/null; then
        echo "SKIP $tag — checkpoint already exists in $run_dir"
        continue
    fi

    script=$(mktemp /tmp/${name}_XXXX.sh)
    cat > "$script" <<SBATCH
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --exclude=$EXCLUDE_NODES
#SBATCH -o $LOG_DIR/%j_${name}.out
#SBATCH -e $LOG_DIR/%j_${name}.err

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PYTHONNOUSERSITE=1
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
    vla-scripts/train.py \\
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops \\
    --vla.base_vlm $BASE_VLM \\
    --data_root_dir $RLDS_DIR \\
    --run_root_dir $POLICY_DIR \\
    --run_id minivla_${tag} \\
    --image_aug True \\
    --save_interval 5000 \\
    --vla.expected_world_size 1 \\
    --vla.global_batch_size 16 \\
    --vla.per_device_batch_size 16 \\
    --vla.freeze_vision_backbone True \\
    --vla.max_steps 50000 \\
    --vla.d_fixed $D_FIXED \\
    --vla.action_tokenizer 'sweep:oat:${tok_ckpt}/full.pth'
SBATCH

    sbatch "$script"
    echo "  Submitted: $name (d_fixed=$D_FIXED, dynamic OAT encoder embedding)"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. Submitted $submitted / ${#JOBS[@]} jobs."
