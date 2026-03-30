#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Retrain VQ-BeT policies with fixed per-group token offset.
#
# Background: CalvinSweepActionTokenizer had a bug where all ResidualVQ
# groups shared the same 16 token IDs. The fix (in calvin_sweep_action_tokenizer.py)
# uses per-group offsets: group g code c → token_id = tokenizer_len - 1 - (c + g * n_embed),
# giving 64 distinct tokens (4 groups × 16 codes) instead of 16.
#
# This script:
#   1. Renames old policy dirs to *_old_16tok (preserves them)
#   2. Submits 5 SLURM jobs for policy retraining
#
# Usage:
#   bash scripts/retrain_vqbet_policy.sh           # submit jobs
#   bash scripts/retrain_vqbet_policy.sh --dry-run  # print commands without submitting
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
RLDS_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
POLICY_DIR="$PROJECT_DIR/checkpoints/calvin_sweep/policy"
TOK_DIR="$PROJECT_DIR/checkpoints/calvin_sweep/tokenizers"
LOG_DIR="$PROJECT_DIR/logs"
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"

VLA_CONFIG="prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin"
POLICY_TIME="12:00:00"

# ── Conditions to retrain ─────────────────────────────────────────────
# Format: "display_name tokenizer_type tokenizer_ckpt_relative"
CONDITIONS=(
  "vq_bet_5_16_4         vq_bet  vq_bet_5_16_4/full.pth"
  "vq_bet_5_16_4_verb0.1 vq_bet  vq_bet_5_16_4_verb0.1/full.pth"
  "vq_bet_5_16_4_clip0.1 vq_bet  vq_bet_5_16_4_clip0.1/full.pth"
  "vq_bet_5_64_2         vq_bet  vq_bet_5_64_2/full.pth"
  "vq_bet_10_16_4        vq_bet  vq_bet_10_16_4/full.pth"
)

mkdir -p "$LOG_DIR"

# ── Step 1: Rename old policy dirs ────────────────────────────────────
echo "Step 1: Renaming old policy directories..."
for entry in "${CONDITIONS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  old_dir="$POLICY_DIR/minivla_${tag}"
  if [[ -d "$old_dir" ]]; then
    new_name="${old_dir}_old_16tok"
    if [[ -d "$new_name" ]]; then
      echo "  SKIP $tag (already renamed)"
    else
      if $DRY_RUN; then
        echo "  [DRY RUN] mv $old_dir → $new_name"
      else
        mv "$old_dir" "$new_name"
        echo "  Renamed $(basename $old_dir) → $(basename $new_name)"
      fi
    fi
  else
    echo "  SKIP $tag (no existing dir)"
  fi
done

# ── Step 2: Submit policy training jobs ───────────────────────────────
echo ""
echo "Step 2: Submitting policy training jobs..."
submitted=0

for entry in "${CONDITIONS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  tok=$(echo "$entry" | awk '{print $2}')
  ckpt_rel=$(echo "$entry" | awk '{print $3}')
  tok_ckpt="$TOK_DIR/$ckpt_rel"

  if [[ ! -f "$tok_ckpt" ]]; then
    echo "  ERROR: tokenizer checkpoint not found: $tok_ckpt"
    continue
  fi

  run_id="minivla_${tag}"
  job_name="pol_${tag}"

  cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
  cmd+=" vla-scripts/train.py"
  cmd+=" --vla.type $VLA_CONFIG"
  cmd+=" --vla.base_vlm $BASE_VLM"
  cmd+=" --data_root_dir $RLDS_DIR"
  cmd+=" --run_root_dir $POLICY_DIR"
  cmd+=" --run_id $run_id"
  cmd+=" --image_aug True"
  cmd+=" --save_interval 5000"
  cmd+=" --vla.expected_world_size 1"
  cmd+=" --vla.global_batch_size 16"
  cmd+=" --vla.per_device_batch_size 16"
  cmd+=" --vla.freeze_vision_backbone True"
  cmd+=" --vla.action_tokenizer 'sweep:${tok}:${tok_ckpt}'"

  script=$(cat <<EOF
#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=${POLICY_TIME}
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/%j_${job_name}.out
#SBATCH --error=${LOG_DIR}/%j_${job_name}.err

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

$cmd
EOF
)

  if $DRY_RUN; then
    echo ""
    echo "  [DRY RUN] $tag:"
    echo "$script" | head -20
    echo "  ..."
  else
    jid=$(echo "$script" | sbatch --parsable)
    echo "  $tag → job $jid"
  fi
  submitted=$((submitted + 1))
done

echo ""
echo "Submitted $submitted policy training jobs."
if $DRY_RUN; then
  echo "(DRY RUN — no jobs were actually submitted)"
fi
