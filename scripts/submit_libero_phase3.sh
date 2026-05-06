#!/bin/bash
# ============================================================================
# Submit both Phase 3 LIBERO-Goal policy fine-tunes (vanilla + LATTiCE) —
# self-configuring, edit-nothing version.
#
# Reads paths from env vars with sensible defaults that match
# lab_notebooks/libero_para/runbook_phase3.md (so following the runbook end-
# to-end gives you a fully-working setup with zero script edits).
#
# Path env vars (override any with `VAR=… bash submit_libero_phase3.sh`):
#   WORK              repo + openvla-mini parent dir            [default $HOME/work]
#   DATA_ROOT         dataset + base VLM parent dir              [default /scratch/$USER/data]
#   PROJECT_DIR       this repo                                  [default $WORK/VLA-in-reverse-VA-to-L-classifier]
#   OPENVLA_DIR       openvla-mini repo                          [default $WORK/openvla-mini]
#   RLDS_DIR          libero_rlds dataset                        [default $DATA_ROOT/libero_rlds]
#   BASE_VLM          base VLM dir / HF snapshot                 [default $DATA_ROOT/base_vlm]
#   POLICY_DIR        where policy ckpts go                      [default $PROJECT_DIR/checkpoints/libero_sweep/policy]
#   TOK_DIR           where the two tokenizer ckpts live         [default $PROJECT_DIR/checkpoints/libero_sweep/tokenizers]
#   LOG_DIR           SLURM log dir                              [default $PROJECT_DIR/logs]
#   CONDA_ENV_BIN     `bin/` dir of the `mmml` conda env         [default $HOME/miniconda3/envs/mmml/bin]
#   CONDA_PREFIX_DIR  prefix of the `mmml` conda env             [default $HOME/miniconda3/envs/mmml]
#
# Cluster env vars (optional — override your cluster's SBATCH directives):
#   SLURM_PARTITION   GPU partition                              [default general]
#   SLURM_CONSTRAINT  GPU type constraint, e.g. "h100|a100"      [default unset]
#   SLURM_EXCLUDE     comma-separated nodes to skip              [default unset]
#   SLURM_TIME        walltime (HH:MM:SS)                        [default 14:00:00]
#   SLURM_MEM         host RAM                                   [default 96G]
#   SLURM_CPUS        cpus-per-task                              [default 8]
#
# Usage (after running the §3 setup in the runbook):
#   bash scripts/submit_libero_phase3.sh
#
# What you get: two SLURM jobs submitted, ckpts at
#   $POLICY_DIR/minivla_libero_goal_oat_16_855_4_fullproj/
#   $POLICY_DIR/minivla_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj/
# ~6–11 h each depending on GPU class. World-readable outputs (via umask 022
# in the job + a final chmod) so the project owner can pick them up.
# ============================================================================
set -euo pipefail

# ── Path defaults (match runbook §3) ───────────────────────────────────────
: "${WORK:=$HOME/work}"
: "${DATA_ROOT:=/scratch/$USER/data}"
: "${PROJECT_DIR:=$WORK/VLA-in-reverse-VA-to-L-classifier}"
: "${OPENVLA_DIR:=$WORK/openvla-mini}"
: "${RLDS_DIR:=$DATA_ROOT/libero_rlds}"
: "${BASE_VLM:=$DATA_ROOT/base_vlm}"
: "${POLICY_DIR:=$PROJECT_DIR/checkpoints/libero_sweep/policy}"
: "${TOK_DIR:=$PROJECT_DIR/checkpoints/libero_sweep/tokenizers}"
: "${LOG_DIR:=$PROJECT_DIR/logs}"
: "${CONDA_ENV_BIN:=$HOME/miniconda3/envs/mmml/bin}"
: "${CONDA_PREFIX_DIR:=$HOME/miniconda3/envs/mmml}"

# ── Cluster directive defaults ─────────────────────────────────────────────
: "${SLURM_PARTITION:=general}"
: "${SLURM_TIME:=14:00:00}"
: "${SLURM_MEM:=96G}"
: "${SLURM_CPUS:=8}"
SLURM_CONSTRAINT_LINE=""
[[ -n "${SLURM_CONSTRAINT:-}" ]] && SLURM_CONSTRAINT_LINE="#SBATCH --constraint=$SLURM_CONSTRAINT"
SLURM_EXCLUDE_LINE=""
[[ -n "${SLURM_EXCLUDE:-}" ]]    && SLURM_EXCLUDE_LINE="#SBATCH --exclude=$SLURM_EXCLUDE"

# ── Sanity check the inputs the collaborator might have missed ─────────────
errs=0
[[ -d "$OPENVLA_DIR/vla-scripts" ]] || { echo "ERROR: OPENVLA_DIR=$OPENVLA_DIR not a valid openvla-mini checkout" >&2; errs=1; }
[[ -d "$RLDS_DIR/libero_goal_no_noops" ]] || { echo "ERROR: RLDS_DIR=$RLDS_DIR missing libero_goal_no_noops/" >&2; errs=1; }
[[ -d "$BASE_VLM" ]] || { echo "ERROR: BASE_VLM=$BASE_VLM not found" >&2; errs=1; }
for tok in oat_16_855_4 oat_16_855_4_vlm_clip0.1; do
    [[ -f "$TOK_DIR/$tok/full.pth" ]] || { echo "ERROR: tokenizer ckpt missing: $TOK_DIR/$tok/full.pth" >&2; errs=1; }
done
[[ -x "$CONDA_ENV_BIN/python" ]] || { echo "ERROR: CONDA_ENV_BIN=$CONDA_ENV_BIN/python not executable" >&2; errs=1; }
[[ $errs -ne 0 ]] && { echo "Fix the errors above and re-run. See lab_notebooks/libero_para/runbook_phase3.md." >&2; exit 1; }

mkdir -p "$LOG_DIR" "$POLICY_DIR"

# ── (tag, tokenizer_dir_name) ──────────────────────────────────────────────
JOBS=(
    "libero_goal_oat_16_855_4_fullproj             oat_16_855_4"
    "libero_goal_oat_16_855_4_vlm_clip0.1_fullproj oat_16_855_4_vlm_clip0.1"
)

submit_one() {
    local tag="$1" tok_dir="$2"
    local name="pol_${tag}"
    local run_id="minivla_${tag}"
    local run_dir="$POLICY_DIR/$run_id"

    if [[ -d "$run_dir/checkpoints" ]] && ls "$run_dir/checkpoints/"*.pt &>/dev/null; then
        echo "  SKIP $tag — policy ckpt already exists in $run_dir"
        return 1
    fi

    local script
    script=$(mktemp /tmp/${name}_XXXX.sh)
    cat > "$script" <<SBATCH
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME
$SLURM_CONSTRAINT_LINE
$SLURM_EXCLUDE_LINE
#SBATCH -o $LOG_DIR/%j_${name}.out
#SBATCH -e $LOG_DIR/%j_${name}.err

# Make outputs world-readable for the shared project tree.
umask 022

export PATH="$CONDA_ENV_BIN:\$PATH"
export CONDA_PREFIX="$CONDA_PREFIX_DIR"
export PYTHONNOUSERSITE=1
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
    vla-scripts/train.py \\
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops \\
    --vla.base_vlm "$BASE_VLM" \\
    --data_root_dir "$RLDS_DIR" \\
    --run_root_dir "$POLICY_DIR" \\
    --run_id "$run_id" \\
    --image_aug True \\
    --save_interval 5000 \\
    --vla.expected_world_size 1 \\
    --vla.global_batch_size 16 \\
    --vla.per_device_batch_size 16 \\
    --vla.freeze_vision_backbone True \\
    --vla.max_steps 50000 \\
    --vla.d_fixed 896 \\
    --vla.action_tokenizer "sweep:oat:$TOK_DIR/$tok_dir/full.pth"

# Belt-and-suspenders: open up the run dir + this job's log files for the
# project owner regardless of the user's default umask.
chmod -R o+rX "$run_dir" 2>/dev/null || true
chmod o+r "$LOG_DIR/\${SLURM_JOB_ID}"_*.{out,err} 2>/dev/null || true
SBATCH

    local sb_out
    sb_out=$(sbatch "$script" 2>&1) || { echo "  ERROR: sbatch failed for $tag: $sb_out" >&2; rm -f "$script"; return 1; }
    local jid
    jid=$(echo "$sb_out" | awk '{print $NF}')
    rm -f "$script"

    sleep 5
    local state
    state=$(sacct -j "$jid" --format=State -X -P -n 2>/dev/null | head -1)
    if [[ "$state" == "FAILED" ]]; then
        echo "  ERROR: $jid ($tag) was instant-FAILED (likely QOS)" >&2
        return 1
    fi
    echo "  Submitted: $name [$jid, $state]"
    return 0
}

echo "Submitting Phase 3 LIBERO-Goal policies..."
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "  OPENVLA_DIR=$OPENVLA_DIR"
echo "  RLDS_DIR=$RLDS_DIR"
echo "  BASE_VLM=$BASE_VLM"
echo "  TOK_DIR=$TOK_DIR"
echo "  POLICY_DIR=$POLICY_DIR"
echo "  partition=$SLURM_PARTITION  time=$SLURM_TIME  mem=$SLURM_MEM  cpus=$SLURM_CPUS"
[[ -n "${SLURM_CONSTRAINT:-}" ]] && echo "  constraint=$SLURM_CONSTRAINT"
[[ -n "${SLURM_EXCLUDE:-}" ]]    && echo "  exclude=$SLURM_EXCLUDE"
echo ""

submitted=0
for entry in "${JOBS[@]}"; do
    read -r tag tok_dir <<< "$entry"
    submit_one "$tag" "$tok_dir" && submitted=$((submitted + 1)) || true
done

echo ""
echo "Done. Submitted $submitted / ${#JOBS[@]} Phase 3 jobs."
[[ $submitted -lt ${#JOBS[@]} ]] && exit 1 || exit 0
