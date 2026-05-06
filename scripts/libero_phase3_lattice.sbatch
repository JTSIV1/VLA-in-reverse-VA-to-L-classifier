#!/bin/bash
# ============================================================================
# Phase 3 — LIBERO-Goal LATTiCE (vlm_clip0.1 + TiCE) policy fine-tune.
# Companion to lab_notebooks/libero_para/runbook_phase3.md (run 2 of 2).
#
# Identical to libero_phase3_vanilla.sbatch except for which tokenizer
# checkpoint the policy embeds against (LATTiCE = oat_16_855_4_vlm_clip0.1).
#
# To submit:
#   1. Edit the "EDIT ME" path block + SBATCH cluster directives below.
#   2. sbatch scripts/libero_phase3_lattice.sbatch
#
# ~11 h on 1×L40S; 14 h walltime budget.
# ============================================================================

# ── Cluster directives — EDIT for your scheduler ───────────────────────────
#SBATCH --job-name=pol_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj
#SBATCH --partition=general          # ← change to your cluster's GPU partition
#SBATCH --gres=gpu:1                 # Pick the best GPU class available
#                                    #   (H100 > A100 80G > A100 40G > A6000 > L40S).
#                                    #   Faster GPU = shorter walltime; we used L40S
#                                    #   only because it was what we had.
# #SBATCH --constraint=h100|a100      # ← uncomment + edit for your cluster's tags
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00              # H100/A100 will finish in ~6–8 h; the budget is conservative
# #SBATCH --exclude=                  # ← uncomment + populate if you have bad nodes
#SBATCH -o logs/%j_pol_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj.out
#SBATCH -e logs/%j_pol_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj.err

# Make outputs world-readable for the shared project dir.
umask 022

# ── EDIT ME — fill in your absolute paths ──────────────────────────────────
PROJECT_DIR="$HOME/work/VLA-in-reverse-VA-to-L-classifier"
OPENVLA_DIR="$HOME/work/openvla-mini"
RLDS_DIR="$HOME/work/data/libero_rlds"
BASE_VLM="$HOME/work/data/base_vlm"   # or the snapshot dir under HF cache
POLICY_DIR="$PROJECT_DIR/checkpoints/libero_sweep/policy"
TOK_DIR="$PROJECT_DIR/checkpoints/libero_sweep/tokenizers"
CONDA_ENV_BIN="$HOME/miniconda3/envs/mmml/bin"
CONDA_PREFIX_DIR="$HOME/miniconda3/envs/mmml"
# ────────────────────────────────────────────────────────────────────────────

TAG=libero_goal_oat_16_855_4_vlm_clip0.1_fullproj
TOK_DIR_NAME=oat_16_855_4_vlm_clip0.1
RUN_ID=minivla_${TAG}

mkdir -p "$POLICY_DIR" logs

export PATH="$CONDA_ENV_BIN:$PATH"
export CONDA_PREFIX="$CONDA_PREFIX_DIR"
export PYTHONNOUSERSITE=1
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline

cd "$OPENVLA_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops \
    --vla.base_vlm "$BASE_VLM" \
    --data_root_dir "$RLDS_DIR" \
    --run_root_dir "$POLICY_DIR" \
    --run_id "$RUN_ID" \
    --image_aug True \
    --save_interval 5000 \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 16 \
    --vla.per_device_batch_size 16 \
    --vla.freeze_vision_backbone True \
    --vla.max_steps 50000 \
    --vla.d_fixed 896 \
    --vla.action_tokenizer "sweep:oat:$TOK_DIR/$TOK_DIR_NAME/full.pth"

# Make outputs readable to the project owner regardless of the user's umask.
# (The `umask 022` above only affects this shell's children; SLURM may write
# the log files with the user's default umask, and `train.py`'s heredoc-style
# config writes can be tightened too. This `chmod` is the belt-and-suspenders.)
chmod -R o+rX "$POLICY_DIR/$RUN_ID" 2>/dev/null || true
chmod o+r "$PROJECT_DIR/logs/${SLURM_JOB_ID}"_*.{out,err} 2>/dev/null || true
