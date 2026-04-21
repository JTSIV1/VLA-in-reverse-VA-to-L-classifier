#!/bin/bash
set -euo pipefail

# Submit all 3 failure-analysis experiments to SLURM.
#
# Each job: trains for 30 epochs → evaluates best checkpoint →
#           runs analyze_failures.py → writes results/failure_analysis/<name>/
#
# Usage: bash submit_failure_analysis.sh
#
# What gets submitted:
#   1. action_only  — action-only, native continuous actions
#   2. scene_obs    — scene-obs MLP (24-d state-change oracle + delta)
#   3. fusion       — action + scene token fusion (best overall model)

PROJECT_DIR="${HOME}/11777"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results figures

SBATCH="sbatch --parsable"

echo "=== Submitting failure analysis experiments ==="

# 1. Action-only (native) — kinematic-only baseline
JOB1=$($SBATCH \
    --job-name=fa_action_only \
    --output=logs/fa_action_only-%j.out \
    --error=logs/fa_action_only-%j.err \
    run_failure_analysis.sh action_only action_only native)
echo "[1] action_only (native)        → job $JOB1"

# 2. Scene-obs MLP — state-change-only oracle
#    scene_dim=48: [start_obs(24) | delta_obs(24)]
JOB2=$($SBATCH \
    --job-name=fa_scene_obs \
    --output=logs/fa_scene_obs-%j.out \
    --error=logs/fa_scene_obs-%j.err \
    run_failure_analysis.sh scene_obs scene_mlp native)
echo "[2] scene_obs (scene_mlp/native) → job $JOB2"

# 3. Fusion — action + scene token, best overall
JOB3=$($SBATCH \
    --job-name=fa_fusion \
    --output=logs/fa_fusion-%j.out \
    --error=logs/fa_fusion-%j.err \
    run_failure_analysis.sh fusion scene_token native)
echo "[3] fusion (scene_token/native)  → job $JOB3"

echo ""
echo "=== 3 jobs submitted ==="
echo "Monitor:  squeue -u \$USER"
echo "Logs:     tail -f logs/fa_<name>-<jobid>.out"
echo "Outputs:  results/failure_analysis/{action_only,scene_obs,fusion}/"
