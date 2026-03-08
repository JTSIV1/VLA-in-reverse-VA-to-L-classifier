#!/bin/bash
# Round 7: Oracle privileged-state baselines
# scene_obs (24-d) and robot_obs (15-d) from CALVIN simulator
# All: sp+wt (21 classes, weighted CE), native projection
set -euo pipefail
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# ── scene_obs, 2 frames (first+last) ─────────────────────────────────────────
JID1=$(sbatch --parsable --job-name=scene_obs_2f \
    --time=04:00:00 \
    -o logs/scene_obs_2f-%j.out \
    -e logs/scene_obs_2f-%j.err \
    old_run_scripts/run_experiment.sh scene_obs_2f scene_obs native \
    --num_frames 2 \
    --min_class_count 30 --weighted_loss)
echo "  $JID1  scene_obs_2f"

# ── scene_obs, 8 frames ──────────────────────────────────────────────────────
JID2=$(sbatch --parsable --job-name=scene_obs_8f \
    --time=04:00:00 \
    -o logs/scene_obs_8f-%j.out \
    -e logs/scene_obs_8f-%j.err \
    old_run_scripts/run_experiment.sh scene_obs_8f scene_obs native \
    --num_frames 8 \
    --min_class_count 30 --weighted_loss)
echo "  $JID2  scene_obs_8f"

# ── robot_obs, 2 frames (first+last) ─────────────────────────────────────────
JID3=$(sbatch --parsable --job-name=robot_obs_2f \
    --time=04:00:00 \
    -o logs/robot_obs_2f-%j.out \
    -e logs/robot_obs_2f-%j.err \
    old_run_scripts/run_experiment.sh robot_obs_2f robot_obs native \
    --num_frames 2 \
    --min_class_count 30 --weighted_loss)
echo "  $JID3  robot_obs_2f"

# ── robot_obs, 8 frames ──────────────────────────────────────────────────────
JID4=$(sbatch --parsable --job-name=robot_obs_8f \
    --time=04:00:00 \
    -o logs/robot_obs_8f-%j.out \
    -e logs/robot_obs_8f-%j.err \
    old_run_scripts/run_experiment.sh robot_obs_8f robot_obs native \
    --num_frames 8 \
    --min_class_count 30 --weighted_loss)
echo "  $JID4  robot_obs_8f"

echo ""
echo "Round 7 oracle jobs submitted. Monitor with: squeue -u wenjiel2"
