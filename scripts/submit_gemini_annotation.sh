#!/bin/bash
# Submit Gemini annotation jobs for CALVIN training and validation splits.
# Uses cpu partition (no GPU needed — just API calls).
# Resumes from where previous runs left off.

set -e

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier

# Count how many episodes are already done
TRAIN_DONE=$(wc -l < data/hierarchy_annotations/calvin_training.jsonl 2>/dev/null || echo 0)
VAL_DONE=$(wc -l < data/hierarchy_annotations/calvin_validation.jsonl 2>/dev/null || echo 0)

# Account for errors (start_from = last_ep_index + 1, but we use line count + error count)
TRAIN_ERRORS=$(wc -l < data/hierarchy_annotations/errors_training.jsonl 2>/dev/null || echo 0)
VAL_ERRORS=$(wc -l < data/hierarchy_annotations/errors_validation.jsonl 2>/dev/null || echo 0)

TRAIN_START=$((TRAIN_DONE + TRAIN_ERRORS))
VAL_START=$((VAL_DONE + VAL_ERRORS))

echo "Training: resuming from episode $TRAIN_START (done=$TRAIN_DONE, errors=$TRAIN_ERRORS)"
echo "Validation: resuming from episode $VAL_START (done=$VAL_DONE, errors=$VAL_ERRORS)"

# Submit training job
TRAIN_JOB=$(sbatch --parsable <<'SBATCH_TRAIN'
#!/bin/bash
#SBATCH --job-name=gemini_train
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=ybisk
#SBATCH --output=logs/gemini_train_%j.out
#SBATCH --error=logs/gemini_train_%j.err

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

# Re-count at job start time (in case we submit multiple times)
TRAIN_DONE=$(wc -l < data/hierarchy_annotations/calvin_training.jsonl 2>/dev/null || echo 0)
TRAIN_ERRORS=$(wc -l < data/hierarchy_annotations/errors_training.jsonl 2>/dev/null || echo 0)
START=$((TRAIN_DONE + TRAIN_ERRORS))

echo "Starting training annotation from episode $START"
python3 scripts/annotate_calvin_hierarchy.py \
    --split training \
    --start_from $START \
    --rate_limit_delay 1.0
SBATCH_TRAIN
)

echo "Submitted training job: $TRAIN_JOB"

# Submit validation job
VAL_JOB=$(sbatch --parsable <<'SBATCH_VAL'
#!/bin/bash
#SBATCH --job-name=gemini_val
#SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=ybisk
#SBATCH --output=logs/gemini_val_%j.out
#SBATCH --error=logs/gemini_val_%j.err

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

VAL_DONE=$(wc -l < data/hierarchy_annotations/calvin_validation.jsonl 2>/dev/null || echo 0)
VAL_ERRORS=$(wc -l < data/hierarchy_annotations/errors_validation.jsonl 2>/dev/null || echo 0)
START=$((VAL_DONE + VAL_ERRORS))

echo "Starting validation annotation from episode $START"
python3 scripts/annotate_calvin_hierarchy.py \
    --split validation \
    --start_from $START \
    --rate_limit_delay 1.0
SBATCH_VAL
)

echo "Submitted validation job: $VAL_JOB"
echo "Monitor: squeue -u wenjiel2 | grep gemini"
