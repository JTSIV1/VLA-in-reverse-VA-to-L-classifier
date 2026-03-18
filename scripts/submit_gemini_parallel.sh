#!/bin/bash
# Submit parallel Gemini annotation jobs for CALVIN.
# Each shard handles a non-overlapping episode range → separate output files.
# After all shards finish, merge with: cat calvin_training_shard*.jsonl >> calvin_training.jsonl
#
# Usage: bash scripts/submit_gemini_parallel.sh

set -e
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
mkdir -p logs

# ── Current progress ──────────────────────────────────────────────────────────
TRAIN_DONE=$(wc -l < data/hierarchy_annotations/calvin_training.jsonl 2>/dev/null || echo 0)
TRAIN_ERRORS=$(wc -l < data/hierarchy_annotations/errors_training.jsonl 2>/dev/null || echo 0)
TRAIN_START=$((TRAIN_DONE + TRAIN_ERRORS))
TRAIN_TOTAL=5124

VAL_DONE=$(wc -l < data/hierarchy_annotations/calvin_validation.jsonl 2>/dev/null || echo 0)
VAL_ERRORS=$(wc -l < data/hierarchy_annotations/errors_validation.jsonl 2>/dev/null || echo 0)
VAL_START=$((VAL_DONE + VAL_ERRORS))
VAL_TOTAL=1011

echo "Training: $TRAIN_DONE done, starting from ep $TRAIN_START / $TRAIN_TOTAL"
echo "Validation: $VAL_DONE done, starting from ep $VAL_START / $VAL_TOTAL"

# ── Training: 8 parallel shards ──────────────────────────────────────────────
N_SHARDS=8
TRAIN_REMAINING=$((TRAIN_TOTAL - TRAIN_START))
CHUNK=$((TRAIN_REMAINING / N_SHARDS))

echo ""
echo "=== Submitting $N_SHARDS training shards (chunk=$CHUNK eps each) ==="

for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD_START=$((TRAIN_START + i * CHUNK))
    if [ $i -eq $((N_SHARDS - 1)) ]; then
        # Last shard gets the remainder
        SHARD_MAX=$((TRAIN_TOTAL - SHARD_START))
    else
        SHARD_MAX=$CHUNK
    fi

    echo "  Shard $i: eps $SHARD_START..$((SHARD_START + SHARD_MAX - 1)) ($SHARD_MAX eps)"

    sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gem_tr_${i}
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=ybisk
#SBATCH --output=logs/gemini_train_shard${i}_%j.out
#SBATCH --error=logs/gemini_train_shard${i}_%j.err

cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
source /data/user_data/wenjiel2/miniconda3/etc/profile.d/conda.sh
conda activate mmml

echo "Shard $i: start=$SHARD_START max=$SHARD_MAX"
python3 scripts/annotate_calvin_hierarchy.py \
    --split training \
    --start_from $SHARD_START \
    --max_episodes $SHARD_MAX \
    --shard_id $i \
    --rate_limit_delay 0.5
SBATCH_EOF
done

# ── Validation: 1 job for remaining ──────────────────────────────────────────
VAL_REMAINING=$((VAL_TOTAL - VAL_START))
if [ $VAL_REMAINING -gt 0 ]; then
    echo ""
    echo "=== Submitting validation job ($VAL_REMAINING eps remaining) ==="

    sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gem_val
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

echo "Validation: start=$VAL_START remaining=$VAL_REMAINING"
python3 scripts/annotate_calvin_hierarchy.py \
    --split validation \
    --start_from $VAL_START \
    --rate_limit_delay 0.5
SBATCH_EOF
else
    echo ""
    echo "Validation already complete ($VAL_DONE / $VAL_TOTAL)"
fi

echo ""
echo "Monitor: squeue -u wenjiel2 | grep gem"
echo "After all done, merge: cat data/hierarchy_annotations/calvin_training_shard*.jsonl >> data/hierarchy_annotations/calvin_training.jsonl"
