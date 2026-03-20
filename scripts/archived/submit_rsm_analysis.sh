#!/bin/bash
# Submit RSM vs Confusion Matrix Analysis (Fixed)
# Computes symmetric confusion matrix + RSMs from 5 LLMs × 2 embedding types
# Usage: bash scripts/submit_rsm_analysis.sh

set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
LOGS="${PROJECT}/logs"
mkdir -p "${LOGS}"

# Job 1: Main RSM analysis (compute embeddings & correlations)
cat > /tmp/rsm_analysis.sh << 'SBATCH'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
source ~/.bashrc
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python3 analysis/compute_rsm_confusion_correlations.py
SBATCH

# Job 2: Visualization (runs after Job 1)
cat > /tmp/rsm_viz.sh << 'SBATCH'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd /data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier
python3 figures/plot_rsm_confusion_v2.py
SBATCH

# Submit main analysis job
echo "Submitting RSM analysis job..."
JOB1=$(sbatch \
    --job-name="rsm_analysis" \
    --partition=general \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=4:00:00 \
    -o "${LOGS}/rsm_analysis-%j.out" \
    -e "${LOGS}/rsm_analysis-%j.err" \
    /tmp/rsm_analysis.sh | awk '{print $4}')

echo "Submitted RSM analysis as job $JOB1"

# Submit visualization job (depends on main job completing)
echo "Submitting visualization job (dependent on $JOB1)..."
JOB2=$(sbatch \
    --job-name="rsm_viz" \
    --partition=general \
    --cpus-per-task=4 \
    --mem=16G \
    --time=0:30:00 \
    --dependency=afterok:$JOB1 \
    -o "${LOGS}/rsm_viz-%j.out" \
    -e "${LOGS}/rsm_viz-%j.err" \
    /tmp/rsm_viz.sh | awk '{print $4}')

echo "Submitted visualization as job $JOB2 (depends on $JOB1)"
echo ""
echo "Jobs submitted:"
echo "  Analysis:     $JOB1 (4h, 1 GPU, 64GB RAM)"
echo "  Visualization: $JOB2 (30m, CPU only, 16GB RAM) → depends on $JOB1"
echo ""
echo "Monitor with:"
echo "  squeue -u wenjiel2"
echo "  tail -f ${LOGS}/rsm_analysis-${JOB1}.out"
