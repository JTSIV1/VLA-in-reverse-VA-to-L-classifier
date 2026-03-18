#!/bin/bash
# Run complementarity analysis after scene_obs_allT training completes
# Usage: sbatch --dependency=afterok:<JOBID> scripts/r3c_complementarity.sh
#    or: bash scripts/r3c_complementarity.sh  (if training already done)
set -e

PROJECT="/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier"
LOGS="${PROJECT}/logs"

SCRIPT="/tmp/r3c_complementarity_run.sh"
cat > "$SCRIPT" << SBATCH
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
# Ensure conda torch takes priority over ~/.local torch 2.8.0
export PYTHONPATH=\$(python -c "import site; print(site.getsitepackages()[0])"):\$PYTHONPATH
cd ${PROJECT}
python3 analyze_repr_complementarity.py
SBATCH

sbatch --job-name="r3c_compl" \
       --partition=general \
       --gres=gpu:1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=2:00:00 \
       --dependency=afterok:6574533 \
       -o "${LOGS}/r3c_compl-%j.out" \
       -e "${LOGS}/r3c_compl-%j.err" \
       "$SCRIPT"
echo "Submitted r3c_compl (depends on 6574457)"
