#!/bin/bash
# Submit verb probe jobs for all HP-sweep tokenizer checkpoints.
# Needs GPU (Transformer-based MotionVerbClassifier).
#
# Usage:
#   bash scripts/submit_verb_probe.sh           # submit all
#   bash scripts/submit_verb_probe.sh --dry_run # print commands only

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_BASE="${PROJECT_DIR}/checkpoints/calvind_hp_sweep"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

DRY_RUN=false
if [[ "${1:-}" == "--dry_run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

# All HP sweep configs: (tokenizer_type, checkpoint_dir_name)
CONFIGS=(
    # VQ-BeT vanilla
    "vq_bet vq_bet_c5_e16_g2"
    "vq_bet vq_bet_c5_e16_g4"
    "vq_bet vq_bet_c5_e64_g2"
    "vq_bet vq_bet_c10_e16_g2"
    "vq_bet vq_bet_c10_e16_g4"
    "vq_bet vq_bet_c10_e64_g2"
    # VQ-BeT aux
    "vq_bet vq_bet_verb0.1_c5e16g4_verb01"
    "vq_bet vq_bet_clip0.1_c5e16g4_clip01"
    # OAT vanilla
    "oat oat_h32_f1000_r8"
    "oat oat_h32_f256_r8"
    "oat oat_h32_f256_r4"
    "oat oat_h32_f64_r4"
    "oat oat_h16_f256_r4"
    "oat oat_h16_f256_r8"
    # QueST vanilla
    "quest quest_h32_f1000_d4"
    "quest quest_h32_f256_d4"
    "quest quest_h32_f256_d8"
    "quest quest_h32_f64_d4"
    "quest quest_h16_f256_d4"
    "quest quest_h16_f256_d2"
    # QueST aux (post-FSQ)
    "quest quest_verb0.1_h16d2_verb01"
    "quest quest_clip0.1_h16d2_clip01"
    # QueST aux (pre-FSQ)
    "quest quest_verb0.1_h16d2_verb01_prefsq"
    "quest quest_clip0.1_h16d2_clip01_prefsq"
    # QueST aux (VQ variant)
    "quest quest_h16d2_vq_vanilla"
    "quest quest_verb0.1_h16d2_vq_verb01"
    "quest quest_clip0.1_h16d2_vq_clip01"
)

for cfg in "${CONFIGS[@]}"; do
    read -r tok_type ckpt_name <<< "$cfg"
    ckpt_path="${CKPT_BASE}/${ckpt_name}/full.pth"

    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[SKIP] ${ckpt_name}: checkpoint not found"
        continue
    fi

    job_name="vprobe_${ckpt_name}"

    SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o ${LOG_DIR}/${job_name}_%j.out
#SBATCH -e ${LOG_DIR}/${job_name}_%j.err

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH:-}"

python tokenization/verb_probe_tokenizer.py \\
    --tokenizer_type ${tok_type} \\
    --checkpoint ${ckpt_path} \\
    --min_class_count 30 \\
    --device cuda \\
    --weighted_loss
EOF
)

    if $DRY_RUN; then
        echo "--- ${ckpt_name} ---"
        echo "$SCRIPT"
        echo ""
    else
        tmpfile=$(mktemp /tmp/vprobe_XXXXXX.sh)
        echo "$SCRIPT" > "$tmpfile"
        sbatch "$tmpfile"
        rm "$tmpfile"
    fi
done

if ! $DRY_RUN; then
    echo ""
    echo "Monitor: squeue -u $(whoami)"
fi
