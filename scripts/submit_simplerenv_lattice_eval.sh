#!/bin/bash
# Submit the LATTiCE Bridge MiniVLA smoke eval on SimplerEnv WidowX tasks.

#SBATCH --job-name=simpler_lattice
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/simpler_lattice_%j.out
#SBATCH --error=logs/simpler_lattice_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/istepka/11777}"
OPENVLA_DIR="${OPENVLA_DIR:-/data/user_data/wenjiel2/Code/openvla-mini}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-/tmp/${USER}/SimplerEnv-OpenVLA}"
SIMPLERENV_REPO="${SIMPLERENV_REPO:-https://github.com/DelinQu/SimplerEnv-OpenVLA}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/simplerenv/lattice_oat_clip_pfsq_smoke}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mmml}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PRISMATIC_DATA_ROOT="${PRISMATIC_DATA_ROOT:-/data/user_data/wenjiel2/datasets/bridge_rlds}"
export PYTHONPATH="${PROJECT_DIR}:${OPENVLA_DIR}:${SIMPLERENV_DIR}:${PYTHONPATH:-}"

python - <<'PY' || NEED_SIMPLER=1
import simpler_env
print("simpler_env", getattr(simpler_env, "__file__", None))
PY
NEED_SIMPLER="${NEED_SIMPLER:-0}"

if [[ "${NEED_SIMPLER}" == "1" ]]; then
    echo "[setup] simpler_env missing; installing into ${SIMPLERENV_DIR}"
    if [[ ! -d "${SIMPLERENV_DIR}" ]]; then
        git clone "${SIMPLERENV_REPO}" --recurse-submodules --depth 1 "${SIMPLERENV_DIR}"
    fi
    python -m pip install "numpy==1.24.4"
    if [[ -d "${SIMPLERENV_DIR}/ManiSkill2_real2sim" ]]; then
        python -m pip install -e "${SIMPLERENV_DIR}/ManiSkill2_real2sim"
    fi
    python -m pip install -e "${SIMPLERENV_DIR}"
fi

# Some SimplerEnv dependencies may otherwise upgrade numpy to 2.x, which breaks
# the TensorFlow/ml_dtypes stack imported transitively by older transformers.
python -m pip install "numpy==1.24.4"
python -m pip install -e "${OPENVLA_DIR}" --quiet 2>/dev/null || true

cd "${PROJECT_DIR}"

COMMON_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --tasks "widowx_spoon_on_towel,widowx_carrot_on_plate,widowx_stack_cube,widowx_put_eggplant_in_basket"
    --episodes_per_task 3
    --seeds "0,1,2"
    --max_steps 240
)

if ! python -u policy/scripts/evaluate_simplerenv_lattice.py "${COMMON_ARGS[@]}" "$@"; then
    echo "[fallback] full smoke eval failed; running 1-task diagnostic"
    python -u policy/scripts/evaluate_simplerenv_lattice.py \
        --output_dir "${OUTPUT_DIR}_diagnostic" \
        --tasks "widowx_spoon_on_towel" \
        --episodes_per_task 1 \
        --seeds "0" \
        --max_steps 80 \
        --verbose \
        "$@"
fi
