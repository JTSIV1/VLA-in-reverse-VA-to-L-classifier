#!/bin/bash
# Submit the official openvla-mini SimplerEnv evaluator for a native Prismatic checkpoint.

#SBATCH --job-name=simpler_official
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/simpler_official_%j.out
#SBATCH --error=logs/simpler_official_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/istepka/11777}"
OPENVLA_DIR="${OPENVLA_DIR:-/data/user_data/wenjiel2/Code/openvla-mini}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-/tmp/${USER}/SimplerEnv-OpenVLA}"
SIMPLERENV_REPO="${SIMPLERENV_REPO:-https://github.com/DelinQu/SimplerEnv-OpenVLA}"

STUB_DIR="${PROJECT_DIR}/results/simplerenv/official_stub"
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results/simplerenv/official" "${STUB_DIR}/libero/libero/envs"
cat > "${STUB_DIR}/libero/__init__.py" <<'PY'
PY
cat > "${STUB_DIR}/libero/libero/__init__.py" <<'PY'
def get_libero_path(_name):
    return ""
PY
cat > "${STUB_DIR}/libero/libero/envs/__init__.py" <<'PY'
class OffScreenRenderEnv:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("LIBERO env is unavailable in this SimplerEnv-only run")
PY

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mmml}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PRISMATIC_DATA_ROOT="${PRISMATIC_DATA_ROOT:-/data/user_data/wenjiel2/datasets/bridge_rlds}"
export PYTHONPATH="${STUB_DIR}:${PROJECT_DIR}:${OPENVLA_DIR}:${SIMPLERENV_DIR}:${PYTHONPATH:-}"

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

python -m pip install "numpy==1.24.4"
python -m pip install -e "${OPENVLA_DIR}" --quiet 2>/dev/null || true

cd "${OPENVLA_DIR}"
python - <<'PY'
from pathlib import Path
src = Path("experiments/robot/simpler/run_simpler_eval.py")
dst = Path("/tmp") / "run_simpler_eval_openvla_official_patched.py"
text = src.read_text()
old = "        task_description = env.get_language_instruction()"
new = "        task_description = getattr(env, 'get_language_instruction', env.unwrapped.get_language_instruction)()"
if old in text and new not in text:
    text = text.replace(old, new)
old = "                img = get_simpler_img(env, obs, resize_size)"
new = "                img = get_simpler_img(getattr(env, 'unwrapped', env), obs, resize_size)"
if old in text and new not in text:
    text = text.replace(old, new)
old = "    model = get_model(cfg)"
new = """    model = get_model(cfg)
    if not hasattr(model, "_supports_cache_class"):
        model._supports_cache_class = False
    if not hasattr(model, "_supports_static_cache"):
        model._supports_static_cache = False"""
if old in text and new not in text:
    text = text.replace(old, new)
old = "    num_tasks_in_suite = task_suite.n_tasks"
new = "    num_tasks_in_suite = min(task_suite.n_tasks, int(os.environ.get('SIMPLER_MAX_TASKS', task_suite.n_tasks)))"
if old in text and new not in text:
    text = text.replace(old, new)
dst.write_text(text)
PY
cp "${OPENVLA_DIR}/.hf_token" "${STUB_DIR}/.hf_token"
cd "${STUB_DIR}"
python -u /tmp/run_simpler_eval_openvla_official_patched.py "$@"
