#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Bridge policy sweep — new nested directory structure.
#
# Directory layout:
#   tokenizers/<base_type>/<condition>/full.pth
#   policy/<base_type>/<condition>/checkpoints/*.pt
#   policy/<base_type>/<condition>_fullproj/checkpoints/*.pt
#
# Stages per tokenizer condition:
#   1. verb_probe     — latent + tokid probes (+ native baseline, once)
#   2. train_policy   — standard (random init embeddings)
#   3. train_fullproj — fullproj embeddings (static for VQ-BeT, dynamic for OAT/QueST)
#   4. eval_policy    — teacher-forced L1 eval on val set (both standard + fullproj)
#
# Usage:
#   bash run_sweep.sh                   # full pipeline
#   bash run_sweep.sh --verb-probe-only # verb probes only
#   bash run_sweep.sh --eval-only       # teacher-forced eval only
#   bash run_sweep.sh --dry-run         # print sbatch scripts without submitting
#   bash run_sweep.sh --force           # retrain even if checkpoints exist
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
RLDS_DIR="/data/user_data/wenjiel2/datasets/bridge_rlds"
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"
LOG_DIR="$PROJECT_DIR/logs"
SWEEP_DIR="$PROJECT_DIR/checkpoints/bridge_sweep"
TOK_DIR="$SWEEP_DIR/tokenizers"
POLICY_DIR="$SWEEP_DIR/policy"

# ── Bridge data ───────────────────────────────────────────────────────
BRIDGE_SHARD_DIR="/data/user_data/wenjiel2/datasets/bridge_actions"
BRIDGE_CSV="data/bridge_episodes_filtered.csv"

# ── Verb probe settings ──────────────────────────────────────────────
PROBE_EPOCHS=100
PROBE_BATCH_SIZE=32
PROBE_D_MODEL=128
PROBE_LR=1e-4
PROBE_PATIENCE=15
PROBE_LABEL_SMOOTHING=0.1

# ── Policy settings ──────────────────────────────────────────────────
POLICY_MAX_STEPS=50000
D_FIXED=896  # for fullproj

# ── SLURM ────────────────────────────────────────────────────────────
PARTITION=general
EXCLUDE_NODES="babel-l5-[16,20],babel-m9-[16,20],babel-n9-20,babel-z5-28"
PROBE_TIME="4:00:00"
POLICY_TIME="14:00:00"
TF_TIME="4:00:00"

mkdir -p "$LOG_DIR"

# ── CLI flags ────────────────────────────────────────────────────────
DRY_RUN=false
FORCE=false
VERB_PROBE_ONLY=false
EVAL_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --force)            FORCE=true ;;
        --verb-probe-only)  VERB_PROBE_ONLY=true ;;
        --eval-only)        EVAL_ONLY=true ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
#                        Helper functions
# ═══════════════════════════════════════════════════════════════════════

submit() {
    local script="$1" name="$2" dep="${3:-}"
    if $DRY_RUN; then
        echo "    [DRY RUN] $name" >&2
        return
    fi
    local tmp; tmp=$(mktemp "/tmp/${name}_XXXX.sh")
    echo "$script" > "$tmp"
    local cmd=(sbatch)
    [[ -n "$dep" ]] && cmd+=(--dependency "afterok:$dep")
    cmd+=("$tmp")
    local out; out=$("${cmd[@]}" 2>&1)
    local jid; jid=$(echo "$out" | awk '{print $NF}')
    rm -f "$tmp"
    echo "$jid"
}

sbatch_header() {
    local name="$1" time="$2" mem="${3:-32G}"
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH --exclude=$EXCLUDE_NODES
#SBATCH -o $LOG_DIR/%j_${name}.out
#SBATCH -e $LOG_DIR/%j_${name}.err
EOF
}

preamble_project() {
    cat <<EOF

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"

EOF
}

preamble_openvla() {
    cat <<EOF

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PYTHONNOUSERSITE=1
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

EOF
}

ckpt_exists() {
    ! $FORCE && [[ -f "$1" ]]
}

dir_has_ckpt() {
    # Check if a policy dir has at least one .pt checkpoint
    ! $FORCE && [[ -d "$1/checkpoints" ]] && ls "$1/checkpoints/"*.pt &>/dev/null
}

infer_tok_type() {
    local base="$1"
    case "$base" in
        vq_bet_*) echo "vq_bet" ;;
        quest_*)  echo "quest" ;;
        oat_*)    echo "oat" ;;
        *)        echo "unknown" ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════
#                        Stage builders
# ═══════════════════════════════════════════════════════════════════════

build_probe() {
    local base="$1" cond="$2" mode="$3"
    local tok_type; tok_type=$(infer_tok_type "$base")
    local tok_ckpt="$TOK_DIR/$base/$cond/full.pth"
    local save_path="$TOK_DIR/$base/$cond/probe_${mode}.pth"
    local name="probe_${mode}_${base}_${cond}"

    local cmd="python -u verb_probe/train_verb_probe.py"
    if [[ "$mode" == "native" ]]; then
        cmd+=" --action_rep native"
    elif [[ "$mode" == "tokid" ]]; then
        cmd+=" --action_rep $tok_type --tokenizer_type $tok_type --tokenizer_ckpt $tok_ckpt"
    elif [[ "$mode" == "latent" ]]; then
        cmd+=" --action_rep latent --tokenizer_type $tok_type --tokenizer_ckpt $tok_ckpt"
    fi
    cmd+=" --modality action_only --dataset bridge"
    cmd+=" --epochs $PROBE_EPOCHS --batch_size $PROBE_BATCH_SIZE"
    cmd+=" --d_model $PROBE_D_MODEL --lr $PROBE_LR"
    cmd+=" --patience $PROBE_PATIENCE --label_smoothing $PROBE_LABEL_SMOOTHING"
    cmd+=" --min_class_count 30 --weighted_loss"
    cmd+=" --save_path $save_path"
    cmd+=" --shard_dir $BRIDGE_SHARD_DIR --bridge_csv $BRIDGE_CSV"

    echo "$(sbatch_header "$name" "$PROBE_TIME")$(preamble_project)
$cmd"
}

build_policy() {
    local base="$1" cond="$2" d_fixed="${3:-0}"
    local tok_type; tok_type=$(infer_tok_type "$base")
    local tok_ckpt="$TOK_DIR/$base/$cond/full.pth"

    local pol_cond="$cond"
    [[ "$d_fixed" -gt 0 ]] && pol_cond="${cond}_fullproj"
    local run_id="${base}/${pol_cond}"
    local name="pol_${base}_${pol_cond}"

    local cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
    cmd+=" vla-scripts/train.py"
    cmd+=" --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-bridge"
    cmd+=" --vla.base_vlm $BASE_VLM"
    cmd+=" --data_root_dir $RLDS_DIR"
    cmd+=" --run_root_dir $POLICY_DIR"
    cmd+=" --run_id $run_id"
    cmd+=" --image_aug True"
    cmd+=" --save_interval 5000"
    cmd+=" --vla.expected_world_size 1"
    cmd+=" --vla.global_batch_size 16"
    cmd+=" --vla.per_device_batch_size 16"
    cmd+=" --vla.freeze_vision_backbone True"
    cmd+=" --vla.max_steps $POLICY_MAX_STEPS"
    cmd+=" --vla.action_tokenizer 'sweep:${tok_type}:${tok_ckpt}'"
    [[ "$d_fixed" -gt 0 ]] && cmd+=" --vla.d_fixed $d_fixed"

    echo "$(sbatch_header "$name" "$POLICY_TIME" "64G")$(preamble_openvla)
$cmd"
}

build_eval() {
    local base="$1" pol_cond="$2"
    local pol_dir="$POLICY_DIR/$base/$pol_cond"
    local out_dir="$SWEEP_DIR/results/teacher_force/$base"
    local name="tf_${base}_${pol_cond}"

    local cmd="python -u policy/eval_policy.py"
    cmd+=" --policy_dir $pol_dir"
    cmd+=" --mode teacher_force"
    cmd+=" --dataset bridge"
    cmd+=" --output_dir $out_dir"

    echo "$(sbatch_header "$name" "$TF_TIME" "64G")$(preamble_project)
export PRISMATIC_DATA_ROOT=\"$RLDS_DIR\"
export PYTHONPATH=\"$OPENVLA_DIR:\${PYTHONPATH:-}\"

$cmd"
}

# ═══════════════════════════════════════════════════════════════════════
#                           Main loop
# ═══════════════════════════════════════════════════════════════════════

echo "Sweep: bridge_sweep (nested directory structure)"
echo "Dry run: $DRY_RUN | Force: $FORCE"

n_probe=0
n_pol_std=0
n_pol_fp=0
n_eval=0
n_skip=0

# Iterate over all active tokenizers
for base_dir in "$TOK_DIR"/*/; do
    base=$(basename "$base_dir")
    [[ "$base" == "archive" ]] && continue

    tok_type=$(infer_tok_type "$base")

    for cond_dir in "$base_dir"*/; do
        [[ -d "$cond_dir" ]] || continue
        cond=$(basename "$cond_dir")
        # Skip _old, .collapsed, etc.
        [[ "$cond" == *_old* || "$cond" == *.collapsed* ]] && continue
        # Must have full.pth
        [[ -f "$cond_dir/full.pth" ]] || continue

        echo ""
        echo "  [$base/$cond]  (type=$tok_type)"

        # ── Verb probes ──────────────────────────────────────────────
        for mode in native latent tokid; do
            probe_path="$cond_dir/probe_${mode}.pth"
            if ckpt_exists "$probe_path"; then
                echo "    probe ($mode): SKIP"
            elif ! $EVAL_ONLY; then
                script=$(build_probe "$base" "$cond" "$mode")
                jid=$(submit "$script" "probe_${mode}_${base}_${cond}")
                [[ -n "$jid" ]] && echo "    probe ($mode): job $jid"
                n_probe=$((n_probe + 1))
            fi
        done

        if $VERB_PROBE_ONLY; then
            continue
        fi

        # ── Standard policy ──────────────────────────────────────────
        pol_dir="$POLICY_DIR/$base/$cond"
        if dir_has_ckpt "$pol_dir"; then
            echo "    policy (standard): SKIP"
        elif ! $EVAL_ONLY; then
            script=$(build_policy "$base" "$cond" 0)
            pol_jid=$(submit "$script" "pol_${base}_${cond}")
            [[ -n "$pol_jid" ]] && echo "    policy (standard): job $pol_jid"
            n_pol_std=$((n_pol_std + 1))
        fi

        # ── Fullproj policy ──────────────────────────────────────────
        fp_dir="$POLICY_DIR/$base/${cond}_fullproj"
        if dir_has_ckpt "$fp_dir"; then
            echo "    policy (fullproj): SKIP"
        elif ! $EVAL_ONLY; then
            script=$(build_policy "$base" "$cond" "$D_FIXED")
            fp_jid=$(submit "$script" "pol_${base}_${cond}_fullproj")
            [[ -n "$fp_jid" ]] && echo "    policy (fullproj): job $fp_jid"
            n_pol_fp=$((n_pol_fp + 1))
        fi

        # ── Teacher-force eval (standard) ────────────────────────────
        # eval_policy.py appends condition to output_dir, so result is at base/cond/metrics.json
        tf_results="$SWEEP_DIR/results/teacher_force/$base/$cond/teacher_force_metrics.json"
        if ckpt_exists "$tf_results"; then
            echo "    eval (standard): SKIP"
        elif dir_has_ckpt "$pol_dir"; then
            script=$(build_eval "$base" "$cond")
            dep=""
            jid=$(submit "$script" "tf_${base}_${cond}" "$dep")
            [[ -n "$jid" ]] && echo "    eval (standard): job $jid"
            n_eval=$((n_eval + 1))
        elif [[ -n "${pol_jid:-}" ]]; then
            script=$(build_eval "$base" "$cond")
            jid=$(submit "$script" "tf_${base}_${cond}" "$pol_jid")
            [[ -n "$jid" ]] && echo "    eval (standard): job $jid (dep $pol_jid)"
            n_eval=$((n_eval + 1))
        fi

        # ── Teacher-force eval (fullproj) ────────────────────────────
        tf_fp_results="$SWEEP_DIR/results/teacher_force/$base/${cond}_fullproj/teacher_force_metrics.json"
        if ckpt_exists "$tf_fp_results"; then
            echo "    eval (fullproj): SKIP"
        elif dir_has_ckpt "$fp_dir"; then
            script=$(build_eval "$base" "${cond}_fullproj")
            jid=$(submit "$script" "tf_${base}_${cond}_fullproj")
            [[ -n "$jid" ]] && echo "    eval (fullproj): job $jid"
            n_eval=$((n_eval + 1))
        elif [[ -n "${fp_jid:-}" ]]; then
            script=$(build_eval "$base" "${cond}_fullproj")
            jid=$(submit "$script" "tf_${base}_${cond}_fullproj" "$fp_jid")
            [[ -n "$jid" ]] && echo "    eval (fullproj): job $jid (dep $fp_jid)"
            n_eval=$((n_eval + 1))
        fi

        # Reset per-condition job IDs
        pol_jid=""
        fp_jid=""
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Probes: $n_probe | Standard: $n_pol_std | Fullproj: $n_pol_fp | Eval: $n_eval"
echo "═══════════════════════════════════════════════════════════════"

if ! $DRY_RUN; then
    echo "Monitor: squeue -u \$(whoami)"
fi
