#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Experiment sweep.  Edit the settings below, then run:
#   bash run_sweep.sh              # submit to SLURM
#   bash run_sweep.sh --dry-run    # print sbatch scripts only
#
# Any setting that is an array () means "sweep over these values".
# The script submits one job per combination (cartesian product).
# ═══════════════════════════════════════════════════════════════════════

# ── What to run ──────────────────────────────────────────────────────
STAGES=(tokenizer probe policy)     # any combo of: tokenizer  probe  policy

# ── Sweep dimensions (arrays = sweep, scalars = fixed) ───────────────
TOKENIZER=(quest)
AUX_HEAD=(none "verb:0.1" "clip:0.1")  # none | head:lambda  (arrays = sweep)

# Tokenizer config overrides: each entry is one config to train.
# KEY=VAL pairs are passed to train_tokenizer.py --set.
TOK_SET=(
  "horizon=16 fsq_levels=[4,4,4,4] downsample_factor=2"
  "horizon=32 fsq_levels=[8,5,5,5] downsample_factor=4"
  "horizon=16 fsq_levels=[4,4,4,4] downsample_factor=4"
)

# ── OR: use existing checkpoints (comment out TOK_SET above) ────────
# TAGS and CKPT must match 1-to-1. Tokenizer stage is skipped.
# TAGS=(quest_h16f256d2  quest_h32f1000d4  quest_h16f256d4)
# CKPT=(calvind_hp_sweep/quest_h16_f256_d2 \
#       calvind_hp_sweep/quest_h32_f1000_d4 \
#       calvind_hp_sweep/quest_h16_f256_d4)

# ── Fixed settings ───────────────────────────────────────────────────
TOK_EPOCHS=200
TOK_BATCH_SIZE=32
TOK_LR=1e-4
PROBE_EPOCHS=50
PROBE_BATCH_SIZE=64
PROBE_D_MODEL=128
POLICY_MODEL=minivla                # minivla | openvla
POLICY_BATCH_SIZE=8
POLICY_LR=5e-4
POLICY_MAX_STEPS=50000

# ── SLURM ────────────────────────────────────────────────────────────
PARTITION=general
TOK_TIME="1:00:00"
PROBE_TIME="4:00:00"
POLICY_TIME="12:00:00"

# ═══════════════════════════════════════════════════════════════════════
#                    Nothing to edit below this line
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail
# Default optional arrays that may be commented out above
TAGS=("${TAGS[@]+"${TAGS[@]}"}")
CKPT=("${CKPT[@]+"${CKPT[@]}"}")
TOK_SET=("${TOK_SET[@]+"${TOK_SET[@]}"}")
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
RLDS_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
LOG_DIR="$PROJECT_DIR/logs"
CKPT_BASE="$PROJECT_DIR/checkpoints"
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

has_stage() { printf '%s\n' "${STAGES[@]}" | grep -qx "$1"; }

# Parse "verb:0.1" -> aux_name=verb aux_lam=0.1; "none" -> aux_name=none aux_lam=0
parse_aux() {
    local spec="$1"
    if [[ "$spec" == "none" ]]; then
        aux_name="none"; aux_lam="0"
    else
        aux_name="${spec%%:*}"
        aux_lam="${spec#*:}"
    fi
}

# ── sbatch helpers ───────────────────────────────────────────────────

submit() {
    local script="$1" name="$2" dep="${3:-}"
    if $DRY_RUN; then
        echo "" >&2
        echo "============================================================" >&2
        echo "  $name (dry run)" >&2
        echo "============================================================" >&2
        echo "$script" >&2
        return
    fi
    local tmp; tmp=$(mktemp "/tmp/${name}_XXXX.sh")
    echo "$script" > "$tmp"
    local cmd=(sbatch)
    [[ -n "$dep" ]] && cmd+=(--dependency "afterok:$dep")
    cmd+=("$tmp")
    local out; out=$("${cmd[@]}" 2>&1)
    local jid; jid=$(echo "$out" | awk '{print $NF}')
    echo "  Submitted $name: job $jid" >&2
    rm -f "$tmp"
    echo "$jid"  # captured by caller
}

sbatch_header() {
    local name="$1" time="$2" mem="${3:-32G}" gres="${4:-gpu:1}"
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=$PARTITION
#SBATCH --gres=$gres
#SBATCH --cpus-per-task=8
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH -o $LOG_DIR/${name}_%j.out
#SBATCH -e $LOG_DIR/${name}_%j.err
EOF
}

preamble() {
    cat <<EOF

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"

EOF
}

# ── Stage builders ───────────────────────────────────────────────────

build_tokenizer() {
    local tag="$1" tok="$2" aux="$3" lam="$4" tok_set="${5:-}"
    local name="tok_${tag}"
    local cmd="python -u tokenization/train_tokenizer.py"
    cmd+=" --tokenizer $tok --dataset calvin"
    cmd+=" --epochs $TOK_EPOCHS --batch_size $TOK_BATCH_SIZE --lr $TOK_LR"
    cmd+=" --save_dir $CKPT_BASE --min_class_count 30 --max_chunks 8"
    cmd+=" --tag $tag"
    [[ "$aux" != "none" ]] && cmd+=" --aux_head $aux --aux_lambda $lam"
    [[ -n "$tok_set" ]] && cmd+=" --set $tok_set"

    echo "$(sbatch_header "$name" "$TOK_TIME")$(preamble)
$cmd"
}

build_probe() {
    local tag="$1" tok="$2" ckpt="$3" mode="$4"
    local name="probe_${mode}_${tag}"
    local save="$CKPT_BASE/$tag/probe_${mode}_best.pth"
    local cmd="python -u verb_probe/train_verb_probe.py"

    if [[ "$mode" == "native" ]]; then
        cmd+=" --action_rep native"
    elif [[ "$mode" == "tokid" ]]; then
        cmd+=" --action_rep $tok --tokenizer_type $tok --tokenizer_ckpt $ckpt"
    elif [[ "$mode" == "latent" ]]; then
        cmd+=" --action_rep latent --tokenizer_type $tok --tokenizer_ckpt $ckpt"
    fi

    cmd+=" --modality action_only"
    cmd+=" --epochs $PROBE_EPOCHS --batch_size $PROBE_BATCH_SIZE"
    cmd+=" --d_model $PROBE_D_MODEL"
    cmd+=" --min_class_count 30 --weighted_loss"
    cmd+=" --save_path $save"

    echo "$(sbatch_header "$name" "$PROBE_TIME")$(preamble)
$cmd"
}

build_policy() {
    local tag="$1" tok="$2" ckpt="$3"
    local name="pol_${tag}"
    local run_dir="$PROJECT_DIR/runs/${POLICY_MODEL}_${tag}"
    local vla_config="prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin"

    local header; header=$(sbatch_header "$name" "$POLICY_TIME" "64G")
    local pre
    pre=$(cat <<EOF

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmml
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

mkdir -p "$run_dir"

EOF
)

    local cmd
    if [[ "$POLICY_MODEL" == "minivla" ]]; then
        cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
        cmd+=" vla-scripts/train.py"
        cmd+=" --vla.type $vla_config"
        cmd+=" --vla.base_vlm $BASE_VLM"
        cmd+=" --data_root_dir $RLDS_DIR"
        cmd+=" --run_root_dir $run_dir"
        cmd+=" --image_aug True"
        cmd+=" --save_interval 5000"
        cmd+=" --run_id_note $tag"
        cmd+=" --vla.expected_world_size 1"
        cmd+=" --vla.global_batch_size 16"
        cmd+=" --vla.per_device_batch_size 16"
        cmd+=" --vla.freeze_vision_backbone True"
        if [[ "$tok" != "bin" && -n "$ckpt" ]]; then
            cmd+=" --vla.action_tokenizer 'sweep:${tok}:${ckpt}'"
        fi
    else
        cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
        cmd+=" vla-scripts/finetune.py"
        cmd+=" --vla_path openvla/openvla-7b"
        cmd+=" --data_root_dir $RLDS_DIR"
        cmd+=" --dataset_name calvin_dataset"
        cmd+=" --run_root_dir $run_dir"
        cmd+=" --lora_rank 32"
        cmd+=" --batch_size $POLICY_BATCH_SIZE"
        cmd+=" --grad_accumulation_steps 2"
        cmd+=" --learning_rate $POLICY_LR"
        cmd+=" --max_steps $POLICY_MAX_STEPS"
        cmd+=" --save_steps 5000 --val_steps 1000 --warmup_steps 500"
        cmd+=" --max_grad_norm 1.0 --image_aug True"
        cmd+=" --shuffle_buffer_size 50000"
        cmd+=" --run_id_note $tag"
        if [[ "$tok" != "bin" && -n "$ckpt" ]]; then
            cmd+=" --action_tokenizer_type $tok"
            cmd+=" --action_tokenizer_ckpt $ckpt"
        fi
    fi

    echo "${header}${pre}
${cmd}"
}

# ── Run one condition ────────────────────────────────────────────────

run_condition() {
    local tag="$1" tok="$2" ckpt_dir="$3" aux="$4" lam="$5" tok_set="${6:-}"
    local ckpt=""
    [[ "$ckpt_dir" != "-" ]] && ckpt="$CKPT_BASE/$ckpt_dir/full.pth"

    # Display name: tok[_aux][_tag]
    local display="${tok}"
    [[ "$aux" != "none" ]] && display+="_${aux}${lam}"
    [[ -n "$tag" ]] && display+="_${tag}"
    echo ""
    echo "  [$display]"

    # Stage 1: tokenizer
    local tok_jid=""
    if has_stage tokenizer && [[ -z "$ckpt" ]]; then
        local script; script=$(build_tokenizer "$tag" "$tok" "$aux" "$lam" "$tok_set")
        tok_jid=$(submit "$script" "tok_${tok}_${tag}" "")
        # Predict where checkpoint will land (matches setup_output_dir logic)
        # Format: {tokenizer}[_{aux}{lam}]_{tag}
        local dir_name="${tok}"
        [[ "$aux" != "none" ]] && dir_name+="_${aux}${lam}"
        [[ -n "$tag" ]] && dir_name+="_${tag}"
        ckpt="$CKPT_BASE/$dir_name/full.pth"
    fi

    # Stage 2: probe
    if has_stage probe && [[ "$tok" != "bin" ]]; then
        for mode in native tokid latent; do
            local script; script=$(build_probe "$display" "$tok" "$ckpt" "$mode")
            submit "$script" "probe_${mode}_${display}" "$tok_jid" >/dev/null
        done
    fi

    # Stage 3: policy
    if has_stage policy; then
        local script; script=$(build_policy "$display" "$tok" "$ckpt")
        submit "$script" "pol_${display}" "$tok_jid" >/dev/null
    fi
}

# ── Main: generate conditions and run ────────────────────────────────

echo "Stages: ${STAGES[*]}"
echo "Dry run: $DRY_RUN"

# Mode 1: Explicit TAG/CKPT arrays (existing checkpoints)
if [[ ${#TAGS[@]} -gt 0 && ${#CKPT[@]} -gt 0 ]]; then
    echo "Conditions: ${#TAGS[@]} (explicit)"
    for i in "${!TAGS[@]}"; do
        for tok in "${TOKENIZER[@]}"; do
            for aux_spec in "${AUX_HEAD[@]}"; do
                parse_aux "$aux_spec"
                run_condition "${TAGS[$i]}" "$tok" "${CKPT[$i]}" "$aux_name" "$aux_lam"
            done
        done
    done

# Mode 2: TOK_SET sweep (train from scratch, cartesian product)
elif [[ ${#TOK_SET[@]} -gt 0 ]]; then
    n=$(( ${#TOKENIZER[@]} * ${#AUX_HEAD[@]} * ${#TOK_SET[@]} ))
    echo "Conditions: $n (${#TOKENIZER[@]} tok x ${#AUX_HEAD[@]} aux x ${#TOK_SET[@]} config)"
    for tok in "${TOKENIZER[@]}"; do
        for aux_spec in "${AUX_HEAD[@]}"; do
            parse_aux "$aux_spec"
            for tok_set in "${TOK_SET[@]}"; do
                # Derive tag from KEY=VAL values (no tokenizer prefix — setup_output_dir adds it)
                tag=""
                for kv in $tok_set; do
                    val="$(echo "$kv" | cut -d= -f2 | tr -d '[],')"
                    [[ -n "$tag" ]] && tag+="_"
                    tag+="$val"
                done
                run_condition "$tag" "$tok" "-" "$aux_name" "$aux_lam" "$tok_set"
            done
        done
    done

# Mode 3: Simple tokenizer x aux sweep (no config overrides)
else
    n=$(( ${#TOKENIZER[@]} * ${#AUX_HEAD[@]} ))
    echo "Conditions: $n (${#TOKENIZER[@]} tok x ${#AUX_HEAD[@]} aux)"
    for tok in "${TOKENIZER[@]}"; do
        for aux_spec in "${AUX_HEAD[@]}"; do
            parse_aux "$aux_spec"
            run_condition "" "$tok" "-" "$aux_name" "$aux_lam"
        done
    done
fi

if ! $DRY_RUN; then
    echo ""
    echo "Monitor: squeue -u \$(whoami)"
fi
