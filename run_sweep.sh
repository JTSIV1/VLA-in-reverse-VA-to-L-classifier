#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Experiment sweep.  Edit the settings below, then run:
#   bash run_sweep.sh                        # full pipeline (train → probe → policy → eval)
#   bash run_sweep.sh --verb-probe-only       # verb probes only on existing tokenizers
#   bash run_sweep.sh --teacher-force-only    # teacher-forced L1 eval on existing policies
#   bash run_sweep.sh --dry-run               # print sbatch scripts without submitting
#   bash run_sweep.sh --force                 # retrain even if checkpoints exist
#
# The pipeline has four stages per condition:
#
#   1. train_tokenizer  — Train an action tokenizer from scratch.
#   2. verb_probe       — Evaluate tokenizer quality via verb classification
#                         on (a) raw actions, (b) token IDs, (c) latents.
#   3. train_policy     — Fine-tune a VLA policy (MiniVLA / OpenVLA).
#   4. eval_policy      — Evaluate policy via CALVIN rollout (SR1–SR5).
#
# If a checkpoint already exists for a stage,
# that stage is skipped and the existing checkpoint is passed downstream.
# Use --force to retrain everything from scratch.
#
# Sweep grid = TOK_SET x AUX_HEAD (cartesian product).
# ═══════════════════════════════════════════════════════════════════════

# ── Dataset (calvin or bridge) ──────────────────────────────────────
# Switches sweep name, data paths, and which pipeline stages run.
# Bridge runs tokenizer training only (no probe / policy / eval).x
DATASET="bridge"

# ── Sweep name (creates checkpoints/<SWEEP_NAME>/{tokenizers,policy})
SWEEP_NAME="${DATASET}_sweep"

# ── Sweep grid ─────────────────────────────────────────────────────
# Each entry: "tokenizer_type key=val key=val ..."
# First word = tokenizer family (quest/vq_bet/oat/fast/bin).
# Remaining key=val pairs are passed to train_tokenizer.py --set.
# AUX_HEAD is crossed with every entry (cartesian product).
#
# Directory naming is auto-derived: {tokenizer}_{val1}_{val2}_...[_{aux}{lambda}]
#   e.g. "vq_bet chunk_size=5 num_codes=16 vq_groups=4" → vq_bet_5_16_4
#        + verb:0.1 → vq_bet_5_16_4_verb0.1
#
# Special per-entry keys (consumed by the script, NOT passed to --set):
#   _epochs=N  _batch_size=N  _lr=X   — override TOK_EPOCHS/BATCH_SIZE/LR
#
# Available hyperparameters:
#   quest:  horizon, fsq_levels, downsample_factor, vq_type
#   vq_bet: chunk_size, num_codes, vq_groups, latent_dim, hidden_dim, num_mlp_layers
#   oat:    horizon, fsq_levels, num_registers
#   fast:   fast_vocab_size, fast_scale
#   bin:    (none — no tokenizer training needed)

if [[ "$DATASET" == "bridge" ]]; then
TOK_SET=(
  # ── OAT ─────────────────────────────────────────────────────────
  "oat horizon=16 fsq_levels=[8,5,5]     num_registers=4 _epochs=300 _batch_size=256 _lr=5e-5"   # h16_r4_v200
#   "oat horizon=32 fsq_levels=[5,5,5]     num_registers=8 _epochs=300 _batch_size=256 _lr=5e-5"   # h32_r8_v125
  "oat horizon=32 fsq_levels=[8,8,8]     num_registers=8 _epochs=300 _batch_size=256 _lr=5e-5"   # h32_r8_v512
  "oat horizon=16 fsq_levels=[8,5,5,5]   num_registers=4 _epochs=300 _batch_size=256 _lr=5e-5"   # h16_r4_v1000
#   "oat horizon=32 fsq_levels=[8,8,8,8]   num_registers=8 _epochs=300 _batch_size=256 _lr=5e-5"   # h32_r8_v4096

  # ── QueST ───────────────────────────────────────────────────────
  # "quest horizon=16 fsq_levels=[8,5,5]   downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h16_ds4_v200
#   "quest horizon=32 fsq_levels=[5,5,5]   downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h32_ds4_v125
  # "quest horizon=32 fsq_levels=[8,8,8]   downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h32_ds4_v512
  # "quest horizon=16 fsq_levels=[8,5,5,5] downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h16_ds4_v1000
#   "quest horizon=32 fsq_levels=[8,8,8,8] downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h32_ds4_v4096

  # ── QueST + VQ (replace FSQ with learned VQ codebook) ────────
  # "quest horizon=16 vq_type=vq vq_codebook_size=200 downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"   # h16_vq200_ds4
  # "quest horizon=32 vq_type=vq vq_codebook_size=512 downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"   # h32_vq512_ds4
  # "quest horizon=16 vq_type=vq vq_codebook_size=1000 downsample_factor=4 _epochs=300 _batch_size=128 _lr=1e-4"  # h16_vq1000_ds4

  # ── VQ-BeT ─────────────────────────────────────────────────────
  "vq_bet chunk_size=5  num_codes=16 vq_groups=2 latent_dim=256 _epochs=200 _batch_size=256 _lr=1e-4"  # c5_e16_g2_l256
  "vq_bet chunk_size=5  num_codes=16 vq_groups=2 latent_dim=512 _epochs=200 _batch_size=256 _lr=1e-4"  # c5_e16_g2_l512
  "vq_bet chunk_size=10 num_codes=16 vq_groups=2 latent_dim=256 _epochs=200 _batch_size=256 _lr=1e-4"  # c10_e16_g2_l256
)

AUX_HEAD=(none "verb:0.1" "clip:0.1" "verb:0.1:pfsq" "clip:0.1:pfsq")

else
TOK_SET=(
  # ── VQ-BeT (MLP encoder → ResidualVQ → MLP decoder) ──────────────
  "vq_bet chunk_size=5  num_codes=16 vq_groups=2"                       # → vq_bet_5_16_2
  "vq_bet chunk_size=5  num_codes=16 vq_groups=4"                       # → vq_bet_5_16_4
  "vq_bet chunk_size=5  num_codes=64 vq_groups=2"                       # → vq_bet_5_64_2
  "vq_bet chunk_size=10 num_codes=16 vq_groups=2"                       # → vq_bet_10_16_2
  "vq_bet chunk_size=10 num_codes=16 vq_groups=4"                       # → vq_bet_10_16_4
  "vq_bet chunk_size=10 num_codes=64 vq_groups=2"                       # → vq_bet_10_64_2

  # ── QueST (causal conv + Transformer + FSQ) ───────────────────────
#   "quest horizon=16 fsq_levels=[4,4,4,4] downsample_factor=2"           # → quest_16_4444_2
#   "quest horizon=16 fsq_levels=[4,4,4,4] downsample_factor=4"           # → quest_16_4444_4
#   "quest horizon=32 fsq_levels=[8,5,5,5] downsample_factor=4"           # → quest_32_8555_4
#   "quest horizon=32 fsq_levels=[4,4,4,4] downsample_factor=4"           # → quest_32_4444_4
#   "quest horizon=32 fsq_levels=[4,4,4,4] downsample_factor=8"           # → quest_32_4444_8
#   "quest horizon=32 fsq_levels=[4,4,4]   downsample_factor=4"           # → quest_32_444_4

#   # ── OAT (register encoder + FSQ) ─────────────────────────────────
#   "oat horizon=32 fsq_levels=[8,5,5,5] num_registers=8"                 # → oat_32_8555_8
#   "oat horizon=32 fsq_levels=[4,4,4,4] num_registers=8"                 # → oat_32_4444_8
#   "oat horizon=32 fsq_levels=[4,4,4,4] num_registers=4"                 # → oat_32_4444_4
#   "oat horizon=32 fsq_levels=[4,4,4]   num_registers=4"                 # → oat_32_444_4
#   "oat horizon=16 fsq_levels=[4,4,4,4] num_registers=4"                 # → oat_16_4444_4
#   "oat horizon=16 fsq_levels=[4,4,4,4] num_registers=8"                 # → oat_16_4444_8

  # ── Bin baseline (per-dim uniform binning, no tokenizer training) ──
#   "bin"
)

AUX_HEAD=(none "verb:0.1" "clip:0.1")
fi

# ── Fixed settings ───────────────────────────────────────────────────
TOK_EPOCHS=200
TOK_BATCH_SIZE=32
TOK_LR=1e-4
PROBE_EPOCHS=100
PROBE_BATCH_SIZE=64
PROBE_D_MODEL=128
POLICY_MODEL=minivla                # minivla | openvla
POLICY_BATCH_SIZE=8
POLICY_LR=5e-4
POLICY_MAX_STEPS=50000
EVAL_NUM_SEQUENCES=1000
EVAL_DATASET_PATH="/data/user_data/yashagar/task_D_D"

# ── Bridge-specific paths ───────────────────────────────────────────
BRIDGE_SHARD_DIR="/data/user_data/wenjiel2/datasets/bridge_actions"
BRIDGE_CSV="data/bridge_episodes_filtered.csv"

# ── SLURM ────────────────────────────────────────────────────────────
PARTITION=general
# Blackwell nodes (sm_120) incompatible with current PyTorch (max sm_90)
EXCLUDE_NODES="babel-l5-[16,20],babel-m9-[16,20],babel-n9-20"
TOK_TIME="2:00:00"
PROBE_TIME="4:00:00"
POLICY_TIME="12:00:00"
EVAL_TIME="24:00:00"
TF_TIME="4:00:00"

# ═══════════════════════════════════════════════════════════════════════
#                    Nothing to edit below this line
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail
TOK_SET=("${TOK_SET[@]+"${TOK_SET[@]}"}")
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="/data/user_data/wenjiel2/Code/openvla-mini"
if [[ "$DATASET" == "bridge" ]]; then
    RLDS_DIR="/data/user_data/wenjiel2/datasets/bridge_rlds"
else
    RLDS_DIR="/data/user_data/wenjiel2/datasets/calvin_rlds"
fi
LOG_DIR="$PROJECT_DIR/logs"
SWEEP_DIR="$PROJECT_DIR/checkpoints/$SWEEP_NAME"
TOK_DIR="$SWEEP_DIR/tokenizers"
POLICY_DIR="$SWEEP_DIR/policy"
BASE_VLM="/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"
mkdir -p "$LOG_DIR" "$TOK_DIR" "$POLICY_DIR"

DRY_RUN=false
FORCE=false
VERB_PROBE_ONLY=false
TF_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)              DRY_RUN=true ;;
        --force)                FORCE=true ;;
        --verb-probe-only)      VERB_PROBE_ONLY=true ;;
        --teacher-force-only)   TF_ONLY=true ;;
    esac
done

# Parse "verb:0.1" -> aux_name=verb aux_lam=0.1 aux_target=latent
# Parse "verb:0.1:pfsq" -> aux_name=verb aux_lam=0.1 aux_target=post_fsq
# Parse "none" -> aux_name=none aux_lam=0 aux_target=latent
parse_aux() {
    local spec="$1"
    aux_target="latent"
    if [[ "$spec" == "none" ]]; then
        aux_name="none"; aux_lam="0"
    else
        # Split on ':'
        IFS=':' read -ra parts <<< "$spec"
        aux_name="${parts[0]}"
        aux_lam="${parts[1]}"
        if [[ "${parts[2]:-}" == "pfsq" ]]; then
            aux_target="post_fsq"
        fi
    fi
}

# Check if a checkpoint file exists (returns false in --force mode)
ckpt_exists() {
    ! $FORCE && [[ -f "$1" ]]
}

# Infer tokenizer type from directory name prefix
infer_tokenizer_type() {
    local name="$1"
    case "$name" in
        vq_bet_*) echo "vq_bet" ;;
        quest_*)  echo "quest" ;;
        oat_*)    echo "oat" ;;
        fast_*)   echo "fast" ;;
        *)        echo "unknown" ;;
    esac
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
#SBATCH --exclude=$EXCLUDE_NODES
#SBATCH -o $LOG_DIR/%j_${name}.out
#SBATCH -e $LOG_DIR/%j_${name}.err
EOF
}

preamble() {
    cat <<EOF

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"

EOF
}

# ── Stage builders ───────────────────────────────────────────────────

build_tokenizer() {
    local display="$1" tok="$2" aux="$3" lam="$4" tok_set="${5:-}" hparam_tag="${6:-}" target="${7:-latent}"
    local name="tok_${display}"

    # Extract per-entry overrides (_epochs, _batch_size, _lr) from tok_set
    local ep="$TOK_EPOCHS" bs="$TOK_BATCH_SIZE" lr="$TOK_LR"
    local filtered_set=""
    for kv in $tok_set; do
        case "$kv" in
            _epochs=*)      ep="${kv#*=}" ;;
            _batch_size=*)  bs="${kv#*=}" ;;
            _lr=*)          lr="${kv#*=}" ;;
            *)              filtered_set+=" $kv" ;;
        esac
    done
    filtered_set="${filtered_set# }"  # trim leading space

    local cmd="python -u tokenization/train_tokenizer.py"
    cmd+=" --tokenizer $tok --dataset $DATASET"
    cmd+=" --epochs $ep --batch_size $bs --lr $lr"
    cmd+=" --save_dir $TOK_DIR --min_class_count 30 --max_chunks 8"
    # hparam_tag has only hyperparams (no aux suffix); setup_output_dir appends aux
    cmd+=" --tag $hparam_tag"
    [[ "$DATASET" == "bridge" ]] && cmd+=" --shard_dir $BRIDGE_SHARD_DIR --bridge_csv $BRIDGE_CSV"
    [[ "$aux" != "none" ]] && cmd+=" --aux_head $aux --aux_lambda $lam"
    [[ "$target" != "latent" ]] && cmd+=" --aux_target $target"
    [[ -n "$filtered_set" ]] && cmd+=" --set $filtered_set"

    echo "$(sbatch_header "$name" "$TOK_TIME")$(preamble)
$cmd"
}

build_probe() {
    local tag="$1" tok="$2" ckpt="$3" mode="$4"
    local name="probe_${mode}_${tag}"
    local save="$TOK_DIR/$tag/probe_${mode}.pth"
    local cmd="python -u verb_probe/train_verb_probe.py"

    if [[ "$mode" == "native" ]]; then
        cmd+=" --action_rep native"
    elif [[ "$mode" == "tokid" ]]; then
        cmd+=" --action_rep $tok --tokenizer_type $tok --tokenizer_ckpt $ckpt"
    elif [[ "$mode" == "latent" ]]; then
        cmd+=" --action_rep latent --tokenizer_type $tok --tokenizer_ckpt $ckpt"
    elif [[ "$mode" == "vla_embed" ]]; then
        local policy_dir="$POLICY_DIR/${POLICY_MODEL}_${tag}"
        cmd+=" --action_rep vla_embed --tokenizer_type $tok --tokenizer_ckpt $ckpt"
        cmd+=" --policy_dir $policy_dir"
    fi

    cmd+=" --modality action_only --dataset $DATASET"
    cmd+=" --epochs $PROBE_EPOCHS --batch_size $PROBE_BATCH_SIZE"
    cmd+=" --d_model $PROBE_D_MODEL"
    cmd+=" --min_class_count 30 --weighted_loss"
    cmd+=" --save_path $save"
    [[ "$DATASET" == "bridge" ]] && cmd+=" --shard_dir $BRIDGE_SHARD_DIR --bridge_csv $BRIDGE_CSV"

    echo "$(sbatch_header "$name" "$PROBE_TIME")$(preamble)
$cmd"
}

build_policy() {
    local tag="$1" tok="$2" ckpt="$3"
    local name="pol_${tag}"
    local run_dir="$POLICY_DIR/${POLICY_MODEL}_${tag}"
    local vla_config
    if [[ "$DATASET" == "bridge" ]]; then
        vla_config="prism-qwen25-dinosiglip-224px+0_5b+mx-bridge"
    else
        vla_config="prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin"
    fi

    local header; header=$(sbatch_header "$name" "$POLICY_TIME" "64G")
    local pre
    pre=$(cat <<EOF

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
export WANDB_MODE=offline
cd "$OPENVLA_DIR"

EOF
)

    local cmd
    if [[ "$POLICY_MODEL" == "minivla" ]]; then
        # train.py creates run_root_dir/run_id/; we want the flat layout
        # checkpoints/<sweep>/policy/minivla_<tag>/. image_aug=True appends
        # "--image_aug" to run_id, so we omit it from run_id and disable the
        # flag (image augmentation is done at the dataset level anyway).
        cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
        cmd+=" vla-scripts/train.py"
        cmd+=" --vla.type $vla_config"
        cmd+=" --vla.base_vlm $BASE_VLM"
        cmd+=" --data_root_dir $RLDS_DIR"
        cmd+=" --run_root_dir $POLICY_DIR"
        cmd+=" --run_id ${POLICY_MODEL}_${tag}"
        cmd+=" --image_aug True"
        cmd+=" --save_interval 5000"
        cmd+=" --vla.expected_world_size 1"
        cmd+=" --vla.global_batch_size 16"
        cmd+=" --vla.per_device_batch_size 16"
        cmd+=" --vla.freeze_vision_backbone True"
        cmd+=" --vla.max_steps $POLICY_MAX_STEPS"
        if [[ "$tok" != "bin" && -n "$ckpt" ]]; then
            cmd+=" --vla.action_tokenizer 'sweep:${tok}:${ckpt}'"
        fi
    else
        cmd="torchrun --standalone --nnodes 1 --nproc-per-node 1"
        cmd+=" vla-scripts/finetune.py"
        cmd+=" --vla_path openvla/openvla-7b"
        cmd+=" --data_root_dir $RLDS_DIR"
        cmd+=" --dataset_name ${DATASET}_dataset"
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

build_eval() {
    local tag="$1"
    local name="eval_${tag}"
    local cmd="python -u policy/eval_policy.py"
    cmd+=" --condition $tag --mode rollout"
    cmd+=" --num_sequences $EVAL_NUM_SEQUENCES"
    cmd+=" --output_dir $SWEEP_DIR/results/rollout"

    echo "$(sbatch_header "$name" "$EVAL_TIME" "64G")$(preamble)
$cmd"
}

build_teacher_force() {
    local tag="$1"
    local name="tf_${tag}"
    local cmd="python -u policy/eval_policy.py"
    cmd+=" --condition $tag --mode teacher_force"
    cmd+=" --output_dir $SWEEP_DIR/results/teacher_force"

    local header; header=$(sbatch_header "$name" "$TF_TIME" "64G")
    local pre
    pre=$(cat <<EOF

export PATH="/data/user_data/wenjiel2/miniconda3/envs/mmml/bin:\$PATH"
export CONDA_PREFIX="/data/user_data/wenjiel2/miniconda3/envs/mmml"
export PRISMATIC_DATA_ROOT="$RLDS_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:\${PYTHONPATH:-}"

EOF
)
    echo "${header}${pre}
${cmd}"
}

# ── Run one condition (skip stages with existing checkpoints) ────────

run_condition() {
    local tok="$1" aux_name="$2" aux_lam="$3" tok_set="${4:-}"

    # Bin has no tokenizer → skip aux variants (only run bin once with aux=none)
    if [[ "$tok" == "bin" && "$aux_name" != "none" ]]; then
        return
    fi

    # post_fsq aux target only applies to FSQ-based tokenizers (OAT/QueST with fsq)
    if [[ "$aux_target" == "post_fsq" ]]; then
        # Skip for VQ-BeT (uses ResidualVQ, no FSQ)
        if [[ "$tok" == "vq_bet" ]]; then return; fi
        # Skip for QueST with vq_type=vq (no FSQ codes)
        if [[ "$tok_set" == *"vq_type=vq"* ]]; then return; fi
    fi

    # Build tag from key=val values (e.g. "chunk_size=5 num_codes=16 vq_groups=4" → "5_16_4")
    # Skip _-prefixed keys (per-entry overrides like _epochs, _batch_size, _lr)
    local tag=""
    for kv in $tok_set; do
        [[ "$kv" == _* ]] && continue
        val="$(echo "$kv" | cut -d= -f2 | tr -d '[],')"
        [[ -n "$tag" ]] && tag+="_"
        tag+="$val"
    done

    # Display name: {tok}_{tag}[_{aux}{lam}[_pfsq]]
    # Matches setup_output_dir() in train_tokenizer.py
    local display="${tok}"
    [[ -n "$tag" ]] && display+="_${tag}"
    if [[ "$aux_name" != "none" ]]; then
        display+="_${aux_name}${aux_lam}"
        [[ "$aux_target" == "post_fsq" ]] && display+="_pfsq"
    fi

    echo ""
    echo "  [$display]"

    local tok_ckpt="$TOK_DIR/$display/full.pth"
    local policy_ckpt_dir="$POLICY_DIR/${POLICY_MODEL}_${display}"

    # ── Stage 1: train tokenizer (skip for bin) ────────────────────
    local tok_jid=""
    if [[ "$tok" == "bin" ]]; then
        echo "    tokenizer: SKIP (bin baseline, no tokenizer)"
    elif ckpt_exists "$tok_ckpt"; then
        echo "    tokenizer: SKIP (checkpoint exists)"
    else
        local script; script=$(build_tokenizer "$display" "$tok" "$aux_name" "$aux_lam" "$tok_set" "$tag" "$aux_target")
        tok_jid=$(submit "$script" "tok_${display}" "")
    fi

    # ── Stage 2: verb probe (skip for bin) ────────────────────────
    if [[ "$tok" == "bin" ]]; then
        echo "    probe: SKIP (bin baseline, no tokenizer)"
    else
        for mode in native tokid latent; do
            local probe_ckpt="$TOK_DIR/$display/probe_${mode}.pth"
            if ckpt_exists "$probe_ckpt"; then
                echo "    probe ($mode): SKIP (checkpoint exists)"
            else
                local script; script=$(build_probe "$display" "$tok" "$tok_ckpt" "$mode")
                submit "$script" "probe_${mode}_${display}" "$tok_jid" >/dev/null
            fi
        done
    fi

    # ── Stage 3: train policy ─────────────────────────────────────
    local pol_jid=""
    if [[ -d "$policy_ckpt_dir" ]] && ! $FORCE; then
        echo "    policy: SKIP (directory exists)"
    else
        local script; script=$(build_policy "$display" "$tok" "$tok_ckpt")
        pol_jid=$(submit "$script" "pol_${display}" "$tok_jid")
    fi

    # ── Stage 3b: vla_embed verb probe (depends on policy) ────────
    if [[ "$tok" != "bin" ]]; then
        local vla_probe_ckpt="$TOK_DIR/$display/probe_vla_embed.pth"
        if ckpt_exists "$vla_probe_ckpt" && ! $FORCE; then
            echo "    probe (vla_embed): SKIP (checkpoint exists)"
        else
            local dep="${pol_jid:-}"
            local script; script=$(build_probe "$display" "$tok" "$tok_ckpt" "vla_embed")
            submit "$script" "probe_vla_embed_${display}" "$dep" >/dev/null
        fi
    fi

    # ── Stage 4: eval policy (CALVIN only — bridge has no rollout eval) ──
    if [[ "$DATASET" != "bridge" ]]; then
    local eval_results="$policy_ckpt_dir/eval_results.json"
    if ckpt_exists "$eval_results"; then
        echo "    eval: SKIP (results exist)"
    else
        local dep="${pol_jid:-$tok_jid}"
        local script; script=$(build_eval "$display")
        submit "$script" "eval_${display}" "$dep" >/dev/null
    fi
    fi  # end DATASET != bridge (eval only)

}

# ── Helpers for resolving display names from TOK_SET entries ──────────

# Parse a TOK_SET entry into tok, tag, display variables
parse_entry() {
    local entry="$1" aux_name="$2" aux_lam="$3" target="${4:-latent}"
    _tok="${entry%% *}"
    local tok_set="${entry#* }"
    [[ "$tok_set" == "$_tok" ]] && tok_set=""

    _tag=""
    for kv in $tok_set; do
        [[ "$kv" == _* ]] && continue
        val="$(echo "$kv" | cut -d= -f2 | tr -d '[],')"
        [[ -n "$_tag" ]] && _tag+="_"
        _tag+="$val"
    done

    _display="${_tok}"
    if [[ -n "$_tag" ]]; then _display+="_${_tag}"; fi
    if [[ "$aux_name" != "none" ]]; then
        _display+="_${aux_name}${aux_lam}"
        [[ "$target" == "post_fsq" ]] && _display+="_pfsq"
    fi
}

# Resolve tokenizer checkpoint path (full.pth or tokenizer_weights.pth)
resolve_tok_ckpt() {
    local display="$1"
    local ckpt="$TOK_DIR/$display/full.pth"
    if [[ -f "$ckpt" ]]; then echo "$ckpt"; return; fi
    ckpt="$TOK_DIR/$display/tokenizer_weights.pth"
    if [[ -f "$ckpt" ]]; then echo "$ckpt"; return; fi
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────

echo "Sweep: $SWEEP_NAME"
echo "Dry run: $DRY_RUN"
echo "Force retrain: $FORCE"

if $TF_ONLY; then
    # ── Teacher-force only mode ───────────────────────────────────────
    echo "Mode: teacher_force_only"
    submitted=0
    skipped=0
    for entry in "${TOK_SET[@]}"; do
        for aux_spec in "${AUX_HEAD[@]}"; do
            parse_aux "$aux_spec"
            parse_entry "$entry" "$aux_name" "$aux_lam" "$aux_target"
            if [[ "$_tok" == "bin" && "$aux_name" != "none" ]]; then continue; fi

            local_results="$SWEEP_DIR/results/teacher_force/$_display/teacher_force_metrics.json"
            policy_dir="$POLICY_DIR/${POLICY_MODEL}_${_display}"
            if [[ ! -d "$policy_dir" ]]; then
                echo "  [$_display] SKIP (no policy dir)"
                skipped=$((skipped + 1))
                continue
            fi
            if ckpt_exists "$local_results"; then
                echo "  [$_display] SKIP (results exist)"
                skipped=$((skipped + 1))
                continue
            fi

            script=$(build_teacher_force "$_display")
            submit "$script" "tf_${_display}" "" >/dev/null
            submitted=$((submitted + 1))
            echo "  [$_display] submitted"
        done
    done

    echo ""
    echo "Submitted $submitted teacher-force jobs, skipped $skipped configs."

elif $VERB_PROBE_ONLY; then
    # ── Verb probe only mode ─────────────────────────────────────────
    # 1. Validate all tokenizer checkpoints exist
    # 2. Submit one native probe (shared baseline, tokenizer-independent)
    # 3. For each tokenizer: submit latent + tokid probes
    echo "Mode: verb_probe_only"

    # Submit native probe once (save under sweep dir)
    native_save="$SWEEP_DIR/probe_native.pth"
    if ckpt_exists "$native_save"; then
        echo "  native probe: SKIP (checkpoint exists)"
    else
        echo "  [native]"
        cmd="python -u verb_probe/train_verb_probe.py"
        cmd+=" --action_rep native"
        cmd+=" --modality action_only --dataset $DATASET"
        cmd+=" --epochs $PROBE_EPOCHS --batch_size $PROBE_BATCH_SIZE"
        cmd+=" --d_model $PROBE_D_MODEL"
        cmd+=" --min_class_count 30 --weighted_loss"
        cmd+=" --save_path $native_save"
        [[ "$DATASET" == "bridge" ]] && cmd+=" --shard_dir $BRIDGE_SHARD_DIR --bridge_csv $BRIDGE_CSV"
        script="$(sbatch_header "probe_native" "$PROBE_TIME")$(preamble)
$cmd"
        submit "$script" "probe_native" "" >/dev/null
    fi

    # Submit latent + tokid probes per tokenizer (skip missing checkpoints)
    submitted=0
    skipped=0
    for entry in "${TOK_SET[@]}"; do
        for aux_spec in "${AUX_HEAD[@]}"; do
            parse_aux "$aux_spec"
            parse_entry "$entry" "$aux_name" "$aux_lam" "$aux_target"
            if [[ "$_tok" == "bin" && "$aux_name" != "none" ]]; then continue; fi
            if [[ "$_tok" == "bin" ]]; then continue; fi

            ckpt=$(resolve_tok_ckpt "$_display")
            if [[ -z "$ckpt" ]]; then
                echo "  [$_display] SKIP (no checkpoint)"
                skipped=$((skipped + 1))
                continue
            fi

            echo ""
            echo "  [$_display]"

            if [[ "$DATASET" == "bridge" ]]; then
                probe_modes=(latent tokid)
            else
                probe_modes=(latent tokid vla_embed)
            fi
            for mode in "${probe_modes[@]}"; do
                local_save="$TOK_DIR/$_display/probe_${mode}.pth"
                if ckpt_exists "$local_save"; then
                    echo "    probe ($mode): SKIP (checkpoint exists)"
                else
                    # vla_embed requires a trained policy
                    if [[ "$mode" == "vla_embed" ]]; then
                        local policy_dir="$POLICY_DIR/${POLICY_MODEL}_${_display}"
                        if [[ ! -d "$policy_dir/checkpoints" ]]; then
                            echo "    probe ($mode): SKIP (no policy)"
                            continue
                        fi
                    fi
                    script=$(build_probe "$_display" "$_tok" "$ckpt" "$mode")
                    submit "$script" "probe_${mode}_${_display}" "" >/dev/null
                    submitted=$((submitted + 1))
                fi
            done
        done
    done

    echo ""
    echo "Submitted $submitted probe jobs, skipped $skipped configs (no checkpoint)."

else
    # ── Full pipeline mode ───────────────────────────────────────────
    n=$(( ${#TOK_SET[@]} * ${#AUX_HEAD[@]} ))
    echo "Conditions: $n (${#TOK_SET[@]} configs x ${#AUX_HEAD[@]} aux)"

    for entry in "${TOK_SET[@]}"; do
        # First word = tokenizer type, rest = key=val overrides
        tok="${entry%% *}"
        tok_set="${entry#* }"
        [[ "$tok_set" == "$tok" ]] && tok_set=""  # no overrides (e.g. "bin")

        for aux_spec in "${AUX_HEAD[@]}"; do
            parse_aux "$aux_spec"
            run_condition "$tok" "$aux_name" "$aux_lam" "$tok_set"
        done
    done
fi

if ! $DRY_RUN; then
    echo ""
    echo "Monitor: squeue -u \$(whoami)"
fi
