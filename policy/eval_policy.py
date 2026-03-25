#!/usr/bin/env python3
"""Unified VLA policy evaluation launcher.

Submits SLURM jobs for evaluating fine-tuned OpenVLA / MiniVLA models on CALVIN.

Supports two model families:
    openvla   — OpenVLA 7B + LoRA (old, runs/openvla/)
    scratch   — MiniVLA from scratch (Qwen2.5-0.5B, runs/calvind_scratch/)

Evaluation modes:
    nll       — Teacher-forcing NLL + action accuracy (GPU)
    real_l1   — Decoded pred tokens vs raw GT continuous actions (GPU)
    verb      — Verb decodability probe via tokenizer round-trip (GPU)
    rollout   — CALVIN rollout evaluation, 1000 sequences (GPU, ~18h)
    attention — Attention analysis: action→verb token attention (GPU, ~6h)

Usage:
    # Scratch MiniVLA: NLL for bin baseline
    python policy/eval_policy.py --family scratch --mode nll --condition bin_baseline

    # Scratch MiniVLA: real_l1 for a VQ-BeT condition
    python policy/eval_policy.py --family scratch --mode real_l1 --condition vb_c5e16g4

    # Scratch MiniVLA: all conditions
    python policy/eval_policy.py --family scratch --mode nll --condition all

    # Old OpenVLA: rollout for bin
    python policy/eval_policy.py --family openvla --mode rollout --condition bin

    # Dry run
    python policy/eval_policy.py --family scratch --mode nll --condition bin_baseline --dry_run
"""

import argparse
import os
import subprocess
import sys
import tempfile

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENVLA_DIR = "/data/user_data/wenjiel2/Code/openvla-mini"
CALVIN_DIR = "/data/user_data/wenjiel2/Code/calvin"
DATA_DIR = "/data/user_data/wenjiel2/datasets/calvin_rlds"
DATASET_PATH = "/data/user_data/yashagar/task_D_D"
CKPT_SWEEP_BASE = os.path.join(PROJECT_DIR, "checkpoints", "calvind_hp_sweep")

VERB_CLF_CKPT = os.path.join(PROJECT_DIR, "checkpoints",
                              "ao_native_sparse_weighted_j6457852_best.pth")

# ── Old OpenVLA (7B + LoRA) conditions ──────────────────────────────────────

VQVLA_TOKENIZER_CKPTS = {
    "vqvla_vanilla": os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_vanilla"),
    "vqvla_verb":    os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.5"),
    "vqvla_verb01":  os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.1"),
}

OPENVLA_CONDITIONS = ["bin", "vqvla_vanilla", "vqvla_verb", "vqvla_verb01"]
OPENVLA_RUN_TAGS = {
    "bin":            "calvin_bin",
    "vqvla_vanilla":  "calvin_vq_vanilla",
    "vqvla_verb":     "calvin_vq_verb",
    "vqvla_verb01":   "calvin_vq_verb01",
}
OPENVLA_CKPT_TEMPLATE = (
    "openvla-7b+{dataset}+b16+lr-0.0005+lora-r32+dropout-0.0"
    "--{run_tag}--image_aug"
)

# ── New MiniVLA from-scratch conditions ─────────────────────────────────────

# Maps condition tag → (sweep_tokenizer_type, sweep_checkpoint_path)
# None values mean bin tokenizer (no sweep checkpoint needed)
SCRATCH_CONDITIONS = {
    "bin_baseline":         (None, None),
    "vb_c5e16g4":           ("vq_bet", os.path.join(CKPT_SWEEP_BASE, "vq_bet_c5_e16_g4", "full.pth")),
    "vb_c5e64g2":           ("vq_bet", os.path.join(CKPT_SWEEP_BASE, "vq_bet_c5_e64_g2", "full.pth")),
    "vb_c10e16g4":          ("vq_bet", os.path.join(CKPT_SWEEP_BASE, "vq_bet_c10_e16_g4", "full.pth")),
    "vb_c5e16g4_verb01":    ("vq_bet", os.path.join(CKPT_SWEEP_BASE, "vq_bet_verb0.1_c5e16g4_verb01", "full.pth")),
    "vb_c5e16g4_clip01":    ("vq_bet", os.path.join(CKPT_SWEEP_BASE, "vq_bet_clip0.1_c5e16g4_clip01", "full.pth")),
    "quest_h16f256d2":      ("quest",  os.path.join(CKPT_SWEEP_BASE, "quest_h16_f256_d2", "full.pth")),
    "quest_h32f1000d4":     ("quest",  os.path.join(CKPT_SWEEP_BASE, "quest_h32_f1000_d4", "full.pth")),
    "quest_h16f256d4":      ("quest",  os.path.join(CKPT_SWEEP_BASE, "quest_h16_f256_d4", "full.pth")),
    "quest_h16d2_verb01":   ("quest",  os.path.join(CKPT_SWEEP_BASE, "quest_verb0.1_h16d2_verb01", "full.pth")),
    "quest_h16d2_clip01":   ("quest",  os.path.join(CKPT_SWEEP_BASE, "quest_clip0.1_h16d2_clip01", "full.pth")),
}

# MiniVLA checkpoint dir pattern: runs/calvind_scratch/<config_prefix>--<tag>--image_aug
SCRATCH_CKPT_PREFIX = "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin+n0+b16+x7"


def get_scratch_ckpt_dir(condition):
    """Find the MiniVLA checkpoint dir for a scratch condition."""
    run_dir = os.path.join(PROJECT_DIR, "runs", "calvind_scratch")
    expected = f"{SCRATCH_CKPT_PREFIX}--{condition}--image_aug"
    candidate = os.path.join(run_dir, expected)
    if os.path.isdir(candidate):
        return candidate
    # Try batch_size variants (b8, b32)
    import glob
    pattern = os.path.join(run_dir, f"*--{condition}--image_aug")
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]  # most recent
    return candidate  # return expected path even if missing (for dry_run)


def find_best_checkpoint(run_dir):
    """Find the FSDP .pt checkpoint with lowest loss in a run directory."""
    import glob
    import re
    pattern = os.path.join(run_dir, "checkpoints", "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    best_file, best_loss = None, float("inf")
    for f in files:
        m = re.search(r'loss=(\d+\.\d+)', f)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_file = f
    return best_file


def get_openvla_ckpt_dir(condition, dataset):
    """Derive the OpenVLA checkpoint directory."""
    tag = OPENVLA_RUN_TAGS[condition]
    if dataset == "calvin_abcd_dataset":
        tag = tag.replace("calvin_", "calvin_abcd_")
    suffix = OPENVLA_CKPT_TEMPLATE.format(dataset=dataset, run_tag=tag)
    return os.path.join(PROJECT_DIR, "runs", "openvla", suffix)


# ── SLURM configs per mode ───────────────────────────────────────────────────

SLURM_CONFIGS = {
    "nll":       {"gres": "gpu:1", "mem": "48G", "time": "4:00:00",  "cpus": 4},
    "real_l1":   {"gres": "gpu:1", "mem": "48G", "time": "4:00:00",  "cpus": 4},
    "verb":      {"gres": "gpu:1", "mem": "32G", "time": "2:00:00",  "cpus": 4},
    "rollout":   {"gres": "gpu:1", "mem": "64G", "time": "24:00:00", "cpus": 8},
    "attention": {"gres": "gpu:1", "mem": "64G", "time": "8:00:00",  "cpus": 4},
}


# ── Command builders ─────────────────────────────────────────────────────────

def build_scratch_eval_command(mode, condition, args):
    """Build eval command for MiniVLA from-scratch conditions."""
    ckpt_dir = get_scratch_ckpt_dir(condition)
    sweep_type, sweep_ckpt = SCRATCH_CONDITIONS[condition]
    out_dir = f"results/scratch/{condition}"

    sweep_args = ""
    if sweep_type and sweep_ckpt:
        sweep_args = f"--sweep_tokenizer_type {sweep_type} --sweep_checkpoint_path {sweep_ckpt}"

    cond_label = "bin" if condition == "bin_baseline" else condition

    if mode == "nll":
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond_label} --eval_nll "
            f"--checkpoint_dir {ckpt_dir} "
            f"--data_root_dir {DATA_DIR} "
            f"--max_nll_batches {args.max_nll_batches} "
            f"--output_dir {out_dir} "
            f"{sweep_args}"
        )
    elif mode == "real_l1":
        best_ckpt = find_best_checkpoint(ckpt_dir)
        fsdp_arg = f"--fsdp_checkpoint {best_ckpt}" if best_ckpt else ""
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond_label} --eval_real_l1 "
            f"--checkpoint_dir {ckpt_dir} "
            f"--data_root_dir {DATA_DIR} "
            f"--max_nll_batches {args.max_nll_batches} "
            f"--output_dir {out_dir} "
            f"{sweep_args} {fsdp_arg}"
        )
    elif mode == "rollout":
        best_ckpt = find_best_checkpoint(ckpt_dir)
        fsdp_arg = f"--fsdp_checkpoint {best_ckpt}" if best_ckpt else ""
        return (
            f"python -u -m policy.scripts.evaluate_openvla_rollout "
            f"--condition {cond_label} "
            f"--checkpoint_dir {ckpt_dir} "
            f"--dataset_path {DATASET_PATH} "
            f"--output_dir results/rollout_scratch "
            f"--num_sequences {args.num_sequences} "
            f"{sweep_args} {fsdp_arg}"
        )
    elif mode == "verb":
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond_label} --eval_verb_probe "
            f"--verb_classifier_ckpt {VERB_CLF_CKPT} "
            f"--min_class_count 30 "
            f"--output_dir {out_dir} "
            f"{sweep_args}"
        )
    elif mode == "attention":
        return (
            f"python -u -m policy.scripts.analyze_attention "
            f"--condition {cond_label} "
            f"--checkpoint_dir {ckpt_dir} "
            f"--output_dir results/attention_scratch "
            f"--max_examples {args.max_examples} "
            f"{sweep_args}"
        )


def build_openvla_eval_command(mode, condition, dataset, args):
    """Build eval command for old OpenVLA 7B + LoRA conditions."""
    ckpt_dir = get_openvla_ckpt_dir(condition, dataset)
    cond = condition.replace("vqvla_", "vq_")

    extra = ""
    if condition in VQVLA_TOKENIZER_CKPTS:
        extra = f"--vqvla_checkpoint_dir {VQVLA_TOKENIZER_CKPTS[condition]}"

    if mode == "nll":
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond} --eval_nll "
            f"--checkpoint_dir {ckpt_dir} "
            f"--data_root_dir {DATA_DIR} "
            f"--max_nll_batches {args.max_nll_batches} "
            f"--output_dir results/stage3/{cond} "
            f"{extra}"
        )
    elif mode == "real_l1":
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond} --eval_real_l1 "
            f"--checkpoint_dir {ckpt_dir} "
            f"--data_root_dir {DATA_DIR} "
            f"--max_nll_batches {args.max_nll_batches} "
            f"--output_dir results/stage3/{cond} "
            f"{extra}"
        )
    elif mode == "verb":
        return (
            f"python -m policy.scripts.evaluate_openvla "
            f"--condition {cond} --eval_verb_probe "
            f"--verb_classifier_ckpt {VERB_CLF_CKPT} "
            f"--min_class_count 30 "
            f"--output_dir results/stage3/{cond} "
            f"{extra}"
        )
    elif mode == "rollout":
        out_dir = "results/rollout"
        if dataset == "calvin_abcd_dataset":
            out_dir = "results/rollout_abcd"
        return (
            f"python -u -m policy.scripts.evaluate_openvla_rollout "
            f"--condition {cond} "
            f"--checkpoint_dir {ckpt_dir} "
            f"--dataset_path {DATASET_PATH} "
            f"--output_dir {out_dir} "
            f"--num_sequences {args.num_sequences} "
            f"{extra}"
        )
    elif mode == "attention":
        return (
            f"python -u -m policy.scripts.analyze_attention "
            f"--condition {cond} "
            f"--checkpoint_dir {ckpt_dir} "
            f"--output_dir results/attention_analysis "
            f"--max_examples {args.max_examples} "
            f"{extra}"
        )


# ── SLURM submission ──────────────────────────────────────────────────────────

def build_sbatch_script(mode, condition, eval_cmd, args):
    """Generate sbatch script."""
    slurm = SLURM_CONFIGS[mode]
    job_name = f"{mode}_{condition}"

    log_dir = os.path.join(PROJECT_DIR, "logs")

    pythonpath = f"{PROJECT_DIR}:{OPENVLA_DIR}"
    conda_env = "mmml"
    if mode == "rollout":
        pythonpath += f":{CALVIN_DIR}/calvin_models:{CALVIN_DIR}/calvin_env"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition or 'general'}",
        f"#SBATCH --gres={args.gres or slurm['gres']}",
        f"#SBATCH --cpus-per-task={slurm['cpus']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --time={args.time or slurm['time']}",
        f"#SBATCH -o {log_dir}/{job_name}_%j.out",
        f"#SBATCH -e {log_dir}/{job_name}_%j.err",
        "",
        'source $(conda info --base)/etc/profile.d/conda.sh',
        f"conda activate {conda_env}",
        "",
        f'export PYTHONPATH="{pythonpath}:${{PYTHONPATH}}"',
        f'export PRISMATIC_DATA_ROOT="{DATA_DIR}"',
        "export DISPLAY=''",
        "",
        f'pip install -e "{OPENVLA_DIR}" --quiet 2>/dev/null || true',
        "",
        f'cd "{PROJECT_DIR}"',
        "",
        eval_cmd,
    ]
    return "\n".join(lines) + "\n"


def submit_script(script_content, dry_run=False):
    """Submit or print a sbatch script."""
    if dry_run:
        print(script_content)
        print("---")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        tmp = f.name
    try:
        result = subprocess.run(["sbatch", tmp], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  Error: {result.stderr.strip()}", file=sys.stderr)
    finally:
        os.unlink(tmp)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified VLA policy evaluation launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--family", required=True,
                        choices=["openvla", "scratch"],
                        help="Model family: openvla (7B+LoRA) or scratch (MiniVLA)")
    parser.add_argument("--mode", required=True,
                        choices=["nll", "real_l1", "verb", "rollout", "attention"],
                        help="Evaluation mode")
    parser.add_argument("--condition", required=True,
                        help="Condition tag (or 'all' for all conditions in family)")
    parser.add_argument("--dataset", default="calvin_dataset",
                        choices=["calvin_dataset", "calvin_abcd_dataset"])

    # Mode-specific args
    parser.add_argument("--max_nll_batches", type=int, default=500)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--max_examples", type=int, default=300)

    # SLURM overrides
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--time", default=None)

    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Resolve condition list
    if args.family == "scratch":
        all_conditions = list(SCRATCH_CONDITIONS.keys())
    else:
        all_conditions = OPENVLA_CONDITIONS

    if args.condition == "all":
        conditions = all_conditions
    elif args.condition in all_conditions:
        conditions = [args.condition]
    else:
        print(f"Unknown condition: {args.condition}")
        print(f"Options for --family {args.family}: {all_conditions + ['all']}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN ===\n")

    for cond in conditions:
        if args.family == "scratch":
            eval_cmd = build_scratch_eval_command(args.mode, cond, args)
            ckpt_dir = get_scratch_ckpt_dir(cond)
        else:
            eval_cmd = build_openvla_eval_command(args.mode, cond, args.dataset, args)
            ckpt_dir = get_openvla_ckpt_dir(cond, args.dataset)

        # For modes needing a VLA checkpoint, check it exists
        if args.mode in ("nll", "real_l1", "rollout", "attention"):
            if not os.path.isdir(ckpt_dir) and not args.dry_run:
                print(f"  [SKIP] {cond}: checkpoint not found at {ckpt_dir}")
                continue

        script = build_sbatch_script(args.mode, cond, eval_cmd, args)
        submit_script(script, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nMonitor: squeue -u {os.environ.get('USER', 'wenjiel2')}")


if __name__ == "__main__":
    main()
