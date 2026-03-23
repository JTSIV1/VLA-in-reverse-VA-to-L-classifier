#!/usr/bin/env python3
"""Unified VLA policy evaluation launcher.

Submits SLURM jobs for evaluating fine-tuned OpenVLA / MiniVLA models on CALVIN.

Evaluation modes:
    nll       — Teacher-forcing NLL + action accuracy (GPU)
    verb      — Verb decodability probe via tokenizer round-trip (GPU)
    rollout   — CALVIN rollout evaluation, 1000 sequences (GPU, ~18h)
    attention — Attention analysis: action→verb token attention (GPU, ~6h)

Usage:
    # NLL eval for one condition
    python policy/eval_policy.py --mode nll --tokenizer vqvla_verb

    # Verb probe for all conditions
    python policy/eval_policy.py --mode verb --tokenizer all

    # Rollout eval for bin baseline
    python policy/eval_policy.py --mode rollout --tokenizer bin

    # Attention analysis for all conditions
    python policy/eval_policy.py --mode attention --tokenizer all

    # ABCD variant
    python policy/eval_policy.py --mode rollout --tokenizer bin --dataset calvin_abcd_dataset

    # Dry run
    python policy/eval_policy.py --mode nll --tokenizer bin --dry_run
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

# VQ-VLA tokenizer checkpoints
TOKENIZER_CKPTS = {
    "vqvla_vanilla": os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_vanilla"),
    "vqvla_verb":    os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.5"),
    "vqvla_verb01":  os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.1"),
}

VERB_CLF_CKPT = os.path.join(PROJECT_DIR, "checkpoints",
                              "ao_native_sparse_weighted_j6457852_best.pth")

ALL_TOKENIZERS = ["bin", "vqvla_vanilla", "vqvla_verb", "vqvla_verb01"]

# Checkpoint naming convention from finetune.py
CKPT_SUFFIX_TEMPLATE = (
    "openvla-7b+{dataset}+b16+lr-0.0005+lora-r32+dropout-0.0"
    "--{run_tag}--image_aug"
)

# Maps tokenizer name → run_id_note used during training
RUN_TAGS = {
    "bin":            "calvin_bin",
    "vqvla_vanilla":  "calvin_vq_vanilla",
    "vqvla_verb":     "calvin_vq_verb",
    "vqvla_verb01":   "calvin_vq_verb01",
}


def get_openvla_ckpt_dir(tokenizer, dataset):
    """Derive the OpenVLA checkpoint directory from training conventions."""
    tag = RUN_TAGS[tokenizer]
    if dataset == "calvin_abcd_dataset":
        tag = tag.replace("calvin_", "calvin_abcd_")
    suffix = CKPT_SUFFIX_TEMPLATE.format(dataset=dataset, run_tag=tag)
    return os.path.join(PROJECT_DIR, "runs", "openvla", suffix)


def vqvla_extra_args(tokenizer):
    """Return --vqvla_checkpoint_dir arg if tokenizer is VQ-based."""
    if tokenizer in TOKENIZER_CKPTS:
        return f"--vqvla_checkpoint_dir {TOKENIZER_CKPTS[tokenizer]}"
    return ""


# ── SLURM configs per mode ───────────────────────────────────────────────────

SLURM_CONFIGS = {
    "nll":       {"gres": "gpu:1", "mem": "48G", "time": "4:00:00",  "cpus": 4},
    "verb":      {"gres": "gpu:1", "mem": "32G", "time": "2:00:00",  "cpus": 4},
    "rollout":   {"gres": "gpu:1", "mem": "64G", "time": "24:00:00", "cpus": 8},
    "attention": {"gres": "gpu:1", "mem": "64G", "time": "8:00:00",  "cpus": 4},
}


def build_eval_command(mode, tokenizer, dataset, args):
    """Build the python evaluation command."""
    ckpt_dir = get_openvla_ckpt_dir(tokenizer, dataset)
    extra = vqvla_extra_args(tokenizer)
    # Short condition name for output dirs
    cond = tokenizer.replace("vqvla_", "vq_")

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


def build_sbatch_script(mode, tokenizer, dataset, eval_cmd, args):
    """Generate sbatch script."""
    slurm = SLURM_CONFIGS[mode]
    cond = tokenizer.replace("vqvla_", "vq_")
    job_name = f"{mode}_{cond}"
    if dataset == "calvin_abcd_dataset":
        job_name += "_abcd"

    log_dir = os.path.join(PROJECT_DIR, "logs")

    # Rollout needs calvin env + pybullet
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


def main():
    parser = argparse.ArgumentParser(
        description="Unified VLA policy evaluation launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True,
                        choices=["nll", "verb", "rollout", "attention"],
                        help="Evaluation mode")
    parser.add_argument("--tokenizer", required=True,
                        help="Tokenizer condition (or 'all' for all 4)")
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

    # Resolve tokenizer list
    if args.tokenizer == "all":
        tokenizers = ALL_TOKENIZERS
    elif args.tokenizer in ALL_TOKENIZERS:
        tokenizers = [args.tokenizer]
    else:
        print(f"Unknown tokenizer: {args.tokenizer}")
        print(f"Options: {ALL_TOKENIZERS + ['all']}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN ===\n")

    for tok in tokenizers:
        eval_cmd = build_eval_command(args.mode, tok, args.dataset, args)
        script = build_sbatch_script(args.mode, tok, args.dataset, eval_cmd, args)

        # For NLL/rollout, check checkpoint exists
        if args.mode in ("nll", "rollout", "attention"):
            ckpt = get_openvla_ckpt_dir(tok, args.dataset)
            if not os.path.isdir(ckpt) and not args.dry_run:
                print(f"  [SKIP] {tok}: checkpoint not found at {ckpt}")
                continue

        submit_script(script, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nMonitor: squeue -u {os.environ.get('USER', 'wenjiel2')}")


if __name__ == "__main__":
    main()
