#!/usr/bin/env python3
"""End-to-end experiment runner: tokenizer training → verb probe → policy.

Orchestrates three stages as SLURM jobs with dependency chaining:
  Stage 1: Fit action tokenizer on CALVIN (with optional aux head).
  Stage 2: Run verb probes on native actions and tokenized representations.
  Stage 3: Train VLA policy (MiniVLA or OpenVLA) with the fitted tokenizer.

Usage:
    # Full pipeline: train VQ-BeT with verb aux, probe, then train MiniVLA
    python scripts/run_experiment.py \
        --tokenizer vq_bet --aux_head verb --aux_lambda 0.1 \
        --policy_model minivla

    # Skip tokenizer training, probe with existing checkpoint
    python scripts/run_experiment.py \
        --tokenizer quest --tokenizer_ckpt checkpoints/quest_best/full.pth \
        --stages probe

    # Just tokenizer + probe (no policy)
    python scripts/run_experiment.py \
        --tokenizer oat --aux_head clip --aux_lambda 0.5 \
        --stages tokenizer probe

    # Dry run — print sbatch scripts without submitting
    python scripts/run_experiment.py \
        --tokenizer vq_bet --aux_head verb \
        --dry_run
"""

import argparse
import os
import subprocess
import sys
import tempfile

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENVLA_DIR = "/data/user_data/wenjiel2/Code/openvla-mini"
DATA_DIR = "/data/user_data/wenjiel2/datasets/calvin/task_ABCD_D"
VAL_DIR = "/data/user_data/wenjiel2/datasets/calvin/task_ABCD_D"
RLDS_DIR = "/data/user_data/wenjiel2/datasets/calvin_rlds"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

BASE_VLM = (
    "/data/user_data/wenjiel2/.cache/huggingface/models--Stanford-ILIAD--"
    "prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/"
    "5cfd2cc6da00c06e0be7abf35d43ec792d8e9498"
)

# ── SLURM defaults ───────────────────────────────────────────────────────────

SLURM = {
    "tokenizer": {"partition": "general", "gres": "gpu:1", "mem": "32G",
                   "cpus": 8, "time": "8:00:00"},
    "probe":     {"partition": "general", "gres": "gpu:1", "mem": "32G",
                   "cpus": 8, "time": "4:00:00"},
    "policy":    {"partition": "general", "gres": "gpu:1", "mem": "64G",
                   "cpus": 8, "time": "24:00:00"},
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _sbatch_header(job_name, stage, args):
    """Generate SBATCH header lines for a stage."""
    s = SLURM[stage]
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition or s['partition']}",
        f"#SBATCH --gres={args.gres or s['gres']}",
        f"#SBATCH --cpus-per-task={s['cpus']}",
        f"#SBATCH --mem={s['mem']}",
        f"#SBATCH --time={s['time']}",
        f"#SBATCH -o {LOG_DIR}/{job_name}_%j.out",
        f"#SBATCH -e {LOG_DIR}/{job_name}_%j.err",
    ])


def _conda_preamble():
    return "\n".join([
        "",
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate mmml",
        f'cd "{PROJECT_DIR}"',
        f'export PYTHONPATH="{PROJECT_DIR}:${{PYTHONPATH:-}}"',
        "",
    ])


def _run_name(args):
    """Derive a short run name from experiment args."""
    name = args.tokenizer
    if args.aux_head != "none":
        name += f"_{args.aux_head}{args.aux_lambda}"
    if args.tag:
        name += f"_{args.tag}"
    return name


def _ckpt_dir(args):
    """Expected checkpoint directory for this tokenizer run."""
    return os.path.join(args.save_dir, _run_name(args))


def _submit(script_content, job_name, dry_run, dependency_id=None):
    """Submit an sbatch script. Returns the SLURM job ID or None if dry run."""
    if dry_run:
        print(f"\n{'='*60}")
        print(f"=== {job_name} (dry run) ===")
        print(f"{'='*60}")
        print(script_content)
        return None

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix=f"{job_name}_") as f:
        f.write(script_content)
        tmp_path = f.name

    cmd = ["sbatch"]
    if dependency_id is not None:
        cmd.extend(["--dependency", f"afterok:{dependency_id}"])
    cmd.append(tmp_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted {job_name}: job {job_id}")
        os.unlink(tmp_path)
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ERROR submitting {job_name}: {e.stderr.strip()}")
        os.unlink(tmp_path)
        return None


# ── Stage 1: Tokenizer training ─────────────────────────────────────────────

def build_tokenizer_script(args):
    """Build sbatch script for tokenizer training."""
    job_name = f"tok_{_run_name(args)}"

    cmd_parts = [
        "python -u tokenization/train_tokenizer.py",
        f"--tokenizer {args.tokenizer}",
        f"--dataset calvin",
        f"--epochs {args.tok_epochs}",
        f"--batch_size {args.tok_batch_size}",
        f"--lr {args.tok_lr}",
        f"--save_dir {args.save_dir}",
        f"--min_class_count 30",
        f"--max_chunks 8",
    ]
    if args.aux_head != "none":
        cmd_parts.append(f"--aux_head {args.aux_head}")
        cmd_parts.append(f"--aux_lambda {args.aux_lambda}")
    if args.tag:
        cmd_parts.append(f"--tag {args.tag}")
    if args.tok_config:
        cmd_parts.append(f"--config {args.tok_config}")
    if args.tok_set:
        cmd_parts.append("--set " + " ".join(args.tok_set))

    script = "\n".join([
        _sbatch_header(job_name, "tokenizer", args),
        _conda_preamble(),
        " \\\n    ".join(cmd_parts),
        "",
    ])
    return script, job_name


# ── Stage 2: Verb probes ────────────────────────────────────────────────────

def build_probe_scripts(args, ckpt_path):
    """Build sbatch scripts for verb probes.

    Returns list of (script_content, job_name) tuples.
    Runs probes for:
      1. native action (baseline)
      2. token_id (discrete codes from tokenizer)
      3. latent (continuous latents from tokenizer)
    """
    scripts = []
    run = _run_name(args)

    # Common probe args
    common = [
        "--modality action_only",
        f"--epochs {args.probe_epochs}",
        f"--batch_size {args.probe_batch_size}",
        "--min_class_count 30",
        "--weighted_loss",
        f"--d_model {args.probe_d_model}",
    ]

    # 1) Native action baseline
    job_name = f"probe_native_{run}"
    save_path = os.path.join(args.save_dir, run, f"probe_native_best.pth")
    cmd_parts = [
        "python -u verb_probe/train_verb_probe.py",
        "--action_rep native",
        f"--save_path {save_path}",
    ] + common

    scripts.append(("\n".join([
        _sbatch_header(job_name, "probe", args),
        _conda_preamble(),
        " \\\n    ".join(cmd_parts),
        "",
    ]), job_name))

    # 2) Token ID probe (discrete codes)
    job_name = f"probe_tokid_{run}"
    save_path = os.path.join(args.save_dir, run, f"probe_tokid_best.pth")
    cmd_parts = [
        "python -u verb_probe/train_verb_probe.py",
        f"--action_rep {args.tokenizer}",
        f"--tokenizer_type {args.tokenizer}",
        f"--tokenizer_ckpt {ckpt_path}",
        f"--save_path {save_path}",
    ] + common

    scripts.append(("\n".join([
        _sbatch_header(job_name, "probe", args),
        _conda_preamble(),
        " \\\n    ".join(cmd_parts),
        "",
    ]), job_name))

    # 3) Latent probe (continuous vectors)
    job_name = f"probe_latent_{run}"
    save_path = os.path.join(args.save_dir, run, f"probe_latent_best.pth")
    cmd_parts = [
        "python -u verb_probe/train_verb_probe.py",
        "--action_rep latent",
        f"--tokenizer_type {args.tokenizer}",
        f"--tokenizer_ckpt {ckpt_path}",
        f"--save_path {save_path}",
    ] + common

    scripts.append(("\n".join([
        _sbatch_header(job_name, "probe", args),
        _conda_preamble(),
        " \\\n    ".join(cmd_parts),
        "",
    ]), job_name))

    return scripts


# ── Stage 3: Policy training ────────────────────────────────────────────────

def build_policy_script(args, ckpt_path):
    """Build sbatch script for MiniVLA or OpenVLA policy training."""
    run = _run_name(args)
    job_name = f"pol_{args.policy_model}_{run}"

    vla_config = "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin"
    run_dir = os.path.join(PROJECT_DIR, "runs", f"{args.policy_model}_{run}")

    if args.policy_model == "minivla":
        cmd_parts = [
            "torchrun --standalone --nnodes 1 --nproc-per-node 1",
            "vla-scripts/train.py",
            f"--vla.type {vla_config}",
            f"--vla.base_vlm {BASE_VLM}",
            f"--data_root_dir {RLDS_DIR}",
            f"--run_root_dir {run_dir}",
            "--image_aug True",
            "--save_interval 5000",
            f"--run_id_note {run}",
            "--vla.expected_world_size 1",
            "--vla.global_batch_size 16",
            "--vla.per_device_batch_size 16",
            "--vla.freeze_vision_backbone True",
        ]
        # Tokenizer config — sweep format for custom tokenizers
        if args.tokenizer != "bin":
            tok_str = f"sweep:{args.tokenizer}:{ckpt_path}"
            cmd_parts.append(f"--vla.action_tokenizer '{tok_str}'")

    else:  # openvla
        cmd_parts = [
            "torchrun --standalone --nnodes 1 --nproc-per-node 1",
            "vla-scripts/finetune.py",
            "--vla_path openvla/openvla-7b",
            f"--data_root_dir {RLDS_DIR}",
            f"--dataset_name calvin_dataset",
            f"--run_root_dir {run_dir}",
            "--lora_rank 32",
            f"--batch_size {args.policy_batch_size}",
            "--grad_accumulation_steps 2",
            f"--learning_rate {args.policy_lr}",
            f"--max_steps {args.policy_max_steps}",
            "--save_steps 5000",
            "--val_steps 1000",
            "--warmup_steps 500",
            "--max_grad_norm 1.0",
            "--image_aug True",
            "--shuffle_buffer_size 50000",
            f"--run_id_note {run}",
        ]
        if args.tokenizer != "bin":
            cmd_parts.extend([
                f"--action_tokenizer_type {args.tokenizer}",
                f"--action_tokenizer_ckpt {ckpt_path}",
            ])

    preamble = "\n".join([
        "",
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate mmml",
        f'export PRISMATIC_DATA_ROOT="{RLDS_DIR}"',
        "export WANDB_MODE=offline",
        f'cd "{OPENVLA_DIR}"',
        "",
    ])

    script = "\n".join([
        _sbatch_header(job_name, "policy", args),
        preamble,
        f'mkdir -p "{run_dir}"',
        "",
        " \\\n    ".join(cmd_parts),
        "",
    ])
    return script, job_name


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end experiment: tokenizer → verb probe → policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment identity
    parser.add_argument("--tokenizer", required=True,
                        choices=["vq_bet", "oat", "quest"],
                        help="Action tokenizer architecture")
    parser.add_argument("--aux_head", default="none",
                        choices=["none", "verb", "clip"],
                        help="Auxiliary head for tokenizer training")
    parser.add_argument("--aux_lambda", type=float, default=0.5,
                        help="Auxiliary loss weight")
    parser.add_argument("--tag", default="",
                        help="Optional experiment tag suffix")

    # Stage selection
    parser.add_argument("--stages", nargs="+",
                        default=["tokenizer", "probe", "policy"],
                        choices=["tokenizer", "probe", "policy"],
                        help="Which stages to run (default: all three)")
    parser.add_argument("--tokenizer_ckpt", default=None,
                        help="Skip tokenizer training; use this checkpoint for probe/policy")

    # Tokenizer training (Stage 1)
    parser.add_argument("--tok_epochs", type=int, default=100)
    parser.add_argument("--tok_batch_size", type=int, default=32)
    parser.add_argument("--tok_lr", type=float, default=1e-4)
    parser.add_argument("--tok_config", default=None,
                        help="YAML config for tokenizer (default: auto from tokenizer type)")
    parser.add_argument("--tok_set", nargs="*", default=[],
                        help="Override tokenizer config values (KEY=VAL)")

    # Verb probe (Stage 2)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--probe_d_model", type=int, default=128)

    # Policy training (Stage 3)
    parser.add_argument("--policy_model", default="minivla",
                        choices=["minivla", "openvla"],
                        help="VLA model for policy training")
    parser.add_argument("--policy_batch_size", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=5e-4)
    parser.add_argument("--policy_max_steps", type=int, default=50000)

    # Output / SLURM
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_DIR, "checkpoints"))
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print sbatch scripts without submitting")

    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    stages = args.stages
    run = _run_name(args)

    # Determine checkpoint path
    if args.tokenizer_ckpt:
        ckpt_path = args.tokenizer_ckpt
        if "tokenizer" in stages:
            print("Warning: --tokenizer_ckpt provided; skipping tokenizer stage")
            stages = [s for s in stages if s != "tokenizer"]
    else:
        ckpt_path = os.path.join(_ckpt_dir(args), "full.pth")

    print(f"Experiment: {run}")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"  Tokenizer: {args.tokenizer}")
    if args.aux_head != "none":
        print(f"  Aux head: {args.aux_head} (lambda={args.aux_lambda})")
    print(f"  Checkpoint: {ckpt_path}")
    print()

    # ── Stage 1: Tokenizer ───────────────────────────────────────────────
    tok_job_id = None
    if "tokenizer" in stages:
        script, job_name = build_tokenizer_script(args)
        tok_job_id = _submit(script, job_name, args.dry_run)

    # ── Stage 2: Verb probes ─────────────────────────────────────────────
    probe_job_ids = []
    if "probe" in stages:
        probe_scripts = build_probe_scripts(args, ckpt_path)
        for script, job_name in probe_scripts:
            jid = _submit(script, job_name, args.dry_run,
                          dependency_id=tok_job_id)
            if jid:
                probe_job_ids.append(jid)

    # ── Stage 3: Policy ──────────────────────────────────────────────────
    if "policy" in stages:
        script, job_name = build_policy_script(args, ckpt_path)
        # Policy depends on tokenizer (not probes — probes are informational)
        _submit(script, job_name, args.dry_run,
                dependency_id=tok_job_id)

    if not args.dry_run:
        print(f"\nMonitor: squeue -u $(whoami)")
        print(f"Checkpoints: {_ckpt_dir(args)}/")


if __name__ == "__main__":
    main()
