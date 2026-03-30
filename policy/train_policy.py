#!/usr/bin/env python3
"""Unified VLA policy training launcher.

Submits SLURM jobs for fine-tuning OpenVLA-7B (LoRA) or training MiniVLA (0.5B)
on CALVIN with configurable action tokenizers.

Usage:
    # OpenVLA-7B with bin tokenizer on CALVIN D
    python policy/train_policy.py --model openvla --tokenizer bin

    # MiniVLA with VQ-VLA verb tokenizer
    python policy/train_policy.py --model minivla --tokenizer vqvla_verb

    # OpenVLA on ABCD with VQ-VLA vanilla
    python policy/train_policy.py --model openvla --tokenizer vqvla_vanilla --dataset calvin_abcd_dataset

    # Dry run (print sbatch command without submitting)
    python policy/train_policy.py --model openvla --tokenizer bin --dry_run

Tokenizer options:
    bin            — per-dim uniform binning (256 bins, 7 tokens/step)
    vqvla_vanilla  — VQ-VLA pretrained, fine-tuned vanilla (4 tokens/5-step chunk)
    vqvla_verb     — VQ-VLA fine-tuned with verb aux loss (lambda=0.5)
    vqvla_verb01   — VQ-VLA fine-tuned with verb aux loss (lambda=0.1)
    vq_bet         — VQ-BeT vanilla (chunk MLP + ResidualVQ, 2 tokens/5-step chunk)
    oat            — OAT vanilla (register encoder + FSQ, 4 tokens/chunk)
    quest          — QueST vanilla (causal conv + FSQ, 4 tokens/chunk)
    fast           — FAST (DCT + BPE, variable tokens/step)

    Sweep variants: append _verb{L} or _clip{L} to vq_bet/oat/quest, e.g.:
        vq_bet_verb0.5, oat_clip1.0, quest_verb0.1
"""

import argparse
import os
import subprocess
import sys

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
import config as C  # noqa: E402

OPENVLA_DIR = C.OPENVLA_DIR
DATA_DIR = C.RLDS_DIR

# Tokenizer checkpoint directories
# VQ-VLA (from Stage 1 fine-tuning)
VQVLA_CKPTS = {
    "vqvla_vanilla": os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_vanilla"),
    "vqvla_verb":    os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.5"),
    "vqvla_verb01":  os.path.join(PROJECT_DIR, "checkpoints", "vqvla_ft_verb_l0.1"),
}

# Calvind sweep tokenizers (from tokenization/train_tokenizer.py)
SWEEP_DIR = C.SWEEP_DIR

# FAST tokenizer (from DCT + BPE fitting)
FAST_DIR = os.path.join(PROJECT_DIR, "checkpoints")

# Map tokenizer family → type string passed to finetune.py
TOKENIZER_FAMILIES = {
    "bin":   "bin",
    "fast":  "fast",
    "vq_bet": "vq_bet",
    "oat":   "oat",
    "quest": "quest",
}


def resolve_tokenizer(name):
    """Resolve tokenizer name to (tokenizer_type, checkpoint_dir_or_None).

    Supported patterns:
      bin                     → ("bin", None)
      vqvla_vanilla           → ("vqvla", <vqvla ckpt>)
      vq_bet                  → ("vq_bet", <sweep>/vq_bet_vanilla)
      vq_bet_verb0.5          → ("vq_bet", <sweep>/vq_bet_verb0.5)
      oat_clip1.0             → ("oat",   <sweep>/oat_clip1.0)
      quest                   → ("quest", <sweep>/quest_vanilla)
      fast                    → ("fast",  <fast dir>/fast_s1_v256)
    """
    if name == "bin":
        return "bin", None
    if name in VQVLA_CKPTS:
        return "vqvla", VQVLA_CKPTS[name]
    if name == "fast":
        return "fast", os.path.join(FAST_DIR, "fast_s1_v256")

    # Sweep tokenizers: vq_bet, oat, quest (with optional _variant suffix)
    for family in ("vq_bet", "oat", "quest"):
        if name == family:
            return family, os.path.join(SWEEP_DIR, f"{family}_vanilla")
        if name.startswith(family + "_"):
            return family, os.path.join(SWEEP_DIR, name)

    raise ValueError(f"Unknown tokenizer: {name}")

# MiniVLA config IDs (registered in openvla-mini/prismatic/conf/vla.py)
MINIVLA_CONFIGS = {
    "bin":            "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-bin",
    "vqvla_vanilla":  "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-vq-vanilla",
    "vqvla_verb":     "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-vq-verb",
    "vqvla_verb01":   "prism-qwen25-dinosiglip-224px+0_5b+mx-calvin-d-vq-verb01",
}


# ── SLURM defaults ──────────────────────────────────────────────────────────

SLURM_DEFAULTS = {
    "openvla": {
        "partition": "general",
        "gres": "gpu:L40S:1",
        "cpus": 8,
        "mem": "64G",
        "time": "30:00:00",
    },
    "minivla": {
        "partition": "general",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "64G",
        "time": "30:00:00",
    },
}

# ── Training defaults ────────────────────────────────────────────────────────

OPENVLA_DEFAULTS = dict(
    vla_path="openvla/openvla-7b",
    lora_rank=32,
    batch_size=8,
    grad_accumulation_steps=2,
    learning_rate=5e-4,
    max_steps=50000,
    save_steps=5000,
    val_steps=1000,
    warmup_steps=500,
    max_grad_norm=1.0,
    image_aug=True,
    shuffle_buffer_size=50000,
)


def build_openvla_command(args):
    """Build torchrun command for OpenVLA-7B LoRA fine-tuning."""
    run_dir = os.path.join(PROJECT_DIR, "checkpoints", "calvin_sweep", "policy", "openvla")
    adapter_dir = os.path.join(PROJECT_DIR, "runs", "openvla_adapter_tmp")
    run_note = f"calvin_{args.tokenizer}"
    if args.dataset != "calvin_dataset":
        run_note = f"abcd_{args.tokenizer}"

    tok_type, tok_ckpt = resolve_tokenizer(args.tokenizer)

    cmd = [
        "torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        "vla-scripts/finetune.py",
        "--vla_path", OPENVLA_DEFAULTS["vla_path"],
        "--data_root_dir", DATA_DIR,
        "--dataset_name", args.dataset,
        "--run_root_dir", run_dir,
        "--adapter_tmp_dir", adapter_dir,
        "--lora_rank", str(OPENVLA_DEFAULTS["lora_rank"]),
        "--batch_size", str(args.batch_size or OPENVLA_DEFAULTS["batch_size"]),
        "--grad_accumulation_steps", str(OPENVLA_DEFAULTS["grad_accumulation_steps"]),
        "--learning_rate", str(args.lr or OPENVLA_DEFAULTS["learning_rate"]),
        "--max_steps", str(args.max_steps or OPENVLA_DEFAULTS["max_steps"]),
        "--save_steps", str(OPENVLA_DEFAULTS["save_steps"]),
        "--val_steps", str(OPENVLA_DEFAULTS["val_steps"]),
        "--warmup_steps", str(OPENVLA_DEFAULTS["warmup_steps"]),
        "--max_grad_norm", str(OPENVLA_DEFAULTS["max_grad_norm"]),
        "--image_aug", str(OPENVLA_DEFAULTS["image_aug"]),
        "--shuffle_buffer_size", str(OPENVLA_DEFAULTS["shuffle_buffer_size"]),
        "--run_id_note", run_note,
    ]

    # Tokenizer-specific args
    if tok_type == "vqvla":
        cmd.extend(["--vqvla_checkpoint_dir", tok_ckpt])
    elif tok_type in ("vq_bet", "oat", "quest", "fast"):
        cmd.extend(["--action_tokenizer_type", tok_type,
                     "--action_tokenizer_ckpt", tok_ckpt])
    # bin: no extra args

    return cmd


def build_minivla_command(args):
    """Build torchrun command for MiniVLA training."""
    tok_type, tok_ckpt = resolve_tokenizer(args.tokenizer)

    if args.tokenizer in MINIVLA_CONFIGS:
        vla_type = MINIVLA_CONFIGS[args.tokenizer]
    else:
        # For new tokenizers, use the bin base config (tokenizer override via args)
        print(f"Note: MiniVLA config not registered for '{args.tokenizer}', "
              f"using bin base config with tokenizer override.")
        vla_type = MINIVLA_CONFIGS["bin"]

    run_dir = os.path.join(PROJECT_DIR, "checkpoints", "calvin_sweep", "policy", "minivla")
    run_note = f"calvin_d_{args.tokenizer}"
    if args.dataset != "calvin_dataset":
        run_note = f"abcd_{args.tokenizer}"

    cmd = [
        "torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        "vla-scripts/train.py",
        f"--vla.type={vla_type}",
        "--data_root_dir", DATA_DIR,
        "--run_root_dir", run_dir,
        "--image_aug", "True",
        "--save_interval", "5000",
        "--run_id_note", run_note,
    ]

    # Tokenizer-specific args
    if tok_type == "vqvla":
        cmd.extend(["--vqvla_checkpoint_dir", tok_ckpt])
    elif tok_type in ("vq_bet", "oat", "quest", "fast"):
        cmd.extend(["--action_tokenizer_type", tok_type,
                     "--action_tokenizer_ckpt", tok_ckpt])

    return cmd


def build_sbatch_script(args, train_cmd):
    """Generate sbatch script content."""
    slurm = SLURM_DEFAULTS[args.model]
    job_name = f"{args.model}_{args.tokenizer}"
    if args.dataset != "calvin_dataset":
        job_name += "_abcd"

    log_dir = os.path.join(PROJECT_DIR, "logs")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition or slurm['partition']}",
        f"#SBATCH --gres={args.gres or slurm['gres']}",
        f"#SBATCH --cpus-per-task={slurm['cpus']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --time={args.time or slurm['time']}",
        f"#SBATCH -o {log_dir}/{job_name}_%j.out",
        f"#SBATCH -e {log_dir}/{job_name}_%j.err",
        "",
        f'source $(conda info --base)/etc/profile.d/conda.sh',
        "conda activate mmml",
        "",
        f'export PRISMATIC_DATA_ROOT="{DATA_DIR}"',
        "export WANDB_MODE=offline",
        "",
        f'pip install -e "{OPENVLA_DIR}" --quiet 2>/dev/null || true',
        "",
        f'mkdir -p "{os.path.join(PROJECT_DIR, "runs")}" "{os.path.join(PROJECT_DIR, "logs")}"',
        "",
        f'cd "{OPENVLA_DIR}"',
        "",
        "# train.py reads .hf_token; create if missing",
        f'touch -a "{OPENVLA_DIR}/.hf_token"',
        "",
        " ".join(train_cmd),
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Unified VLA policy training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, choices=["openvla", "minivla"],
                        help="Model to train")
    parser.add_argument("--tokenizer", required=True,
                        help="Action tokenizer (bin, vqvla_vanilla, vqvla_verb, "
                             "vqvla_verb01, vq_bet, oat, quest, fast, or "
                             "sweep variants like vq_bet_verb0.5)")
    parser.add_argument("--dataset", default="calvin_dataset",
                        choices=["calvin_dataset", "calvin_abcd_dataset"],
                        help="TFDS dataset name")

    # Override defaults
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)

    # SLURM overrides
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--time", default=None)

    parser.add_argument("--dry_run", action="store_true",
                        help="Print sbatch script without submitting")
    args = parser.parse_args()

    # Validate tokenizer name
    try:
        resolve_tokenizer(args.tokenizer)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # Build training command
    if args.model == "openvla":
        train_cmd = build_openvla_command(args)
    else:
        train_cmd = build_minivla_command(args)

    # Build sbatch script
    script = build_sbatch_script(args, train_cmd)

    if args.dry_run:
        print("=== SBATCH SCRIPT (dry run) ===")
        print(script)
        return

    # Submit via sbatch --wrap would lose SBATCH directives; use temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"Error: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
