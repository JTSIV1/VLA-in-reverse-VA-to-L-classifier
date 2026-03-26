#!/usr/bin/env python3
"""Evaluate a trained MiniVLA policy from the CALVIN sweep.

This is the evaluation entry point — it loads a model and runs evaluation
directly (not a SLURM launcher). run_sweep.sh wraps this in sbatch.

Modes:
    dummy   — Load model, generate one action from a random image. Quick sanity check.
    rollout — Full CALVIN rollout evaluation (1000 sequences, SR1–SR5).

Usage:
    # Quick load test
    python policy/eval_policy.py --condition vq_bet_5_16_4 --mode dummy

    # Full rollout
    python policy/eval_policy.py --condition bin --mode rollout

    # Explicit policy dir
    python policy/eval_policy.py --policy_dir checkpoints/calvin_sweep/policy/minivla_vq_bet_5_16_4 --mode dummy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
OPENVLA_DIR = Path("/data/user_data/wenjiel2/Code/openvla-mini")
CALVIN_DIR = Path("/data/user_data/wenjiel2/Code/calvin")
SWEEP_DIR = PROJECT_DIR / "checkpoints" / "calvin_sweep"
TOK_DIR = SWEEP_DIR / "tokenizers"
POLICY_DIR = SWEEP_DIR / "policy"
DATASET_PATH = "/data/user_data/yashagar/task_D_D"

for p in [str(PROJECT_DIR), str(OPENVLA_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ─────────────────────────────────────────────────────────────────

def find_last_checkpoint(run_dir):
    """Return the last .pt checkpoint (by filename sort) in run_dir/checkpoints/."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"))
    return str(candidates[-1]) if candidates else None


def resolve_policy_dir(condition):
    """Resolve a condition tag to the policy directory path."""
    policy_dir = POLICY_DIR / f"minivla_{condition}"
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")
    return str(policy_dir)


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(run_dir, device="cuda"):
    """Load a MiniVLA from an FSDP checkpoint.

    Returns (vla, action_tokenizer, is_chunk_tokenizer, n_action_tokens, chunk_size).
    """
    run_dir = Path(run_dir)
    fsdp_ckpt = find_last_checkpoint(run_dir)
    if not fsdp_ckpt:
        raise FileNotFoundError(f"No .pt checkpoint in {run_dir}/checkpoints/")

    print(f"Loading VLA from {fsdp_ckpt} ...")
    from prismatic.models import load_vla

    vla = load_vla(fsdp_ckpt, load_for_training=False)
    vla = vla.to(device).eval()

    action_tokenizer = vla.action_tokenizer

    # Detect if this is a chunk tokenizer (sweep) or per-step (bin)
    is_chunk = hasattr(action_tokenizer, "chunk_size")
    n_action_tokens = getattr(action_tokenizer, "n_codes_per_chunk", 7)
    chunk_size = getattr(action_tokenizer, "chunk_size", 1)

    print(f"  Action tokenizer: {type(action_tokenizer).__name__}")
    print(f"  Chunk tokenizer: {is_chunk}, tokens_per_step={n_action_tokens}, chunk_size={chunk_size}")
    print("Model ready.")

    return vla, action_tokenizer, is_chunk, n_action_tokens, chunk_size


# ── Dummy eval ──────────────────────────────────────────────────────────────

def dummy_eval(vla, action_tokenizer, is_chunk, n_action_tokens, device="cuda"):
    """Generate one action from a random image to verify the model loads correctly."""
    print("\n=== Dummy Eval ===")

    dummy_rgb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_rgb).convert("RGB")
    instruction = "pick up the red block"

    print(f"  Instruction: {instruction}")
    print(f"  Image: 200x200 random RGB")

    if not is_chunk:
        # Bin tokenizer: use predict_action directly
        with torch.no_grad():
            action = vla.predict_action(
                image, instruction,
                unnorm_key="calvin_dataset",
                do_sample=False,
            )
        print(f"  Action shape: {action.shape}")
        print(f"  Action: {action}")
    else:
        # Chunk tokenizer: generate n_action_tokens, decode to chunk
        from transformers import GenerationMixin

        tokenizer = vla.llm_backbone.tokenizer
        image_transform = vla.vision_backbone.get_image_transform()

        prompt_builder = vla.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message="What action should the robot take to {}?".format(instruction),
        )
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(device) for k, v in pixel_values.items()}

        autocast_dtype = vla.llm_backbone.half_precision_dtype
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=vla.enable_mixed_precision_training
        ):
            generated = GenerationMixin.generate(
                vla,
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=n_action_tokens,
                do_sample=False,
            )

        code_token_ids = generated[0, -n_action_tokens:].cpu().numpy()
        print(f"  Generated token IDs: {code_token_ids}")

        actions = action_tokenizer.decode_full_chunk(code_token_ids)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.atleast_2d(actions).astype(np.float32)

        print(f"  Decoded chunk shape: {actions.shape}")
        print(f"  First action: {actions[0]}")

    print("\n  SUCCESS — model loaded and produced actions.")


# ── Rollout eval ────────────────────────────────────────────────────────────

def rollout_eval(run_dir, condition, num_sequences, output_dir, device="cuda"):
    """Run full CALVIN rollout evaluation."""
    fsdp_ckpt = find_last_checkpoint(run_dir)
    if not fsdp_ckpt:
        raise FileNotFoundError(f"No .pt checkpoint in {run_dir}/checkpoints/")

    # Read tokenizer info from config.json
    config_json = Path(run_dir) / "config.json"
    with open(config_json) as f:
        config = json.load(f)
    action_tok = config.get("vla", {}).get("action_tokenizer", "")

    sweep_type, sweep_path = "", ""
    if action_tok.startswith("sweep:"):
        _, sweep_type, sweep_path = action_tok.split(":", 2)

    # Delegate to the existing rollout eval
    from policy.scripts.evaluate_openvla_rollout import run_rollout_eval

    return run_rollout_eval(
        condition=condition,
        checkpoint_dir=str(run_dir),
        vqvla_checkpoint_dir="",
        dataset_path=DATASET_PATH,
        output_dir=output_dir,
        num_sequences=num_sequences,
        ep_len=360,
        device=device,
        sweep_tokenizer_type=sweep_type,
        sweep_checkpoint_path=sweep_path,
        fsdp_checkpoint=fsdp_ckpt,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a MiniVLA policy from the CALVIN sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--condition",
        help="Condition tag (e.g. vq_bet_5_16_4, bin, quest_16_4444_2). "
             "Resolves to checkpoints/calvin_sweep/policy/minivla_<condition>/",
    )
    parser.add_argument(
        "--policy_dir",
        help="Explicit policy directory (overrides --condition)",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["dummy", "rollout"],
        help="dummy = quick load test; rollout = full CALVIN evaluation",
    )
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument(
        "--output_dir", default=str(PROJECT_DIR / "results"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not args.condition and not args.policy_dir:
        parser.error("Provide --condition or --policy_dir")

    # Resolve policy directory
    if args.policy_dir:
        policy_dir = args.policy_dir
        condition = Path(policy_dir).name.replace("minivla_", "")
    else:
        policy_dir = resolve_policy_dir(args.condition)
        condition = args.condition

    print(f"Condition: {condition}")
    print(f"Policy dir: {policy_dir}")

    if args.mode == "dummy":
        vla, action_tokenizer, is_chunk, n_tokens, _chunk_size = load_model(
            policy_dir, device=args.device
        )
        dummy_eval(vla, action_tokenizer, is_chunk, n_tokens, device=args.device)

    elif args.mode == "rollout":
        out_dir = os.path.join(args.output_dir, condition)
        rollout_eval(
            policy_dir, condition, args.num_sequences, out_dir, device=args.device,
        )


if __name__ == "__main__":
    main()
