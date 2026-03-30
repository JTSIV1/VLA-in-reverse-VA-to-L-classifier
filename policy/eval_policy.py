#!/usr/bin/env python3
"""Evaluate a trained MiniVLA policy from the CALVIN sweep.

This is the evaluation entry point — it loads a model and runs evaluation
directly (not a SLURM launcher). run_sweep.sh wraps this in sbatch.

Modes:
    dummy        — Load model, generate one action from a random image. Quick sanity check.
    rollout      — Full CALVIN rollout evaluation (1000 sequences, SR1–SR5).
    teacher_force — Teacher-forced L1 & token accuracy on the CALVIN val set.

Usage:
    # Quick load test
    python policy/eval_policy.py --condition vq_bet_5_16_4 --mode dummy

    # Full rollout
    python policy/eval_policy.py --condition bin --mode rollout

    # Teacher-forced L1 on val set
    python policy/eval_policy.py --condition quest_16_4444_2 --mode teacher_force

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
for p in [str(PROJECT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C  # noqa: E402

OPENVLA_DIR = Path(C.OPENVLA_DIR)
CALVIN_DIR = Path(C.CALVIN_DIR)
SWEEP_DIR = Path(C.SWEEP_DIR)
TOK_DIR = Path(C.TOK_DIR)
POLICY_DIR = Path(C.POLICY_DIR)
DATASET_PATH = C.DATA_ROOT.rstrip("/")

if str(OPENVLA_DIR) not in sys.path:
    sys.path.insert(0, str(OPENVLA_DIR))


# ── Helpers ─────────────────────────────────────────────────────────────────

def find_last_checkpoint(run_dir):
    """Return the checkpoint with the largest step number in run_dir/checkpoints/."""
    import re
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    def step_num(p):
        m = re.search(r"step-(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return str(max(candidates, key=step_num))


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


# ── Teacher-forced eval ────────────────────────────────────────────────────

def teacher_force_eval(run_dir, condition, output_dir, num_batches=None,
                       batch_size=16, device="cuda"):
    """Teacher-forced L1 loss and token accuracy on the CALVIN val set.

    Runs forward passes with ground-truth input tokens (teacher forcing),
    computes argmax predictions at action-token positions, then decodes
    both predicted and GT token IDs back to continuous actions for L1.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from prismatic.models import load_vla
    from prismatic.vla.materialize import get_vla_dataset_and_collator

    run_dir = Path(run_dir)
    fsdp_ckpt = find_last_checkpoint(run_dir)
    if not fsdp_ckpt:
        raise FileNotFoundError(f"No .pt checkpoint in {run_dir}/checkpoints/")

    # Load run config
    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)
    vla_cfg = run_cfg["vla"]

    # Load model
    print(f"Loading VLA from {fsdp_ckpt} ...")
    vla = load_vla(fsdp_ckpt, load_for_training=False)
    vla = vla.to(device).eval()
    action_tokenizer = vla.action_tokenizer

    print(f"  Action tokenizer: {type(action_tokenizer).__name__}")
    print(f"  action_token_begin_idx={action_tokenizer.action_token_begin_idx}, "
          f"action_token_end_idx={action_tokenizer.action_token_end_idx}")

    # Build val dataset using the same config as training
    data_root_dir = Path(run_cfg.get("data_root_dir", C.RLDS_DIR))
    data_mix = vla_cfg.get("data_mix", "calvin_dataset")
    image_transform = vla.vision_backbone.get_image_transform()
    base_tokenizer = vla.llm_backbone.tokenizer
    prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
    default_image_resolution = vla.vision_backbone.default_image_resolution

    future_action_window_size = vla_cfg.get("future_action_window_size", 0)
    future_action_window_size = max(
        action_tokenizer.required_future_horizon, future_action_window_size
    )

    val_dataset, _, collator = get_vla_dataset_and_collator(
        data_root_dir=data_root_dir,
        data_mix=data_mix,
        image_transform=image_transform,
        tokenizer=base_tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        default_image_resolution=default_image_resolution,
        predict_stop_token=vla_cfg.get("predict_stop_token", True),
        shuffle_buffer_size=1000,
        train=False,  # validation split
        image_aug=False,
        action_tokenizer=vla_cfg.get("action_tokenizer", "action_tokenizer"),
        future_action_window_size=future_action_window_size,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collator,
        num_workers=2, pin_memory=True,
    )

    num_patches = vla.vision_backbone.num_patches

    # Accumulators
    total_correct = 0
    total_action_tokens = 0
    total_l1 = 0.0
    total_l1_samples = 0
    total_ce_loss = 0.0
    n_batches = 0

    autocast_dtype = vla.llm_backbone.half_precision_dtype
    print(f"\n=== Teacher-Forced Eval (val split) ===")
    print(f"  Batch size: {batch_size}")
    if num_batches:
        print(f"  Max batches: {num_batches}")

    def to_device(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {kk: to_device(vv, device) for kk, vv in v.items()}
        return v

    for batch in val_loader:
        batch = {k: to_device(v, device) for k, v in batch.items()}

        with torch.no_grad(), torch.autocast(
            "cuda", dtype=autocast_dtype,
            enabled=vla.enable_mixed_precision_training,
        ):
            output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

        # CE loss (computed by the model)
        total_ce_loss += output.loss.item()

        # Extract action predictions and ground truth
        # logits are shifted: logits[t] predicts token[t+1]
        action_preds = output.logits[:, num_patches:-1].argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)

        # Mask for valid action tokens
        mask = ((action_tokenizer.action_token_end_idx > action_gt)
                & (action_gt > action_tokenizer.action_token_begin_idx))

        if mask.sum() == 0:
            continue

        # Token accuracy
        correct = (action_preds == action_gt) & mask
        total_correct += correct.sum().item()
        total_action_tokens += mask.sum().item()

        # L1 on decoded continuous actions
        pred_ids = action_preds[mask].cpu().numpy()
        gt_ids = action_gt[mask].cpu().numpy()

        pred_actions = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(pred_ids),
            dtype=torch.float32,
        )
        gt_actions = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(gt_ids),
            dtype=torch.float32,
        )
        total_l1 += F.l1_loss(pred_actions, gt_actions, reduction="sum").item()
        total_l1_samples += pred_actions.numel()

        n_batches += 1
        if n_batches % 50 == 0:
            running_acc = total_correct / max(total_action_tokens, 1) * 100
            running_l1 = total_l1 / max(total_l1_samples, 1)
            running_ce = total_ce_loss / n_batches
            print(f"  [{n_batches:>5d} batches] "
                  f"CE={running_ce:.4f}  TokAcc={running_acc:.1f}%  L1={running_l1:.4f}")

        if num_batches and n_batches >= num_batches:
            break

    # Final metrics
    token_acc = total_correct / max(total_action_tokens, 1) * 100
    l1 = total_l1 / max(total_l1_samples, 1)
    ce = total_ce_loss / max(n_batches, 1)

    print(f"\n  Results ({n_batches} batches, {total_action_tokens} action tokens):")
    print(f"    CE Loss:        {ce:.4f}")
    print(f"    Token Accuracy: {token_acc:.2f}%")
    print(f"    Action L1:      {l1:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "condition": condition,
        "mode": "teacher_force",
        "n_batches": n_batches,
        "n_action_tokens": total_action_tokens,
        "ce_loss": ce,
        "token_accuracy": token_acc,
        "action_l1": l1,
    }
    out_path = os.path.join(output_dir, "teacher_force_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return results


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
        choices=["dummy", "rollout", "teacher_force"],
        help="dummy = quick load test; rollout = full CALVIN evaluation; "
             "teacher_force = L1 & token accuracy on val set",
    )
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--num_batches", type=int, default=None,
                        help="Max batches for teacher_force mode (default: all)")
    parser.add_argument("--batch_size", type=int, default=16)
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

    elif args.mode == "teacher_force":
        out_dir = os.path.join(args.output_dir, condition)
        teacher_force_eval(
            policy_dir, condition, out_dir,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
