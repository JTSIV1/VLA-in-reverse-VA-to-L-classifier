#!/usr/bin/env python3
"""Evaluate a trained MiniVLA policy.

This is the evaluation entry point — it loads a model and runs evaluation
directly (not a SLURM launcher). run_sweep.sh wraps this in sbatch.

Modes:
    dummy        — Load model, generate one action from a random image. Quick sanity check.
    rollout      — Full CALVIN rollout evaluation (1000 sequences, SR1–SR5).
    teacher_force — Teacher-forced L1 & token accuracy on the val set.
                    Reports: CE loss, token accuracy, overall L1,
                    per-dimension L1 (x/y/z/roll/pitch/yaw/gripper),
                    per-verb L1 and token accuracy (verb extracted from instruction).

Usage:
    # Quick load test
    python policy/eval_policy.py --condition vq_bet_5_16_4 --mode dummy

    # Full rollout (CALVIN only)
    python policy/eval_policy.py --condition bin --mode rollout

    # Teacher-forced eval on val set
    python policy/eval_policy.py --condition quest_16_4444_2 --mode teacher_force

    # Explicit policy dir
    python policy/eval_policy.py --policy_dir checkpoints/bridge_sweep/policy/minivla_vq_bet_5_16_2_256 --mode teacher_force
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

ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _load_bridge_verb_lookup(csv_path):
    """Load instruction→verb mapping from bridge_episodes_filtered.csv.

    This ensures eval uses the same spaCy-based verb extraction and filtering
    as the training set.  Instructions not in the CSV are treated as
    unannotated and skipped during evaluation.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        instr = str(row["instruction"]).strip()
        if instr and instr != "nan":
            lookup[instr] = row["verb"]
    print(f"  Loaded {len(lookup)} instruction→verb mappings from {csv_path}")
    return lookup


def _extract_instructions_from_input_ids(input_ids, tokenizer):
    """Decode input_ids and extract the instruction from the VLA prompt.

    Prompt format: "What action should the robot take to {lang}?"
    Returns a list of instruction strings, one per sample in the batch.
    """
    instructions = []
    for ids in input_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Extract between "to " and "?"
        marker = "take to "
        start = text.find(marker)
        if start == -1:
            instructions.append("")
            continue
        start += len(marker)
        end = text.find("?", start)
        if end == -1:
            end = len(text)
        instructions.append(text[start:end].strip())
    return instructions


# ── Verb perturbation helpers ──────────────────────────────────────────────

_NLP_CACHE = None
PROMPT_MARKER = "take to "


def _get_nlp():
    global _NLP_CACHE
    if _NLP_CACHE is None:
        import spacy
        _NLP_CACHE = spacy.load("en_core_web_sm")
    return _NLP_CACHE


def _find_verb_char_span(instruction, verb, nlp):
    """Find char (start, end) of the verb (any inflection) in instruction via
    spaCy lemma matching. Handles multi-word verbs ('pick up'). Falls back to
    case-insensitive substring match on the verb text itself."""
    if not instruction or not verb:
        return None
    doc = nlp(instruction)
    verb_parts = verb.lower().split()
    n = len(verb_parts)
    for i in range(len(doc) - n + 1):
        if all(doc[i + j].lemma_.lower() == verb_parts[j] for j in range(n)):
            start = doc[i].idx
            end = doc[i + n - 1].idx + len(doc[i + n - 1].text)
            return (start, end)
    lo = instruction.lower()
    vlo = verb.lower()
    pos = lo.find(vlo)
    if pos != -1:
        return (pos, pos + len(vlo))
    return None


def _build_instruction_verb_spans(csv_path, nlp):
    """Precompute {instruction -> (start, end)} for every annotated instruction."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    spans = {}
    n_missed = 0
    for _, row in df.iterrows():
        instr = str(row["instruction"]).strip()
        if not instr or instr == "nan":
            continue
        verb = str(row["verb"])
        sp = _find_verb_char_span(instr, verb, nlp)
        if sp is not None:
            spans[instr] = sp
        else:
            n_missed += 1
    return spans, n_missed


def _make_metrics_accumulator(n_codes, action_dim, step_indices):
    """Create a fresh dict of accumulators for teacher-force metrics."""
    from collections import defaultdict
    return {
        "total_correct": 0,
        "total_action_tokens": 0,
        "total_l1": 0.0,             # decode(pred) vs decode(gt) — first step only (legacy)
        "total_l1_samples": 0,
        "total_l1_vs_raw": 0.0,      # decode(pred) vs RAW gt actions (chunk-mean per-step)
        "total_l1_vs_raw_samples": 0,
        "total_ce_loss": 0.0,
        "n_batches_contributed": 0,
        "pos_correct": np.zeros(n_codes, dtype=np.int64),
        "pos_total": np.zeros(n_codes, dtype=np.int64),
        "cumul_l1_sum": {t: 0.0 for t in step_indices},          # vs decode(gt)
        "cumul_l1_count": {t: 0 for t in step_indices},
        "cumul_l1_vs_raw_sum": {t: 0.0 for t in step_indices},   # vs RAW gt
        "cumul_l1_vs_raw_count": {t: 0 for t in step_indices},
        "dim_l1_sum": np.zeros(action_dim, dtype=np.float64),
        "dim_l1_count": np.zeros(action_dim, dtype=np.int64),
        "verb_l1_sum": defaultdict(float),
        "verb_l1_count": defaultdict(int),
        "verb_correct": defaultdict(int),
        "verb_total_tokens": defaultdict(int),
    }


def _accumulate_batch_metrics(output, batch, instructions, acc, action_tokenizer,
                               num_patches, n_codes, step_indices, action_dim,
                               verb_lookup, per_sample_sink=None, batch_idx=0,
                               n_verb_tokens_per_sample=None):
    """Update acc in place with metrics from one forward pass.

    If per_sample_sink is a list, append one dict per sample with per-sample
    metrics (l1, token_acc, n_tokens_correct, n_tokens_total, verb, etc.).
    """
    acc["total_ce_loss"] += output.loss.item()

    action_preds = output.logits[:, num_patches:-1].argmax(dim=2)
    action_gt = batch["labels"][:, 1:].to(action_preds.device)

    mask = ((action_tokenizer.action_token_end_idx > action_gt)
            & (action_gt > action_tokenizer.action_token_begin_idx))

    if mask.sum() == 0:
        return

    acc["n_batches_contributed"] += 1

    correct = (action_preds == action_gt) & mask
    acc["total_correct"] += correct.sum().item()
    acc["total_action_tokens"] += mask.sum().item()

    for si in range(mask.shape[0]):
        tok_positions = mask[si].nonzero(as_tuple=True)[0]
        if len(tok_positions) == 0:
            continue
        sample_correct = correct[si][tok_positions].cpu().numpy()
        for pi, c in enumerate(sample_correct):
            if pi < n_codes:
                acc["pos_correct"][pi] += int(c)
                acc["pos_total"][pi] += 1

    pred_ids = action_preds[mask].cpu().numpy()
    gt_ids = action_gt[mask].cpu().numpy()
    pred_chunks = action_tokenizer._decode_chunks(pred_ids)
    gt_chunks = action_tokenizer._decode_chunks(gt_ids)
    per_step = np.abs(pred_chunks - gt_chunks).mean(axis=-1)
    cumsum = np.cumsum(per_step, axis=1)
    for t in step_indices:
        if t < cumsum.shape[1]:
            acc["cumul_l1_sum"][t] += cumsum[:, t].sum().item()
            acc["cumul_l1_count"][t] += cumsum.shape[0]

    pred_actions = pred_chunks[:, 0].astype(np.float32)
    gt_actions = gt_chunks[:, 0].astype(np.float32)
    l1_diff = np.abs(pred_actions - gt_actions)
    acc["total_l1"] += l1_diff.sum().item()
    acc["total_l1_samples"] += pred_actions.size

    # ── Correct L1: decode(pred) vs RAW ground-truth actions ──────────────
    # The legacy `total_l1` and `cumul_l1_*` above compare decode(pred) to
    # decode(gt). That hides tokenizer reconstruction error: dense codebooks
    # (high bits/ts) make adjacent codes decode to similar chunks, so even
    # mispredictions look near-zero. The fix: compare decode(pred) directly
    # to the raw gt action chunk supplied by the dataloader.
    raw_actions = batch.get("raw_actions")
    if raw_actions is not None:
        sample_keep = mask.any(dim=1).cpu().numpy()
        raw_acts_np = raw_actions.detach().cpu().float().numpy()[sample_keep]
        # pred_chunks: (B', chunk_size, action_dim). raw_acts_np must match.
        if (raw_acts_np.ndim == 3 and pred_chunks.ndim == 3
                and raw_acts_np.shape[0] == pred_chunks.shape[0]
                and raw_acts_np.shape[2] == pred_chunks.shape[2]):
            # Align time dim: tokenizer-decoded chunk_size may differ from the
            # dataset chunk_size only on the "extra padded steps" tail; trim
            # both to the shorter length.
            T_pred = pred_chunks.shape[1]
            T_raw  = raw_acts_np.shape[1]
            T = min(T_pred, T_raw)
            per_step_raw = np.abs(
                pred_chunks[:, :T] - raw_acts_np[:, :T]
            ).mean(axis=-1)            # (B', T)
            cumsum_raw = np.cumsum(per_step_raw, axis=1)
            for t in step_indices:
                if t < cumsum_raw.shape[1]:
                    acc["cumul_l1_vs_raw_sum"][t] += cumsum_raw[:, t].sum().item()
                    acc["cumul_l1_vs_raw_count"][t] += cumsum_raw.shape[0]
            # Chunk-mean per-step L1 (averaged over time and dim).
            chunk_mean = per_step_raw.mean()
            acc["total_l1_vs_raw"] += chunk_mean * per_step_raw.size
            acc["total_l1_vs_raw_samples"] += per_step_raw.size
    if pred_actions.ndim == 2 and pred_actions.shape[1] == action_dim:
        acc["dim_l1_sum"] += l1_diff.sum(axis=0)
        acc["dim_l1_count"] += pred_actions.shape[0]

    B = len(instructions)
    for i in range(B):
        sample_mask = mask[i]
        if sample_mask.sum() == 0:
            continue
        verb = verb_lookup[instructions[i].strip()]
        n_correct_i = correct[i][sample_mask].sum().item()
        n_total_i = sample_mask.sum().item()
        acc["verb_correct"][verb] += n_correct_i
        acc["verb_total_tokens"][verb] += n_total_i
        sample_pred_ids = action_preds[i][sample_mask].cpu().numpy()
        sample_gt_ids = action_gt[i][sample_mask].cpu().numpy()
        sample_pred = np.atleast_2d(
            action_tokenizer.decode_token_ids_to_actions(sample_pred_ids)
        ).astype(np.float32)
        sample_gt = np.atleast_2d(
            action_tokenizer.decode_token_ids_to_actions(sample_gt_ids)
        ).astype(np.float32)
        sample_l1 = np.abs(sample_pred - sample_gt).mean().item()
        acc["verb_l1_sum"][verb] += sample_l1
        acc["verb_l1_count"][verb] += 1

        if per_sample_sink is not None:
            rec = {
                "batch_idx": batch_idx,
                "within_batch_idx": i,
                "instruction": instructions[i].strip(),
                "verb": verb,
                "l1": float(sample_l1),
                "n_tokens_correct": int(n_correct_i),
                "n_tokens_total": int(n_total_i),
                "token_acc": float(n_correct_i / max(n_total_i, 1)),
            }
            if n_verb_tokens_per_sample is not None \
                    and i < len(n_verb_tokens_per_sample):
                rec["n_verb_tokens_perturbed"] = \
                    int(n_verb_tokens_per_sample[i])
            per_sample_sink.append(rec)


def _summarize_accumulator(acc, n_codes, chunk_size, step_indices, action_dim):
    """Convert accumulator dict to a summary dict suitable for JSON output."""
    total_correct = acc["total_correct"]
    total_action_tokens = acc["total_action_tokens"]
    total_l1 = acc["total_l1"]
    total_l1_samples = acc["total_l1_samples"]
    n_contrib = max(acc["n_batches_contributed"], 1)

    token_acc = total_correct / max(total_action_tokens, 1) * 100
    l1 = total_l1 / max(total_l1_samples, 1)
    ce = acc["total_ce_loss"] / n_contrib

    per_position_acc = {}
    for p in range(n_codes):
        if acc["pos_total"][p] > 0:
            per_position_acc[str(p)] = float(
                acc["pos_correct"][p] / acc["pos_total"][p] * 100
            )

    per_step_l1 = {}
    for t in step_indices:
        if acc["cumul_l1_count"][t] > 0:
            per_step_l1[str(t)] = float(
                acc["cumul_l1_sum"][t] / acc["cumul_l1_count"][t]
            )

    per_dim_l1 = {}
    for d in range(action_dim):
        if acc["dim_l1_count"][d] > 0:
            per_dim_l1[ACTION_DIM_NAMES[d]] = float(
                acc["dim_l1_sum"][d] / acc["dim_l1_count"][d]
            )

    per_verb = {}
    for verb in sorted(acc["verb_l1_count"].keys(),
                       key=lambda v: -acc["verb_l1_count"][v]):
        n = acc["verb_l1_count"][verb]
        per_verb[verb] = {
            "l1": float(acc["verb_l1_sum"][verb] / n),
            "token_acc": float(
                acc["verb_correct"][verb]
                / max(acc["verb_total_tokens"][verb], 1) * 100
            ),
            "n_samples": int(n),
        }

    # decode(pred) vs RAW gt L1 — chunk-mean per-step L1 across all dims.
    l1_vs_raw = (acc["total_l1_vs_raw"]
                  / max(acc["total_l1_vs_raw_samples"], 1))
    per_step_l1_vs_raw = {}
    for t in step_indices:
        if acc["cumul_l1_vs_raw_count"][t] > 0:
            per_step_l1_vs_raw[str(t)] = float(
                acc["cumul_l1_vs_raw_sum"][t] / acc["cumul_l1_vs_raw_count"][t]
            )

    return {
        "ce_loss": ce,
        "token_accuracy": token_acc,
        "action_l1": l1,                       # legacy: decode(pred) vs decode(gt), step-0
        "action_l1_vs_raw": l1_vs_raw,         # decode(pred) vs RAW gt, chunk-mean per-step
        "n_action_tokens": int(total_action_tokens),
        "per_position_acc": per_position_acc,
        "per_step_l1": per_step_l1,                # cumulative, vs decode(gt)
        "per_step_l1_vs_raw": per_step_l1_vs_raw,  # cumulative, vs RAW gt
        "chunk_size": chunk_size,
        "n_codes_per_chunk": n_codes,
        "per_dim_l1": per_dim_l1,
        "per_verb": per_verb,
    }


def _build_batch_verb_mask(input_ids_cpu, instructions, tokenizer,
                           verb_spans_by_instruction):
    """Return (verb_mask, n_samples_perturbed, n_tokens_perturbed, n_not_located)."""
    B, S = input_ids_cpu.shape
    verb_mask = torch.zeros((B, S), dtype=torch.bool)
    n_perturbed = 0
    n_tokens = 0
    n_not_located = 0
    for i, ins in enumerate(instructions):
        span = verb_spans_by_instruction.get(ins.strip())
        if span is None:
            n_not_located += 1
            continue
        tok_idxs = _locate_verb_token_indices(input_ids_cpu[i], tokenizer, span)
        if not tok_idxs:
            n_not_located += 1
            continue
        for ti in tok_idxs:
            verb_mask[i, ti] = True
        n_perturbed += 1
        n_tokens += len(tok_idxs)
    return verb_mask, n_perturbed, n_tokens, n_not_located


def _locate_verb_token_indices(input_ids_row, tokenizer, verb_char_span,
                               marker=PROMPT_MARKER):
    """Return list of token indices in input_ids_row whose char offsets overlap
    the verb span. Uses simple per-token decode to compute char positions."""
    if verb_char_span is None:
        return []
    special_ids = set(tokenizer.all_special_ids or [])
    full_text = tokenizer.decode(input_ids_row, skip_special_tokens=True)
    marker_pos = full_text.find(marker)
    if marker_pos == -1:
        return []
    instr_start = marker_pos + len(marker)
    verb_s_full = instr_start + verb_char_span[0]
    verb_e_full = instr_start + verb_char_span[1]

    ids_list = input_ids_row.tolist() if hasattr(input_ids_row, "tolist") \
        else list(input_ids_row)

    result = []
    cumul_char = 0
    for i, tok_id in enumerate(ids_list):
        if tok_id in special_ids:
            continue
        # Decode incrementally: decode up to i+1, measure length
        prefix = tokenizer.decode([t for t in ids_list[:i+1] if t not in special_ids])
        tok_end = len(prefix)
        tok_start = cumul_char
        if tok_start < verb_e_full and tok_end > verb_s_full:
            result.append(i)
        cumul_char = tok_end
    return result


def teacher_force_eval(run_dir, condition, output_dir, num_batches=None,
                       batch_size=16, device="cuda", zero_proj=False,
                       perturb="none", perturb_seed=42,
                       output_filename="teacher_force_metrics.json",
                       save_per_sample=False):
    """Teacher-forced evaluation on the val set (train[95%:]).

    Runs forward passes with ground-truth input tokens (teacher forcing),
    computes argmax predictions at action-token positions, then decodes
    both predicted and GT token IDs back to continuous actions.

    Metrics:
        Primary:   CE loss, token accuracy, L1 loss (overall)
        Secondary: per-dimension L1 (7 action dims), per-verb L1
    """
    import torch.nn.functional as F
    from collections import defaultdict
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
    base_tokenizer = vla.llm_backbone.tokenizer

    print(f"  Action tokenizer: {type(action_tokenizer).__name__}")
    print(f"  action_token_begin_idx={action_tokenizer.action_token_begin_idx}, "
          f"action_token_end_idx={action_tokenizer.action_token_end_idx}")

    # Determine number of action tokens per sample
    n_codes = getattr(action_tokenizer, "n_codes_per_chunk", 7)

    # Zero-ablation: set projection weights to zero
    if zero_proj:
        embed_layer = vla.llm_backbone.llm.get_input_embeddings()
        if hasattr(embed_layer, 'proj') and embed_layer.proj is not None:
            print("  ZERO-ABLATION: zeroing proj.weight "
                  f"({list(embed_layer.proj.weight.shape)})")
            with torch.no_grad():
                embed_layer.proj.weight.zero_()
        else:
            print("  WARNING: --zero_proj requested but no proj layer found")

    # Load instruction→verb lookup from filtered CSV (same verbs as training)
    verb_lookup = _load_bridge_verb_lookup(C.BRIDGE_CSV)

    # Try to load precomputed val cache (fast path)
    cache_dir = Path(C.SWEEP_DIR) / "val_cache" / condition
    if cache_dir.is_dir() and list(cache_dir.glob("batch_*.pt")):
        print(f"  Loading precomputed val cache from {cache_dir}")
        cached_batches = sorted(cache_dir.glob("batch_*.pt"))
        print(f"  Found {len(cached_batches)} cached batches")

        def _cached_loader():
            for bp in cached_batches:
                yield torch.load(bp, map_location="cpu")

        val_loader = _cached_loader()
    else:
        print(f"  No val cache at {cache_dir}, building RLDS val dataset ...")
        print(f"  (Run policy/precompute_val.py --condition {condition} to cache)")
        # Build val dataset using the same config as training
        data_root_dir = Path(run_cfg.get("data_root_dir", C.RLDS_DIR))
        data_mix = vla_cfg.get("data_mix", "calvin_dataset")
        image_transform = vla.vision_backbone.get_image_transform()
        prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
        default_image_resolution = vla.vision_backbone.default_image_resolution

        future_action_window_size = vla_cfg.get("future_action_window_size", 0)
        future_action_window_size = max(
            action_tokenizer.required_future_horizon, future_action_window_size
        )

        # Use a large shuffle_buffer_size so .take(N).cache() covers full val.
        # Bridge val (train[95%:]) has ~100K frames; 200K ensures full coverage.
        val_dataset, _, collator = get_vla_dataset_and_collator(
            data_root_dir=data_root_dir,
            data_mix=data_mix,
            image_transform=image_transform,
            tokenizer=base_tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            default_image_resolution=default_image_resolution,
            predict_stop_token=vla_cfg.get("predict_stop_token", True),
            shuffle_buffer_size=200000,
            train=False,  # validation split
            image_aug=False,
            action_tokenizer=vla_cfg.get("action_tokenizer", "action_tokenizer"),
            future_action_window_size=future_action_window_size,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collator,
            num_workers=0, pin_memory=True,
        )

    num_patches = vla.vision_backbone.num_patches
    n_batches = 0

    # Step indices for cumulative per-timestep L1
    chunk_size = getattr(action_tokenizer, 'chunk_size',
                         getattr(action_tokenizer, 'horizon', 1))
    step_fractions = [0.0, 0.3, 0.5, 1.0]
    step_indices = [min(int(f * chunk_size), chunk_size - 1) if f < 1.0
                    else chunk_size - 1 for f in step_fractions]
    step_indices = list(dict.fromkeys(step_indices))
    action_dim = len(ACTION_DIM_NAMES)

    # Main accumulator (perturbed pass when verb_noise, else only pass)
    main_acc = _make_metrics_accumulator(n_codes, action_dim, step_indices)
    # Baseline accumulator: populated only when perturb=verb_noise; captures the
    # same batches' vanilla metrics for a fair comparison.
    baseline_acc = _make_metrics_accumulator(n_codes, action_dim, step_indices) \
        if perturb == "verb_noise" else None

    # Per-sample record sinks (optional). When perturb=verb_noise we populate
    # vanilla_log from the vanilla pass and noise_log from the perturbed pass,
    # then merge by (batch_idx, within_batch_idx) at save time.
    vanilla_log = [] if save_per_sample else None
    noise_log = [] if save_per_sample and perturb == "verb_noise" else None

    autocast_dtype = vla.llm_backbone.half_precision_dtype
    print(f"\n=== Teacher-Forced Eval (val split) ===")
    print(f"  Batch size: {batch_size}")
    print(f"  Tokens per sample: {n_codes}")
    if num_batches:
        print(f"  Max batches: {num_batches}")

    # ── Perturbation setup ─────────────────────────────────────────────────
    perturb_embed_layer = None
    verb_spans_by_instruction = None
    noise_sigma = None
    n_perturbed_samples = 0
    n_verb_not_located = 0
    total_verb_tokens_perturbed = 0
    if perturb == "verb_noise":
        print(f"  Perturbation: verb_noise (seed={perturb_seed})")
        nlp = _get_nlp()
        verb_spans_by_instruction, n_span_missed = \
            _build_instruction_verb_spans(C.BRIDGE_CSV, nlp)
        print(f"  Precomputed verb char spans for "
              f"{len(verb_spans_by_instruction)} instructions "
              f"(missed {n_span_missed})")
        perturb_embed_layer = vla.llm_backbone.llm.get_input_embeddings()
        with torch.no_grad():
            emb_weight = perturb_embed_layer.weight \
                if hasattr(perturb_embed_layer, "weight") \
                else next(perturb_embed_layer.parameters())
            noise_sigma = float(emb_weight.float().std().item())
        print(f"  Noise sigma (std of input-embedding matrix): {noise_sigma:.6f}")
        torch.manual_seed(perturb_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(perturb_seed)

    def to_device(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {kk: to_device(vv, device) for kk, vv in v.items()}
        return v

    import time as _time
    _eval_t0 = _time.time()

    n_skipped = 0  # samples skipped (instruction not in filtered CSV)

    for batch in val_loader:
        # ── Filter: keep only samples whose instruction is in the CSV ──
        instructions = _extract_instructions_from_input_ids(
            batch["input_ids"], base_tokenizer
        )
        keep = [i for i, ins in enumerate(instructions)
                if verb_lookup.get(ins.strip())]
        n_skipped += len(instructions) - len(keep)
        if not keep:
            n_batches += 1
            continue

        # Subset the batch to annotated samples only
        keep_idx = torch.tensor(keep, dtype=torch.long)

        def _subset(v, idx, keep_list):
            if isinstance(v, torch.Tensor):
                return v[idx]
            if isinstance(v, dict):
                return {kk: _subset(vv, idx, keep_list) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [v[i] for i in keep_list]
            return v

        batch = {
            k: _subset(v, keep_idx, keep)
            for k, v in batch.items()
        }
        instructions = [instructions[i] for i in keep]
        batch = {k: to_device(v, device) for k, v in batch.items()}

        # Dynamic embedding injection for fullproj OAT/QueST.
        # Use prefix-conditioned embeddings (leak-free) — see
        # lab_notebooks/compression_rate_sweep/oat_tice_leakage.md. Set
        # OAT_TICE_LEAKY_EMBED=1 to revert to the legacy `encode_pre_fsq`
        # path for ablation against an old-style policy ckpt.
        embed_layer = vla.llm_backbone.llm.get_input_embeddings()
        if (hasattr(embed_layer, 'dynamic') and embed_layer.dynamic
                and "raw_actions" in batch):
            with torch.no_grad():
                raw_acts = batch["raw_actions"].cpu().float()
                if os.environ.get("OAT_TICE_LEAKY_EMBED", "0") == "1":
                    latents = action_tokenizer.encode_pre_fsq(raw_acts)
                elif hasattr(action_tokenizer, "compute_prefix_embeddings"):
                    latents = action_tokenizer.compute_prefix_embeddings(raw_acts)
                else:
                    latents = action_tokenizer.encode_pre_fsq(raw_acts)
                latents = latents.to(device)
            embed_layer.set_current_latents(latents)

        # ── Verb-noise: compute verb mask on CPU once per batch ─────────────
        verb_mask_dev = None
        if perturb == "verb_noise":
            input_ids_cpu = batch["input_ids"].detach().cpu()
            verb_mask, bp, bt, bnl = _build_batch_verb_mask(
                input_ids_cpu, instructions, base_tokenizer,
                verb_spans_by_instruction,
            )
            n_perturbed_samples += bp
            total_verb_tokens_perturbed += bt
            n_verb_not_located += bnl
            verb_mask_dev = verb_mask.to(device)

        # ── Vanilla forward pass (no hook) ─────────────────────────────────
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=autocast_dtype,
            enabled=vla.enable_mixed_precision_training,
        ):
            output_vanilla = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

        if perturb == "verb_noise":
            # Save vanilla metrics for this batch into baseline accumulator
            _accumulate_batch_metrics(
                output_vanilla, batch, instructions, baseline_acc,
                action_tokenizer, num_patches, n_codes, step_indices,
                action_dim, verb_lookup,
                per_sample_sink=vanilla_log, batch_idx=n_batches,
            )

            # ── Perturbed forward pass (with noise hook) ────────────────────
            def _noise_hook(module, inputs, output):
                if output.dim() != 3:
                    return output
                mask = verb_mask_dev
                if mask.shape[0] != output.shape[0] \
                        or mask.shape[1] != output.shape[1]:
                    return output
                noise = torch.randn_like(output, dtype=torch.float32) \
                    * noise_sigma
                noise = noise.to(output.dtype)
                return torch.where(mask.unsqueeze(-1), noise, output)

            hook_handle = embed_layer.register_forward_hook(_noise_hook)
            try:
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=autocast_dtype,
                    enabled=vla.enable_mixed_precision_training,
                ):
                    output_main = vla(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
            finally:
                hook_handle.remove()
        else:
            output_main = output_vanilla

        # For dual-pass (verb_noise), main_acc is the perturbed pass and
        # records go to noise_log with per-sample verb-token counts.
        # For single-pass (none), main_acc is the only pass and records go to
        # vanilla_log.
        main_sink = noise_log if perturb == "verb_noise" else vanilla_log
        n_verb_tokens_list = None
        if perturb == "verb_noise" and verb_mask_dev is not None:
            n_verb_tokens_list = verb_mask_dev.sum(dim=1).cpu().tolist()
        _accumulate_batch_metrics(
            output_main, batch, instructions, main_acc,
            action_tokenizer, num_patches, n_codes, step_indices,
            action_dim, verb_lookup,
            per_sample_sink=main_sink, batch_idx=n_batches,
            n_verb_tokens_per_sample=n_verb_tokens_list,
        )

        n_batches += 1
        if n_batches % 50 == 0:
            elapsed = _time.time() - _eval_t0
            running_acc = main_acc["total_correct"] / \
                max(main_acc["total_action_tokens"], 1) * 100
            running_l1 = main_acc["total_l1"] / \
                max(main_acc["total_l1_samples"], 1)
            running_ce = main_acc["total_ce_loss"] / \
                max(main_acc["n_batches_contributed"], 1)
            print(f"  [{n_batches:>5d} batches] "
                  f"CE={running_ce:.4f}  TokAcc={running_acc:.1f}%  "
                  f"L1={running_l1:.4f}  skipped={n_skipped}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

        if num_batches and n_batches >= num_batches:
            print(f"  Reached max batches ({num_batches}), stopping.")
            break

    print(f"  Skipped {n_skipped} samples (instruction not in filtered CSV)")

    # ── Summarize main accumulator ─────────────────────────────────────────
    main_summary = _summarize_accumulator(main_acc, n_codes, chunk_size,
                                          step_indices, action_dim)
    print(f"\n  Results ({n_batches} batches, "
          f"{main_summary['n_action_tokens']} action tokens):")
    print(f"    CE Loss:        {main_summary['ce_loss']:.4f}")
    print(f"    Token Accuracy: {main_summary['token_accuracy']:.2f}%")
    print(f"    Action L1:      {main_summary['action_l1']:.4f}")

    if main_summary["per_position_acc"]:
        print(f"\n  Per-token-position accuracy ({n_codes} codes/chunk):")
        for p in range(n_codes):
            key = str(p)
            if key in main_summary["per_position_acc"]:
                print(f"    token {p}: "
                      f"{main_summary['per_position_acc'][key]:.1f}%")

    if main_summary["per_step_l1"]:
        print(f"\n  Cumulative L1 (chunk_size={chunk_size}):")
        for t in step_indices:
            key = str(t)
            if key in main_summary["per_step_l1"]:
                frac = t / max(chunk_size - 1, 1)
                print(f"    step 0-{t:>2d} ({frac:>4.0%}): "
                      f"cumL1={main_summary['per_step_l1'][key]:.5f}")

    if main_summary["per_dim_l1"]:
        print(f"\n  Per-dimension L1:")
        for d in range(action_dim):
            name = ACTION_DIM_NAMES[d]
            if name in main_summary["per_dim_l1"]:
                print(f"    {name:>8s}: "
                      f"{main_summary['per_dim_l1'][name]:.4f}")

    if main_summary["per_verb"]:
        print(f"\n  Per-verb L1 ({len(main_summary['per_verb'])} verbs):")
        for verb, m in main_summary["per_verb"].items():
            print(f"    {verb:>12s}: L1={m['l1']:.4f}  "
                  f"TokAcc={m['token_acc']:.1f}%  (n={m['n_samples']})")

    # ── Assemble final results dict ────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "condition": condition,
        "mode": "teacher_force",
        "perturb": perturb,
        "n_batches": n_batches,
        "n_skipped_unannotated": n_skipped,
    }
    # Flatten main summary into top-level keys (backward compatibility with
    # previous schema: ce_loss, token_accuracy, action_l1, per_verb, etc.)
    for k, v in main_summary.items():
        results[k] = v

    if perturb == "verb_noise":
        baseline_summary = _summarize_accumulator(
            baseline_acc, n_codes, chunk_size, step_indices, action_dim
        )
        print(f"\n  Baseline (vanilla on same batches):")
        print(f"    CE Loss:        {baseline_summary['ce_loss']:.4f}")
        print(f"    Token Accuracy: {baseline_summary['token_accuracy']:.2f}%")
        print(f"    Action L1:      {baseline_summary['action_l1']:.4f}")
        if baseline_summary["per_verb"]:
            print(f"\n  Baseline per-verb L1:")
            for verb, m in baseline_summary["per_verb"].items():
                print(f"    {verb:>12s}: L1={m['l1']:.4f}  "
                      f"TokAcc={m['token_acc']:.1f}%  (n={m['n_samples']})")

        results["baseline"] = baseline_summary
        results["perturb_seed"] = perturb_seed
        results["noise_sigma"] = noise_sigma
        results["n_perturbed_samples"] = n_perturbed_samples
        results["n_verb_not_located"] = n_verb_not_located
        results["total_verb_tokens_perturbed"] = total_verb_tokens_perturbed
        print(f"\n  Perturbation stats:")
        print(f"    samples perturbed: {n_perturbed_samples}")
        print(f"    verb not located: {n_verb_not_located}")
        print(f"    total verb tokens perturbed: "
              f"{total_verb_tokens_perturbed}")

    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Optionally save per-sample records as JSONL
    if save_per_sample and vanilla_log is not None:
        per_sample_filename = output_filename.replace(
            ".json", "_per_sample.jsonl"
        )
        if per_sample_filename == output_filename:
            per_sample_filename = output_filename + "_per_sample.jsonl"
        ps_path = os.path.join(output_dir, per_sample_filename)

        if perturb == "verb_noise" and noise_log is not None:
            # Merge vanilla and noise records by (batch_idx, within_batch_idx)
            noise_idx = {
                (r["batch_idx"], r["within_batch_idx"]): r for r in noise_log
            }
            with open(ps_path, "w") as f:
                for v in vanilla_log:
                    key = (v["batch_idx"], v["within_batch_idx"])
                    n = noise_idx.get(key)
                    rec = {
                        "batch_idx": v["batch_idx"],
                        "within_batch_idx": v["within_batch_idx"],
                        "instruction": v["instruction"],
                        "verb": v["verb"],
                        "vanilla_l1": v["l1"],
                        "vanilla_token_acc": v["token_acc"],
                        "vanilla_n_tokens_total": v["n_tokens_total"],
                    }
                    if n is not None:
                        rec["noise_l1"] = n["l1"]
                        rec["noise_token_acc"] = n["token_acc"]
                        rec["n_verb_tokens_perturbed"] = \
                            n.get("n_verb_tokens_perturbed", 0)
                    f.write(json.dumps(rec) + "\n")
        else:
            with open(ps_path, "w") as f:
                for v in vanilla_log:
                    f.write(json.dumps(v) + "\n")
        print(f"  Saved {len(vanilla_log)} per-sample records to {ps_path}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MiniVLA policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--condition",
        help="Condition tag (e.g. vq_bet_5_16_2_256, quest_16_855_4_clip0.1). "
             "Resolves to checkpoints/<dataset>_sweep/policy/minivla_<condition>/",
    )
    parser.add_argument(
        "--dataset", default="bridge", choices=["bridge", "calvin"],
        help="Which sweep to resolve --condition against (default: bridge)",
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
    parser.add_argument(
        "--zero_proj", action="store_true",
        help="Zero-ablation: set projection weights to zero before eval. "
             "Tests whether the VLA relies on the projected latent.",
    )
    parser.add_argument(
        "--perturb", default="none", choices=["none", "verb_noise"],
        help="Instruction perturbation. 'verb_noise' replaces the GT verb's "
             "token embedding(s) with Gaussian noise (sigma = std of "
             "input-embedding matrix) before the LLM forward pass.",
    )
    parser.add_argument("--perturb_seed", type=int, default=42)
    parser.add_argument(
        "--output_subdir",
        help="Override the subdirectory name under --output_dir for results. "
             "Default: {condition} when --condition is given, or "
             "{parent}_{leaf} when --policy_dir is given (to avoid collisions "
             "across models with identically-named variants like 'vanilla').",
    )
    parser.add_argument(
        "--save_per_sample", action="store_true",
        help="In addition to the aggregate JSON, save one record per sample "
             "(instruction, verb, l1, token_acc) as a JSONL file alongside. "
             "For --perturb verb_noise, each row has both vanilla and noise "
             "metrics so per-sample Δ can be computed.",
    )
    args = parser.parse_args()

    if not args.condition and not args.policy_dir:
        parser.error("Provide --condition or --policy_dir")

    # Resolve policy directory
    if args.policy_dir:
        policy_dir = args.policy_dir
        pd_parts = Path(policy_dir).parts
        # Disambiguate leaf (e.g. "vanilla") with its parent so different
        # models don't collide when writing results.
        if len(pd_parts) >= 2:
            condition = f"{pd_parts[-2]}_{pd_parts[-1]}".replace("minivla_", "")
        else:
            condition = Path(policy_dir).name.replace("minivla_", "")
    else:
        # Resolve based on dataset
        sweep_name = "{}_sweep".format(args.dataset)
        pol_dir = PROJECT_DIR / "checkpoints" / sweep_name / "policy" / "minivla_{}".format(args.condition)
        if not pol_dir.exists():
            # Fall back to config.py default
            pol_dir = Path(resolve_policy_dir(args.condition))
        policy_dir = str(pol_dir)
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
        if args.perturb == "verb_noise":
            # Disambiguate same-named variants (e.g. .../vq_bet_.../vanilla and
            # .../oat_.../vanilla) by including the parent dir in the tag.
            if args.policy_dir:
                pd_parts = Path(args.policy_dir).parts
                if len(pd_parts) >= 2:
                    tag = f"{pd_parts[-2]}__{pd_parts[-1]}"
                else:
                    tag = Path(args.policy_dir).name
            else:
                tag = args.condition.replace("/", "__")
            out_dir = os.path.join(args.output_dir, "bridge_perturb")
            out_filename = f"verb_noise_{tag}.json"
            condition_label = tag
        else:
            subdir = args.output_subdir or condition
            out_dir = os.path.join(args.output_dir, subdir)
            if args.zero_proj:
                out_dir += "_zero_proj"
            out_filename = "teacher_force_metrics.json"
            condition_label = subdir
        teacher_force_eval(
            policy_dir, condition_label, out_dir,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=args.device,
            zero_proj=args.zero_proj,
            perturb=args.perturb,
            perturb_seed=args.perturb_seed,
            output_filename=out_filename,
            save_per_sample=args.save_per_sample,
        )


if __name__ == "__main__":
    main()
