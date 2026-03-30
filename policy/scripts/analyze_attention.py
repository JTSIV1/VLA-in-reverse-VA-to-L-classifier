"""
analyze_attention.py

Analyze attention from the last action token to verb tokens in fine-tuned MiniVLA models.

For each CALVIN val sample (via the RLDS dataloader, matching training):
  1. Run teacher-forced forward pass with output_attentions=True (batch_size=1)
  2. Identify token positions: image patches, prompt text, verb sub-tokens, action tokens
  3. Extract attention from the LAST action token to all other positions
  4. Compute verb_attn_ratio = attn(verb tokens) / attn(all text tokens), per layer per head

Design:
  - Source position: last action token only (where the next-token prediction head is applied)
  - Target groups: verb tokens, all text tokens, image patches, other action tokens
  - Metric: verb_attn_ratio = sum(attn on verb tokens) / sum(attn on all text tokens)
  - Granularity: recorded per layer (24) and per head (14) — no averaging

Architecture (MiniVLA / Qwen2.5-0.5B):
  - 24 layers, 14 attention heads, 514 image patches (257 DINOv2 + 257 SigLIP)

Output: results/attention_analysis/attention_{condition}.json

Usage:
  python -m policy.scripts.analyze_attention --condition quest_16_4444_2
  python -m policy.scripts.analyze_attention --condition quest_16_4444_2_verb0.1
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as C  # noqa: E402

OPENVLA_DIR = C.OPENVLA_DIR
if OPENVLA_DIR not in sys.path:
    sys.path.insert(0, OPENVLA_DIR)


def aggregate_attention(
    attentions,           # tuple of (batch, heads, seq_len, seq_len), one per layer
    last_action_pos: int,         # position of the last action token
    verb_positions:   list[int],  # positions of verb sub-tokens
    text_positions:   list[int],  # positions of ALL text tokens (prompt + action, no image)
    image_positions:  list[int],  # positions of image patch tokens
    action_positions: list[int],  # positions of OTHER action tokens (excluding last)
) -> dict:
    """
    Extract attention from the last action token to different token groups.

    All outputs are [n_layers][n_heads] matrices (no averaging across layers/heads).

    Attention groups (from the last action token's perspective):
        verb:   sub-tokens of the primary verb in the instruction
        text:   all text tokens (prompt + action, no image patches)
        image:  image patch tokens
        action: other action tokens (excluding the last one itself)

    Ratios are relative to the full sequence (verb/text/image/action all sum to ~1.0):
        verb_attn_ratio  = sum(attn on verb)  / sum(attn on all text)
        image_attn_sum   = sum(attn on image patches)
        action_attn_sum  = sum(attn on other action tokens)
    """
    v_idx = torch.tensor(verb_positions, dtype=torch.long)
    t_idx = torch.tensor(text_positions, dtype=torch.long)
    img_idx = torch.tensor(image_positions, dtype=torch.long)
    act_idx = torch.tensor(action_positions, dtype=torch.long) if action_positions else None

    verb_attn_ratio = []
    verb_attn_raw = []
    text_attn_raw = []
    image_attn_raw = []
    action_attn_raw = []

    for layer_idx in range(len(attentions)):
        # (1, n_heads, seq_len, seq_len) → (n_heads, seq_len)
        attn = attentions[layer_idx][0, :, last_action_pos, :].float()  # (n_heads, seq_len)

        verb_attn = attn[:, v_idx].sum(dim=1)   # (n_heads,)
        text_attn = attn[:, t_idx].sum(dim=1)   # (n_heads,)
        img_attn  = attn[:, img_idx].sum(dim=1)  # (n_heads,)

        if act_idx is not None and len(act_idx) > 0:
            act_attn = attn[:, act_idx].sum(dim=1)  # (n_heads,)
        else:
            act_attn = torch.zeros(attn.shape[0])

        ratio = (verb_attn / text_attn.clamp(min=1e-12))  # (n_heads,)

        verb_attn_ratio.append(ratio.tolist())
        verb_attn_raw.append(verb_attn.tolist())
        text_attn_raw.append(text_attn.tolist())
        image_attn_raw.append(img_attn.tolist())
        action_attn_raw.append(act_attn.tolist())

    return {
        "verb_attn_ratio": verb_attn_ratio,   # [n_layers][n_heads]
        "verb_attn_raw":   verb_attn_raw,      # [n_layers][n_heads]
        "text_attn_raw":   text_attn_raw,      # [n_layers][n_heads]
        "image_attn_raw":  image_attn_raw,     # [n_layers][n_heads]
        "action_attn_raw": action_attn_raw,    # [n_layers][n_heads]
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_subsequence(haystack, needle):
    """Return start index of first occurrence of needle in haystack, or -1."""
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def find_verb_in_input_ids(input_ids_list, verb_text, tokenizer):
    """Find the position of verb sub-tokens within input_ids.

    Tries multiple tokenization strategies for the verb phrase.
    Returns (start_pos, n_verb_tokens) or (-1, 0) if not found.
    """
    # Strategy 1: verb with leading space (most common in sentence context)
    verb_tok_ids = tokenizer.encode(" " + verb_text, add_special_tokens=False)
    pos = find_subsequence(input_ids_list, verb_tok_ids)
    if pos >= 0:
        return pos, len(verb_tok_ids)

    # Strategy 2: verb without leading space
    verb_tok_ids = tokenizer.encode(verb_text, add_special_tokens=False)
    pos = find_subsequence(input_ids_list, verb_tok_ids)
    if pos >= 0:
        return pos, len(verb_tok_ids)

    return -1, 0


def find_last_action_pos(labels):
    """Find the position of the last action token from labels.

    Action tokens have labels != -100. Returns the index of the last one.
    """
    action_mask = (labels != -100)
    action_indices = action_mask.nonzero(as_tuple=True)[0]
    if len(action_indices) == 0:
        return -1
    return action_indices[-1].item()


def find_action_range(labels):
    """Find (first_action_idx, last_action_idx) from labels.

    Action tokens have labels != -100.
    """
    action_mask = (labels != -100)
    action_indices = action_mask.nonzero(as_tuple=True)[0]
    if len(action_indices) == 0:
        return -1, -1
    return action_indices[0].item(), action_indices[-1].item()


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_attention_analysis(
    condition: str,
    output_dir: str,
    max_examples: int,
    device: str,
) -> None:
    from torch.utils.data import DataLoader
    from prismatic.models import load_vla
    from prismatic.vla.materialize import get_vla_dataset_and_collator

    # ── Resolve policy directory ──────────────────────────────────────────────
    policy_dir = Path(C.POLICY_DIR) / "minivla_{}".format(condition)
    if not policy_dir.exists():
        raise FileNotFoundError("Policy directory not found: {}".format(policy_dir))

    run_dir = policy_dir
    config_json = run_dir / "config.json"
    with open(config_json) as f:
        run_cfg = json.load(f)
    vla_cfg = run_cfg["vla"]

    # ── Find checkpoint ──────────────────────────────────────────────────────
    import re
    ckpt_dir = run_dir / "checkpoints"
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError("No .pt checkpoint in {}".format(ckpt_dir))

    def step_num(p):
        m = re.search(r"step-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    fsdp_ckpt = str(max(candidates, key=step_num))
    print("Loading VLA from {} ...".format(fsdp_ckpt))

    # ── Load model ───────────────────────────────────────────────────────────
    vla = load_vla(fsdp_ckpt, load_for_training=False)
    vla = vla.to(device).eval()
    action_tokenizer = vla.action_tokenizer
    num_patches = vla.vision_backbone.num_patches
    print("Model loaded. num_patches={}, action_tokenizer={}".format(
        num_patches, type(action_tokenizer).__name__))

    # ── Build val dataset (same config as training) ──────────────────────────
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
        train=False,
        image_aug=False,
        action_tokenizer=vla_cfg.get("action_tokenizer", "action_tokenizer"),
        future_action_window_size=future_action_window_size,
    )

    # batch_size=1 for attention analysis (variable seq lengths)
    val_loader = DataLoader(
        val_dataset, batch_size=1, collate_fn=collator,
        num_workers=2, pin_memory=True,
    )

    # ── Load verb labels from raw CALVIN annotations ─────────────────────────
    from utils import load_calvin_to_dataframe

    train_df = load_calvin_to_dataframe(C.TRAIN_DIR)
    verb_counts = train_df["primary_verb"].value_counts()
    keep_verbs = set(verb_counts[verb_counts >= 30].index)
    sorted_verbs = sorted(keep_verbs)
    verb_to_id = {v: i for i, v in enumerate(sorted_verbs)}

    val_df = load_calvin_to_dataframe(C.VAL_DIR)
    val_df = val_df[val_df["primary_verb"].isin(keep_verbs)].reset_index(drop=True)
    # Build instruction → verb mapping for lookup
    instr_to_verb = {}
    for _, row in val_df.iterrows():
        instr_to_verb[row["instruction"].lower().strip()] = row["primary_verb"]
    print("Verb mapping: {} unique instructions, {} verb classes".format(
        len(instr_to_verb), len(verb_to_id)))

    # ── Run attention analysis ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results = []
    skipped = 0
    autocast_dtype = vla.llm_backbone.half_precision_dtype

    def to_device(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, dict):
            return {kk: to_device(vv, device) for kk, vv in v.items()}
        return v

    print("\n=== Attention Analysis (condition={}) ===".format(condition))
    if max_examples > 0:
        print("  Max examples: {}".format(max_examples))

    # Track seen input_ids to detect when the RLDS streaming dataset loops
    seen_hashes = set()
    stale_count = 0
    STALE_LIMIT = 50  # break if 50 consecutive duplicates (dataset looped)

    for batch in val_loader:
        if max_examples > 0 and (len(results) + skipped) >= max_examples:
            break

        # Detect dataset loop: hash input_ids to check for repeats
        batch_hash = hash(batch["input_ids"][0].cpu().numpy().tobytes())
        if batch_hash in seen_hashes:
            stale_count += 1
            if stale_count >= STALE_LIMIT:
                print("  Detected dataset loop ({} consecutive duplicates), stopping.".format(
                    STALE_LIMIT))
                break
            continue
        else:
            seen_hashes.add(batch_hash)
            stale_count = 0

        batch = {k: to_device(v, device) for k, v in batch.items()}
        input_ids = batch["input_ids"][0]  # (seq_len,)
        labels = batch["labels"][0]        # (seq_len,)

        # ── Extract instruction text from input_ids ──────────────────────────
        # Decode the prompt portion (non-action tokens) to find the instruction
        first_action, last_action = find_action_range(labels)
        if first_action < 0:
            skipped += 1
            continue

        # Decode prompt tokens to recover instruction text
        prompt_ids = input_ids[:first_action].cpu().tolist()
        prompt_text = base_tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Extract the instruction from "What action should the robot take to {instruction}?"
        marker = "what action should the robot take to "
        lower_text = prompt_text.lower()
        marker_pos = lower_text.find(marker)
        if marker_pos < 0:
            skipped += 1
            continue

        instruction = lower_text[marker_pos + len(marker):].rstrip("?").strip()

        # Look up verb from instruction
        primary_verb = instr_to_verb.get(instruction)
        if primary_verb is None:
            # Try fuzzy match: check if instruction starts with any known instruction
            for known_instr, verb in instr_to_verb.items():
                if instruction.startswith(known_instr) or known_instr.startswith(instruction):
                    primary_verb = verb
                    break
        if primary_verb is None or primary_verb not in verb_to_id:
            skipped += 1
            continue

        verb_id = verb_to_id[primary_verb]

        # ── Find verb token positions in input_ids ───────────────────────────
        input_ids_list = input_ids.cpu().tolist()
        verb_start, n_verb_tokens = find_verb_in_input_ids(
            input_ids_list, primary_verb, base_tokenizer
        )
        if verb_start < 0:
            skipped += 1
            continue

        # ── Forward pass with output_attentions ──────────────────────────────
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=autocast_dtype,
            enabled=vla.enable_mixed_precision_training,
        ):
            output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_attentions=True,
            )

        if output.attentions is None:
            raise RuntimeError(
                "output.attentions is None — model may be using flash attention. "
                "Ensure load_for_training=False disables flash attention."
            )

        attentions_cpu = tuple(a.cpu() for a in output.attentions)
        del output
        torch.cuda.empty_cache()

        # ── Identify sequence positions ──────────────────────────────────────
        # Full sequence in attention: [image patches (0..num_patches-1)] [text tokens (num_patches..)]
        seq_len = attentions_cpu[0].shape[-1]
        n_text = seq_len - num_patches

        # Last action token position in the full sequence
        # labels are aligned with input_ids; action tokens have labels != -100
        # In attention space, text token i maps to position num_patches + i
        last_action_in_text = last_action  # index within input_ids
        last_action_pos = num_patches + last_action_in_text

        # Verb positions in full sequence
        verb_positions = [num_patches + verb_start + k for k in range(n_verb_tokens)]

        # All text token positions (prompt + action, excluding image patches)
        text_positions = list(range(num_patches, seq_len))

        # Image patch positions
        image_positions = list(range(num_patches))

        # Other action token positions (excluding the last action token)
        first_action_pos = num_patches + first_action
        other_action_positions = list(range(first_action_pos, last_action_pos))

        # ── Aggregate attention ──────────────────────────────────────────────
        agg = aggregate_attention(
            attentions_cpu,
            last_action_pos=last_action_pos,
            verb_positions=verb_positions,
            text_positions=text_positions,
            image_positions=image_positions,
            action_positions=other_action_positions,
        )

        n_action_tokens = last_action - first_action + 1
        record = {
            "idx":              len(results),
            "verb_id":          verb_id,
            "primary_verb":     primary_verb,
            "instruction":      instruction,
            "n_prompt_tokens":  first_action,
            "n_action_tokens":  n_action_tokens,
            "n_verb_tokens":    n_verb_tokens,
            "seq_len":          seq_len,
            **agg,
        }
        results.append(record)

        if len(results) % 20 == 0:
            all_ratios = [np.mean(r["verb_attn_ratio"]) for r in results]
            print("  [{}/{}] avg verb_attn_ratio={:.4f}  (skipped {})".format(
                len(results), max_examples if max_examples > 0 else "all",
                np.mean(all_ratios), skipped))

    print("\nDone: {} records, {} skipped".format(len(results), skipped))

    if len(results) == 0:
        print("No results to save.")
        return

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "attention_{}.json".format(condition))
    with open(out_path, "w") as f:
        json.dump({"condition": condition, "n_results": len(results),
                    "results": results}, f, indent=2)
    print("Saved to {}".format(out_path))

    # ── Print summary ────────────────────────────────────────────────────────
    from collections import defaultdict
    verb_ratio_by_class = defaultdict(list)
    for r in results:
        verb_ratio_by_class[r["primary_verb"]].append(np.mean(r["verb_attn_ratio"]))

    n_layers = len(results[0]["verb_attn_ratio"])
    n_heads  = len(results[0]["verb_attn_ratio"][0])
    print("\nVerb attention ratio (last action token -> verb / all text), "
          "{} layers x {} heads:".format(n_layers, n_heads))
    print("  (below: averaged over all layers and heads)")
    for verb in sorted(verb_ratio_by_class):
        vals = verb_ratio_by_class[verb]
        print("  {:<20s} n={:3d}  mean={:.4f}  std={:.4f}".format(
            verb, len(vals), np.mean(vals), np.std(vals)))

    overall_ratio = np.mean([np.mean(r["verb_attn_ratio"]) for r in results])
    print("\nOverall mean verb_attn_ratio: {:.4f}".format(overall_ratio))

    # Per-layer summary
    per_layer_mean = np.mean(
        [np.mean(r["verb_attn_ratio"], axis=1) for r in results], axis=0
    )
    print("\nPer-layer verb_attn_ratio (averaged over heads and examples):")
    for l_idx, val in enumerate(per_layer_mean):
        print("  Layer {:2d}: {:.4f}".format(l_idx, val))

    # Attention budget summary (where does the last action token's attention go?)
    mean_img = np.mean([np.mean(r["image_attn_raw"]) for r in results])
    mean_txt = np.mean([np.mean(r["text_attn_raw"]) for r in results])
    mean_act = np.mean([np.mean(r["action_attn_raw"]) for r in results])
    mean_vrb = np.mean([np.mean(r["verb_attn_raw"]) for r in results])
    print("\nAttention budget (last action token, averaged over layers/heads/examples):")
    print("  image:  {:.4f}".format(mean_img))
    print("  text:   {:.4f}  (of which verb: {:.4f})".format(mean_txt, mean_vrb))
    print("  action: {:.4f}".format(mean_act))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Attention analysis for fine-tuned MiniVLA")
    p.add_argument("--condition", required=True,
                   help="Condition tag (e.g. quest_16_4444_2, quest_16_4444_2_verb0.1)")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "attention_analysis"))
    p.add_argument("--max_examples", type=int, default=2500,
                   help="Max val examples (results + skipped) before stopping. "
                        "RLDS streaming datasets may hang at exhaustion; set this "
                        "above the expected total to ensure clean exit.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    run_attention_analysis(
        condition=args.condition,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
