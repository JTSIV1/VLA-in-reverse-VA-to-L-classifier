"""
analyze_attention.py

Analyze attention from action tokens to verb tokens in fine-tuned OpenVLA models.

For each CALVIN val trajectory (with known verb label):
  1. Load the first frame (rgb_static) + instruction text from raw CALVIN .npz files
  2. Build teacher-forcing input: image + prompt + ground-truth action tokens
  3. Run forward pass with output_attentions=True  (attn_implementation="eager")
  4. Extract attention weights for the last 16 layers (layers 16-31)
  5. Aggregate: mean over layers, heads, action positions, and verb token sub-tokens

Aggregation strategy (agreed design):
  - Layers:          mean over last-half layers (16-31); per-layer breakdown saved
  - Heads:           mean over all 32 heads
  - Action positions: mean over all action token positions
  - Verb positions:  mean over all verb sub-tokens (e.g. "pick up" → 2 tokens)
  - Normalization:   raw_attn × n_text_tokens  (relative to uniform attention over TEXT only;
                     avoids dilution from ~256 image patch tokens)
  - Contrast:        verb_attn_normed − mean(non-verb instruction token attn normed)

Output per condition: results/attention_analysis/{condition}/attention_results.json
  Each record: {verb_id, primary_verb, instruction, n_action_tokens, n_verb_tokens,
                seq_len, verb_attn_raw, verb_attn_normed, verb_attn_contrast,
                instr_attn_normed, per_layer_raw [32 floats]}

Usage:
  python -m openvla_experiment.scripts.analyze_attention \\
      --condition bin \\
      --checkpoint_dir runs/openvla/openvla-7b+...--calvin_bin--image_aug \\
      --output_dir results/attention_analysis \\
      --max_examples 300

  # VQ conditions also need the VQ tokenizer checkpoint:
  python -m openvla_experiment.scripts.analyze_attention \\
      --condition vq_verb \\
      --checkpoint_dir runs/openvla/...--calvin_vq_verb--image_aug \\
      --vqvla_checkpoint_dir checkpoints/vqvla_ft_verb_l0.5 \\
      --output_dir results/attention_analysis \\
      --max_examples 300
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
OPENVLA_DIR  = str(Path("/data/user_data/wenjiel2/Code/openvla-mini"))
for p in [PROJECT_ROOT, OPENVLA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ──────────────────────────────────────────────────────────────────
CALVIN_VAL_DIR  = "/data/user_data/yashagar/task_D_D/validation"
CALVIN_TRAIN_DIR = "/data/user_data/yashagar/task_D_D/training"

# Last-half layers: layers 16–31 of a 32-layer LLaMA-based model
ATTN_LAYER_START = 16
ATTN_LAYER_END   = 32   # exclusive

# Prompt template (must match training: PurePromptBuilder.wrap_human)
PROMPT_TEMPLATE = "In: {instruction}\nOut: "


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return start index of first occurrence of needle in haystack, or -1."""
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def load_trajectory_data(data_dir: str, start_idx: int, end_idx: int):
    """Load rgb_static (first frame) and rel_actions for a CALVIN trajectory."""
    first_path = os.path.join(data_dir, f"episode_{start_idx:07d}.npz")
    rgb = np.load(first_path)["rgb_static"]              # (200, 200, 3), uint8

    actions = []
    for i in range(start_idx, end_idx + 1):
        ep = np.load(os.path.join(data_dir, f"episode_{i:07d}.npz"))
        actions.append(np.array(ep["rel_actions"], dtype=np.float32))
    actions = np.stack(actions)                          # (T, 7)

    return rgb, actions


def tokenize_actions_bin(actions: np.ndarray, action_tokenizer) -> list[int]:
    """
    Convert (T, 7) continuous actions to a flat list of bin token IDs.
    7 tokens per timestep (one per action dim), clipped and digitized into
    the last n_bins=256 vocabulary slots.
    """
    token_ids = []
    tok_len = action_tokenizer.tokenizer_len
    for t in range(len(actions)):
        act = np.clip(actions[t], action_tokenizer.min_action, action_tokenizer.max_action)
        disc = np.digitize(act, action_tokenizer.bins)   # (7,) int in [1, 256]
        ids = (tok_len - disc).tolist()                  # 7 token IDs
        token_ids.extend(ids)
    return token_ids


def tokenize_actions_vq(actions: np.ndarray, action_tokenizer) -> list[int]:
    """
    Convert (T, 7) continuous actions to VQ token IDs.
    CalvinVQActionTokenizer processes 5-step chunks → 4 tokens per chunk.
    Uses the tokenizer's internal VQ encoding to get raw token IDs.
    """
    import torch as _torch
    token_ids = []
    T = len(actions)
    n_chunks = T // 5
    tok_len = action_tokenizer.tokenizer_len
    for i in range(n_chunks):
        chunk = actions[i * 5 : (i + 1) * 5]              # (5, 7)
        x = _torch.from_numpy(chunk).float().reshape(1, 5, 7)  # (1, 5, 7)
        _, vq_code = action_tokenizer.vq_vae.get_code(x)   # (1, 4) int codes in [0, 255]
        ids = (tok_len - 1 - vq_code[0]).tolist()          # 4 token IDs
        token_ids.extend(ids)
    return token_ids


def aggregate_attention(
    attentions,           # tuple of (batch, heads, seq_len, seq_len), one per layer
    action_positions: list[int],
    verb_positions:   list[int],
    instr_positions:  list[int],  # non-verb instruction tokens (text only)
    n_text_tokens:    int,        # total text tokens (prompt + action, no image patches)
    layer_start: int = ATTN_LAYER_START,
    layer_end:   int = ATTN_LAYER_END,
) -> dict:
    """
    Extract and aggregate attention from action token positions to verb/instruction positions.

    Normalization is relative to text tokens only (n_text_tokens), NOT seq_len.
    Attention weights sum to 1 over the full sequence (including ~256 image patches),
    so raw values are tiny. Multiplying by n_text_tokens gives values relative to
    "uniform attention over text" (= 1.0 if attending like a random text token).

    Returns:
        verb_attn_raw:      scalar — mean raw attention weight action→verb
        verb_attn_normed:   verb_attn_raw × n_text_tokens  (relative to uniform text baseline)
        instr_attn_normed:  mean attention action→non-verb instruction tokens × n_text_tokens
        verb_attn_contrast: verb_attn_normed − instr_attn_normed
                            (> 0 means action tokens over-attend to verb vs. other text words)
        per_layer_raw:      list of 32 floats, one per layer (raw, not normed)
    """
    num_layers = len(attentions)

    a_idx = torch.tensor(action_positions, dtype=torch.long)
    v_idx = torch.tensor(verb_positions,   dtype=torch.long)
    i_idx = torch.tensor(instr_positions,  dtype=torch.long)

    # Accumulate per-prompt-position and per-layer attention from action tokens
    n_prompt_positions = max(i_idx.max().item(), v_idx.max().item()) + 1  # upper bound
    per_layer_raw = []
    # per_prompt_attn: mean raw attention from action tokens to each prompt position
    # shape accumulator: (num_heads, n_prompt_max) — we'll compute per last-half layers
    per_prompt_raw = None  # will be (n_prompt_positions,)

    for layer_idx in range(num_layers):
        # shape: (1, num_heads, seq_len, seq_len) → (num_heads, seq_len, seq_len)
        attn = attentions[layer_idx][0].float()          # CPU, float32
        # Slice action rows → verb columns: (heads, n_action, n_verb)
        attn_av = attn[:, a_idx, :][:, :, v_idx]         # (heads, n_action, n_verb)
        per_layer_raw.append(attn_av.mean().item())

    # Aggregated scalar for last-half layers
    verb_attn_raw = float(np.mean(per_layer_raw[layer_start:layer_end]))

    # Non-verb instruction-token attention (same layers)
    instr_raw_layers = []
    for layer_idx in range(layer_start, layer_end):
        attn = attentions[layer_idx][0].float()
        attn_ai = attn[:, a_idx, :][:, :, i_idx]        # (heads, n_action, n_instr)
        instr_raw_layers.append(attn_ai.mean().item())
    instr_attn_raw = float(np.mean(instr_raw_layers))

    # Per-prompt-token attention (last-half layers, mean over heads and action positions)
    # Covers all positions from 0 to max(instr+verb positions) in the prompt
    all_prompt_idx = torch.cat([v_idx, i_idx]).unique().sort().values
    per_prompt_attn_layers = []
    for layer_idx in range(layer_start, layer_end):
        attn = attentions[layer_idx][0].float()
        # (heads, n_action, n_all_prompt) → mean over heads and action
        attn_ap = attn[:, a_idx, :][:, :, all_prompt_idx]
        per_prompt_attn_layers.append(attn_ap.mean(dim=(0, 1)).numpy())  # (n_all_prompt,)
    per_prompt_attn = np.mean(per_prompt_attn_layers, axis=0).tolist()  # (n_all_prompt,)

    # Normalize relative to uniform-over-text baseline (NOT uniform-over-full-seq)
    verb_attn_normed   = verb_attn_raw   * n_text_tokens
    instr_attn_normed  = instr_attn_raw  * n_text_tokens
    verb_attn_contrast = verb_attn_normed - instr_attn_normed

    return {
        "verb_attn_raw":      verb_attn_raw,
        "verb_attn_normed":   verb_attn_normed,
        "instr_attn_normed":  instr_attn_normed,
        "verb_attn_contrast": verb_attn_contrast,
        "per_layer_raw":      per_layer_raw,              # 32 floats, raw
        "per_prompt_attn":    per_prompt_attn,            # one float per prompt token (last-half layers avg)
        "per_prompt_idx":     all_prompt_idx.tolist(),    # absolute seq positions for above
    }


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_attention_analysis(
    condition: str,
    checkpoint_dir: str,
    vqvla_checkpoint_dir: str,
    data_root_dir: str,
    output_dir: str,
    max_examples: int,
    min_class_count: int,
    device: str,
) -> None:
    from transformers import (
        AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
    )
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor, PrismaticProcessor,
    )
    from prismatic.vla.action_tokenizer import ActionTokenizer

    # Register HF custom classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading processor + model from {checkpoint_dir} ...")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

    # Load model with eager attention to enable output_attentions=True
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    vla.eval()
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    print(f"Model loaded. num_patches={num_patches}")

    # Build action tokenizer for this condition
    if condition == "bin":
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        _tokenize_fn = lambda acts: tokenize_actions_bin(acts, action_tokenizer)
    elif condition.startswith("vq_"):
        from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer
        action_tokenizer = CalvinVQActionTokenizer(
            processor.tokenizer,
            vqvla_checkpoint_dir=vqvla_checkpoint_dir,
        )
        _tokenize_fn = lambda acts: tokenize_actions_vq(acts, action_tokenizer)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    tokenizer = processor.tokenizer

    # Load CALVIN val set with verb labels
    sys.path.insert(0, PROJECT_ROOT)
    from utils import load_calvin_to_dataframe
    from config import TRAIN_DIR, VAL_DIR

    val_df  = load_calvin_to_dataframe(VAL_DIR)
    if min_class_count > 0:
        train_df    = load_calvin_to_dataframe(TRAIN_DIR)
        verb_counts = train_df["primary_verb"].value_counts()
        keep_verbs  = set(verb_counts[verb_counts >= min_class_count].index)
        val_df      = val_df[val_df["primary_verb"].isin(keep_verbs)].copy()
        print(f"Sparse filter: {len(keep_verbs)} verbs, {len(val_df)} val examples")

    # Build verb→id mapping from training distribution
    train_df    = load_calvin_to_dataframe(TRAIN_DIR)
    verb_counts = train_df["primary_verb"].value_counts()
    keep_verbs  = set(verb_counts[verb_counts >= min_class_count].index)
    sorted_verbs = sorted(keep_verbs)
    verb_to_id   = {v: i for i, v in enumerate(sorted_verbs)}

    val_df = val_df[val_df["primary_verb"].isin(verb_to_id)].reset_index(drop=True)
    if max_examples > 0:
        val_df = val_df.head(max_examples)
    print(f"Processing {len(val_df)} val examples ...")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    skipped = 0

    for idx, row in val_df.iterrows():
        instruction  = row["instruction"]
        primary_verb = row["primary_verb"]
        verb_id      = verb_to_id[primary_verb]

        try:
            rgb, actions = load_trajectory_data(
                CALVIN_VAL_DIR, int(row["start_idx"]), int(row["end_idx"])
            )
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            skipped += 1
            continue

        # ── Tokenize instruction → find verb positions ─────────────────────────
        prompt_text = PROMPT_TEMPLATE.format(instruction=instruction)
        prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=True)
        # Verb tokens (may be multi-token, e.g. "pick up" → ["▁pick", "▁up"])
        verb_words  = primary_verb.split()
        # Try tokenizing the full verb phrase first, then fallback to per-word
        verb_tok_ids = tokenizer.encode(" " + primary_verb, add_special_tokens=False)
        verb_pos_in_prompt = find_subsequence(prompt_ids, verb_tok_ids)
        if verb_pos_in_prompt < 0:
            # Fallback: try without leading space
            verb_tok_ids = tokenizer.encode(primary_verb, add_special_tokens=False)
            verb_pos_in_prompt = find_subsequence(prompt_ids, verb_tok_ids)
        if verb_pos_in_prompt < 0:
            print(f"  [skip] verb '{primary_verb}' not found in prompt tokens (idx={idx})")
            skipped += 1
            continue

        # ── Tokenize actions ────────────────────────────────────────────────────
        try:
            action_token_ids = _tokenize_fn(actions)
        except Exception as e:
            print(f"  [skip] action tokenization failed: {e}")
            skipped += 1
            continue

        if len(action_token_ids) == 0:
            skipped += 1
            continue

        # ── Build full input_ids = prompt_ids + action_token_ids ───────────────
        full_ids  = prompt_ids + action_token_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        # ── Process image ──────────────────────────────────────────────────────
        img_pil = Image.fromarray(rgb).convert("RGB")
        pixel_values = processor.image_processor(
            img_pil,
            return_tensors="pt",
        )["pixel_values"].to(torch.bfloat16).to(device)

        # ── Forward pass ────────────────────────────────────────────────────────
        with torch.no_grad():
            output = vla(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pixel_values,
                output_attentions=True,
            )

        if output.attentions is None:
            raise RuntimeError("output.attentions is None — did you set attn_implementation='eager'?")

        # Move attentions to CPU immediately to free GPU memory
        attentions_cpu = tuple(a.cpu() for a in output.attentions)
        del output
        torch.cuda.empty_cache()

        # ── Identify sequence positions ────────────────────────────────────────
        # Full sequence: [image patches (0..num_patches-1)] [text tokens (num_patches..)]
        # text token j → position num_patches + j in the attention matrix
        n_prompt     = len(prompt_ids)
        n_action     = len(action_token_ids)
        seq_len      = num_patches + n_prompt + n_action

        # Verb positions in full sequence
        verb_positions = [
            num_patches + verb_pos_in_prompt + k
            for k in range(len(verb_tok_ids))
        ]

        # Non-verb instruction positions: all prompt tokens except the verb sub-tokens
        # Used as the contrast baseline — verb_attn_contrast > 0 means the model
        # specifically over-attends to the verb vs. other instruction words.
        verb_pos_set = set(verb_positions)
        instr_positions = [
            p for p in range(num_patches, num_patches + n_prompt)
            if p not in verb_pos_set
        ]

        # Action positions: text tokens after the prompt
        action_positions = list(range(
            num_patches + n_prompt,
            num_patches + n_prompt + n_action
        ))

        # ── Aggregate attention ────────────────────────────────────────────────
        n_text_tokens = n_prompt + n_action  # text-only token count (no image patches)
        agg = aggregate_attention(
            attentions_cpu,
            action_positions=action_positions,
            verb_positions=verb_positions,
            instr_positions=instr_positions,
            n_text_tokens=n_text_tokens,
        )

        # Decode prompt tokens for later per-token analysis
        prompt_token_strs = [
            tokenizer.decode([tid]) for tid in prompt_ids
        ]

        record = {
            "idx":              int(idx),
            "verb_id":          verb_id,
            "primary_verb":     primary_verb,
            "instruction":      instruction,
            "n_prompt_tokens":  n_prompt,
            "n_action_tokens":  n_action,
            "n_verb_tokens":    len(verb_tok_ids),
            "seq_len":          seq_len,
            "n_text_tokens":    n_text_tokens,
            "prompt_ids":       prompt_ids,
            "prompt_tokens":    prompt_token_strs,
            "verb_tok_pos_in_prompt": verb_pos_in_prompt,
            **agg,
        }
        results.append(record)

        if (len(results)) % 20 == 0:
            avg_contrast = np.mean([r["verb_attn_contrast"] for r in results])
            print(f"  [{len(results)}/{len(val_df)}] avg verb_contrast={avg_contrast:.4f}")

    print(f"\nDone: {len(results)} records, {skipped} skipped")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, f"attention_{condition}.json")
    with open(out_path, "w") as f:
        json.dump({"condition": condition, "results": results}, f, indent=2)
    print(f"Saved to {out_path}")

    # ── Print summary by verb class ────────────────────────────────────────────
    from collections import defaultdict
    verb_contrast_by_class = defaultdict(list)
    for r in results:
        verb_contrast_by_class[r["primary_verb"]].append(r["verb_attn_contrast"])

    print("\nVerb attention contrast (verb_attn_normed - non_verb_instr_attn_normed) by class:")
    for verb in sorted(verb_contrast_by_class):
        vals = verb_contrast_by_class[verb]
        print(f"  {verb:<20s} n={len(vals):3d}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    overall_contrast = np.mean([r["verb_attn_contrast"] for r in results])
    overall_normed   = np.mean([r["verb_attn_normed"]   for r in results])
    print(f"\nOverall: verb_attn_normed={overall_normed:.4f}, contrast={overall_contrast:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Attention analysis for fine-tuned OpenVLA")
    p.add_argument("--condition", required=True,
                   choices=["bin", "vq_vanilla", "vq_verb", "vq_verb01"],
                   help="Fine-tuned condition to analyze")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to merged fine-tuned OpenVLA checkpoint")
    p.add_argument("--vqvla_checkpoint_dir", type=str, default="",
                   help="Path to VQ-VLA tokenizer checkpoint (required for vq_* conditions)")
    p.add_argument("--data_root_dir", type=str,
                   default="/data/user_data/wenjiel2/datasets/calvin_rlds")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "attention_analysis"))
    p.add_argument("--max_examples", type=int, default=300,
                   help="Max val examples to process (0 = all)")
    p.add_argument("--min_class_count", type=int, default=30,
                   help="Min train samples per verb class (sparse filter)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    run_attention_analysis(
        condition=args.condition,
        checkpoint_dir=args.checkpoint_dir,
        vqvla_checkpoint_dir=args.vqvla_checkpoint_dir,
        data_root_dir=args.data_root_dir,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        min_class_count=args.min_class_count,
        device=args.device,
    )


if __name__ == "__main__":
    main()
