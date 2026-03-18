# Round 3: Attention Analysis — Action Token → Verb Token

**Date**: 2026-03-05
**Goal**: Quantify how strongly action tokens attend to verb tokens in the instruction,
comparing across fine-tuned conditions (bin, vq_vanilla, vq_verb λ=0.5, vq_verb λ=0.1).
**Hypothesis**: A verb-decodable tokenizer (verb CE objective) forces the model to ground
action tokens more tightly in the verb semantics of the instruction; this should manifest
as higher attention from action token positions to verb token positions.

---

## Motivation

Levels 1–4 assess how much verb information is preserved in the tokenizer's latent space
(L1/L2) or predicted by the fine-tuned LLM (L3/L4). But none of them directly measure
**where** the model looks when predicting actions. Attention analysis fills this gap:

> Does the model attend more to the verb when predicting actions under a verb-decodable
> tokenizer?

If yes, this is evidence that the verb CE objective during tokenizer training propagates
into the LLM's attention patterns, not just its token predictions.

---

## Setup

### Script
`openvla_experiment/scripts/analyze_attention.py`

### Data
- CALVIN val set (task_D_D) — same 665 sparse-filtered examples as verb probe
- Loaded via `load_calvin_to_dataframe(VAL_DIR)` + raw CALVIN .npz files

### Model
- Fine-tuned OpenVLA-mini (7B LLaMA-based) from Stage 2
- Loaded with `attn_implementation="eager"` to force standard attention computation
  (flash attention does NOT return attention weights)
- Input: image (first frame of trajectory) + instruction + GT action tokens
  (teacher-forcing format, same as L1/L4 NLL eval)

### Conditions
| Condition | Tokenizer | VQ lambda |
|-----------|-----------|-----------|
| bin | ActionTokenizer (256 bins) | — |
| vq_vanilla | CalvinVQActionTokenizer | 0 (recon only) |
| vq_verb λ=0.5 | CalvinVQActionTokenizer | 0.5 |
| vq_verb λ=0.1 | CalvinVQActionTokenizer | 0.1 |

---

## Aggregation Design (Agreed)

The attention tensor per layer has shape `(batch, heads, seq_len, seq_len)`.
Full sequence: `[image patches (N_patches)] [BOS] [instruction tokens] [action tokens]`.

### Layer aggregation
- **Primary metric**: mean over last-half layers (layers 16–31 of 32)
- **Rationale**: Early layers do lexical/syntactic processing; later layers carry semantic
  grounding. Last-half emphasizes the model's "use" of the instruction during action prediction.
- **Also saved**: per-layer breakdown (32 values) for generating a layer-depth heatmap

### Head aggregation
- **Primary**: mean over all 32 attention heads
- **Rationale**: Individual heads specialize (some attend to syntax, some to objects), but
  averaging gives an unbiased global picture. We don't expect a specific head index to be
  consistent across conditions.
- **Could also report**: fraction of heads with above-average verb attention (robustness check)

### Action token positions
- **Primary**: mean over ALL action token positions in the sequence
- **Rationale**: Every action token implicitly depends on the full instruction; averaging
  gives a trajectory-level signal. No reason to privilege early vs. late tokens a priori.

### Verb token positions
- **Primary**: mean over ALL verb sub-tokens
- **Rationale**: Multi-token verbs (e.g., "pick up" → 2 tokens, "turn on" → 2 tokens) should
  be treated symmetrically. Taking the mean over sub-tokens gives a single "verb region" score.
- Sub-tokens identified by: `tokenizer.encode(" " + primary_verb, add_special_tokens=False)`

### Normalization
- **Raw**: `attn[action_pos, verb_pos]` — mean attention weight (in [0,1], sums to 1 per row)
- **Normalized**: `raw × seq_len` — relative to uniform attention (= 1.0 if attending like random)
- **Rationale**: seq_len varies by condition (bin has 7× more action tokens than VQ's 4 per chunk),
  so raw values are not directly comparable without normalization.

### Contrast metric (primary comparison metric)
```
verb_attn_contrast = verb_attn_normed − mean_instr_attn_normed
```
Where `mean_instr_attn_normed` = mean normalized attention from action tokens to all instruction
token positions. Contrast > 0 means action tokens specifically over-attend to the verb relative
to the average instruction word.

**Rationale**: This controls for overall "look at instruction" tendency. If a condition simply
attends more to ALL instruction tokens, contrast stays flat. Only a specific increase toward
the verb word shows up in contrast.

---

## Sequence Structure

```
Full attention sequence (length = num_patches + len(input_ids)):
  [0 .. num_patches-1]          image patch tokens
  [num_patches]                 BOS token
  [num_patches+1 .. num_patches+n_prompt-1]  instruction tokens: "In: {instruction}\nOut: "
  [num_patches+n_prompt .. end]  action tokens (GT, teacher-forcing)
```

Prompt format: `"In: {instruction}\nOut: "` (from `PurePromptBuilder.wrap_human`)

Position mapping:
- Text token j in `input_ids` → attention position `num_patches + j`
- `num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches` (typically 256 for SigLIP)

Verb token positions found by tokenizing `" " + primary_verb` and searching in `input_ids`
(leading space because LLaMA tokenizer merges the preceding space into the token).

---

## Technical Considerations

### Flash Attention incompatibility
`output_attentions=True` requires standard (eager) attention — flash attention / SDPA silently
return `None` for attention weights. Fix: load model with `attn_implementation="eager"`.
**Cost**: ~20–30% slower per forward pass than flash attention.

### Memory
Attention tensors: `32 layers × 32 heads × seq_len × seq_len × 4 bytes`.
For seq_len ≈ 256 + 40 + 21 ≈ 317 (bin condition, 3 steps shown):
  `32 × 32 × 317² × 4 bytes ≈ 41 MB per example`
With a long trajectory (50 steps, bin = 350 action tokens):
  seq_len ≈ 256 + 40 + 350 = 646 → `32 × 32 × 646² × 4 bytes ≈ 170 MB`
We process one example at a time and immediately move attentions to CPU, then empty GPU cache.

### SLURM
- Partition: general (GPU, 1× L40S)
- Memory: 64G, Time: 8h
- Script: `openvla_experiment/scripts/submit_attention_analysis.sh`
- Actual wall time: ~5 min for 291 examples (much faster than estimated — eager attention on L40S is fast)

---

## Jobs

| Job ID | Condition | Status | Wall time |
|--------|-----------|--------|-----------|
| 6505723 | bin | Done | ~2 min |
| 6505724 | vq_vanilla | Done | ~2 min |
| 6505725 | vq_verb λ=0.5 | Done | ~2 min |
| 6505726 | vq_verb λ=0.1 | Done | ~2 min |

Text-normalized values recomputed offline from saved `verb_attn_raw` (no rerun needed).

---

## Results (291 val examples, 21 sparse classes)

Normalization: `× n_text_tokens` (prompt + action tokens only, excludes ~256 image patches).
Baseline: non-verb instruction tokens (verb positions excluded from baseline).

| Condition | verb_attn_normed ↑ | instr_attn_normed | contrast ↑ |
|-----------|-------------------|-------------------|-----------|
| bin | **0.5995** | 9.5598 | −8.960 |
| vq_vanilla | 0.1574 | 1.6206 | **−1.463** |
| vq_verb λ=0.5 | 0.1688 | 1.6492 | −1.480 |
| vq_verb λ=0.1 | 0.1613 | 1.6498 | −1.489 |

Output files: `results/attention_analysis/attention_{condition}.json`

---

## Key Findings

1. **All contrasts remain negative** — action tokens attend less to the verb than to the
   average non-verb instruction word, even after excluding image patch dilution. The verb
   is just 1-2 tokens out of ~40 instruction tokens, so over-attending to any single word
   requires strong structural bias.

2. **Bin has dramatically worse contrast (−9.0) vs. all VQ conditions (−1.46 to −1.49)**.
   The gap is 6× larger for bin, driven by bin's verb_attn_normed (0.60) being much lower
   relative to its instr_attn_normed (9.56). Bin action tokens spread attention broadly
   across all instruction text; VQ action tokens attend much more uniformly (both verb and
   non-verb are near 0.16–1.6 in the text-normalized scale).

3. **VQ verb CE makes no difference to attention contrast** (−1.463 vanilla vs −1.480/−1.489 verb).
   All three VQ conditions are essentially identical. Verb CE improves tokenizer verb
   decodability (L1/L2) and LLM embedding clustering (L3), but does not shift the model's
   attention pattern toward the verb during generation.

4. **The key split is bin vs. VQ, not vanilla vs. verb-decodable**. The VQ tokenizer
   (regardless of verb CE) produces a much more uniform attention pattern over instruction
   text — bin attends ~6× more to non-verb words relative to the verb.

5. **Interpretation**: Bin predicts 7 tokens/step × ~60 steps = ~420 action tokens from a
   7-token instruction window, so the instruction text contributes less per-token. VQ predicts
   4 tokens per 5-step chunk (~48 total), concentrating the instruction signal into fewer
   prediction steps. The overall instruction-attending tendency is higher for VQ relative to
   the action token count.

---

## Planned Visualizations

1. **Bar chart**: `verb_attn_contrast` per condition (primary figure)
2. **Layer depth heatmap**: mean `verb_attn_raw` per layer per condition
3. **Per-verb breakdown**: `verb_attn_contrast` by verb class × condition

---

## Next Steps

- [x] Write submission script `submit_attention_analysis.sh`
- [x] Submit jobs for all 4 conditions
- [x] Fill in results table
- [ ] Generate visualizations (bar chart, layer heatmap, per-verb breakdown)
- [ ] Interpret: does bin vs. VQ gap hold per-verb? (fixture verbs vs. motion verbs)
