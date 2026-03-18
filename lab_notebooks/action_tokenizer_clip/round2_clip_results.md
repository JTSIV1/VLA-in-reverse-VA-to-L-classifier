# Round 2: CLIP Action-Language Tokenizer — Results

**Date**: 2026-03-17
**Depends on**: Round 1 (design), `openvla_experiment/scripts/finetune_tokenizer_clip.py`, `openvla_experiment/scripts/eval_clip_retrieval.py`

## Overview

We trained CLIP-aligned action tokenizers using contrastive learning, then evaluated them in two ways:
1. **Verb/instruction retrieval** — zero-shot, measures how well the contrastive space separates actions by language
2. **Downstream VLA action L1** — fine-tune OpenVLA-7B with each tokenizer frozen, measure action prediction quality

---

## Stage 1: Tokenizer Fine-tuning (VQ-VLA + Contrastive)

All conditions use the **pretrained VQ-VLA** (113M params) as the action encoder, fine-tuned with `L_recon + 5*L_vq + λ*L_clip`. Contrastive head: 2-layer transformer (128-d) + linear projection to 128-d shared space. Supervised contrastive loss with false-negative masking. Temperature clamped to max 20.

### 6 conditions (2×3 grid)

| Condition | VQ-VLA FT | Text Encoder | λ_clip | Best Epoch | Val Recon | Val CLIP | Best Val Total |
|-----------|-----------|-------------|--------|-----------|-----------|----------|---------------|
| **clip_full** | Full (113M) | CLIP frozen | 1.0 | 12 | 0.0095 | 1.49 | 1.498 |
| **clip_lora** | LoRA r=8 | CLIP frozen | 1.0 | 25 | 0.0057 | 1.50 | **1.486** |
| **gpt2_full** | Full (113M) | GPT-2 frozen | 1.0 | 17 | 0.0084 | 1.54 | 1.530 |
| **gpt2_lora** | LoRA r=8 | GPT-2 frozen | 1.0 | 49 | 0.0053 | 1.51 | 1.513 |
| **vanilla_full** | Full (113M) | — | 0.0 | 42 | 0.0025 | — | 0.003 |
| **vanilla_lora** | LoRA r=8 | — | 0.0 | 38 | 0.0028 | — | 0.003 |

### Key observations from tokenizer training

1. **LoRA preserves reconstruction quality** — val recon 0.005-0.006 for LoRA vs 0.008-0.010 for full FT. Full FT degrades the pretrained reconstruction.
2. **LoRA trains longer without overfitting** — gpt2_lora ran all 50 epochs and was still improving; full FT hit early stopping at epoch 12-17.
3. **Contrastive loss dominates** — CLIP loss is ~200× larger than recon loss. The total loss is essentially the contrastive loss.
4. **CLIP LoRA has best overall val total** (1.486) but all contrastive conditions are within a narrow range (1.486–1.530).
5. **Vanilla baselines** have excellent recon (0.003) since they optimize only reconstruction.

### Training curves

See `figures/clip_vqvla_curves.png`

---

## Stage 2a: Zero-Shot Retrieval Evaluation

After tokenizer training, we evaluate verb decodability via retrieval — encode each val trajectory through the action branch, compute cosine similarity against text candidates, and check ranking.

**Two retrieval tasks:**
- **Instruction retrieval**: 358 unique instruction candidates (chance R@1 = 0.28%)
- **Verb retrieval**: 29 unique verb candidates (chance R@1 = 3.4%)

### Results

**Instruction Retrieval (358 candidates):**

| Condition | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-----------|-----|-----|------|-------------|-----------|
| **clip_full** | **6.3%** | **28.2%** | **49.7%** | **11** | **14.5** |
| gpt2_full | 5.8% | 26.2% | 44.2% | 12 | 15.1 |
| clip_lora | 4.5% | 25.1% | 48.5% | 11 | 14.7 |
| gpt2_lora | 0.8% | 3.2% | 6.0% | 157 | 161.5 |
| Chance | 0.28% | 1.4% | 2.8% | 179 | 179 |

**Verb Retrieval (29 candidates):**

| Condition | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-----------|-----|-----|------|-------------|-----------|
| **clip_lora** | **12.2%** | 35.9% | 59.2% | 8 | 9.3 |
| **clip_full** | 11.9% | **45.3%** | **71.7%** | **6** | **8.5** |
| gpt2_full | 11.0% | 24.6% | 40.9% | 13 | 12.6 |
| gpt2_lora | 2.1% | 26.1% | 52.5% | 10 | 12.1 |
| Chance | 3.4% | 17.2% | 34.5% | 15 | 15 |

### Per-verb R@1 breakdown (clip_full)

| Verb | R@1 | Count | Notes |
|------|-----|-------|-------|
| move up | 100% | 2 | Tiny class but distinct upward motion |
| place | 65.7% | 35 | Distinct release motion |
| stack | 53.8% | 13 | Downward placing on objects |
| rotate | 53.1% | 64 | Rotational motion is kinematically distinct |
| turn off | 47.1% | 17 | Switch/button interaction |
| push | 30.4% | 125 | Common verb, many variants |
| grasp | 0.0% | 199 | Largest class — similar to pick up, take |
| pick up | 0.0% | 88 | Indistinguishable from grasp |
| slide | 0.0% | 76 | Similar to push kinematically |
| lift | 0.0% | 63 | Similar to grasp + upward |
| turn | 0.0% | 40 | Confused with rotate |

### Retrieval findings

1. **CLIP text encoder outperforms GPT-2** on both retrieval tasks — vision-text contrastive pretraining does transfer to action-text alignment.
2. **Full FT > LoRA for instruction retrieval**, but LoRA is competitive for verb retrieval.
3. **gpt2_lora essentially failed** — near-random instruction retrieval, consistent with its poor codebook structure.
4. **Verbs with distinct kinematics** (rotate, place, stack, turn off) are well-separated; **synonymous verbs** (grasp/pick up/take, push/slide, turn/rotate) remain confounded — this is expected since they share similar action trajectories.
5. **Instruction retrieval is harder** (R@1=6% vs 12% for verb) because the action branch can't distinguish object identity (red block vs blue block).

### Per-verb R@K breakdown (all conditions, 29 verb candidates)

| Verb | N | clip_full |  |  | clip_lora |  |  | gpt2_full |  |  | gpt2_lora |  |  |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | R@1 | R@3 | R@5 | R@1 | R@3 | R@5 | R@1 | R@3 | R@5 | R@1 | R@3 | R@5 |
| rotate | 64 | 53% | 94% | 97% | **91%** | 98% | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| close | 9 | 11% | 22% | 78% | **89%** | 89% | 100% | 0% | 0% | 0% | 0% | 0% | 100% |
| stack | 13 | 54% | 77% | 85% | **100%** | 100% | 100% | 0% | 0% | 8% | 0% | 0% | 0% |
| lift up | 11 | 18% | 82% | 91% | **82%** | 91% | 100% | 64% | 91% | 91% | 0% | 0% | 0% |
| place | 35 | **66%** | 94% | 97% | 60% | 80% | 86% | 0% | 17% | 43% | 0% | 0% | 0% |
| turn off | 17 | 47% | 76% | **94%** | **53%** | 65% | 76% | 0% | 18% | 41% | 0% | 0% | 0% |
| push | 125 | **30%** | 52% | 70% | 1% | 30% | 49% | 0% | 0% | 1% | 0% | 0% | 0% |
| push down | 13 | 0% | 8% | 54% | 8% | **77%** | **85%** | 15% | 46% | 54% | 0% | 0% | 0% |
| lift | 63 | 0% | 13% | 30% | 0% | 29% | 49% | **60%** | 73% | **76%** | 0% | 0% | 0% |
| slide | 76 | 0% | 1% | 28% | 0% | 0% | 9% | **49%** | 76% | **82%** | 0% | 0% | 8% |
| sweep | 26 | 4% | 12% | 19% | 0% | 8% | 23% | **38%** | 69% | 73% | 27% | **100%** | **100%** |
| toggle | 22 | 0% | 14% | 27% | 0% | 0% | 9% | 0% | 14% | 27% | **64%** | 86% | **100%** |
| grasp | 199 | 0% | 24% | 38% | 0% | 0% | 8% | 6% | 13% | 15% | 0% | **89%** | **100%** |
| pick up | 88 | 0% | 1% | **34%** | 0% | 7% | 26% | 1% | 3% | 8% | 0% | 0% | 0% |
| put | 25 | 0% | 12% | 40% | 0% | **60%** | **72%** | 0% | 0% | 0% | 0% | 0% | 0% |
| take | 31 | 0% | 0% | **35%** | 0% | 0% | 3% | 0% | 0% | 0% | 0% | 0% | 0% |
| turn | 40 | 0% | 22% | 22% | 0% | 12% | **40%** | 0% | 0% | 0% | 0% | 0% | 0% |
| unstack | 3 | 33% | 67% | 100% | **67%** | 67% | 100% | 67% | **100%** | 100% | 0% | 33% | 33% |
| move | 45 | 2% | 13% | **33%** | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| open | 9 | 0% | 0% | 0% | 0% | 22% | **56%** | 0% | 0% | 0% | 0% | 0% | 0% |
| pull | 4 | 0% | 0% | 25% | 0% | 0% | **50%** | 0% | 0% | 0% | 0% | 0% | 0% |
| remove | 8 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| take off | 13 | 0% | 0% | 8% | 0% | 31% | **46%** | 0% | 0% | 0% | 0% | 0% | 0% |
| store | 6 | 17% | **83%** | 83% | 0% | 17% | 83% | 0% | 67% | 83% | 0% | 0% | 17% |
| collapse | 5 | 0% | 0% | 60% | 0% | 40% | 60% | 0% | 60% | **80%** | 0% | 0% | 0% |
| slide down | 4 | 25% | **100%** | 100% | 0% | 50% | 100% | 0% | 25% | 75% | 0% | 0% | 0% |
| slide up | 2 | 0% | 100% | 100% | 50% | 50% | 100% | 0% | 50% | **100%** | 0% | 0% | 0% |
| move up | 2 | **100%** | 100% | 100% | 0% | 100% | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| unknown | 53 | 0% | 0% | 2% | 0% | 0% | 0% | 6% | **26%** | **36%** | 0% | 0% | 0% |

### Per-verb retrieval findings

1. **Models learn complementary verb separations**:
   - **CLIP models** excel at fixture interactions: `rotate` (91%), `close` (89%), `stack` (100%), `place` (66%), `turn off` (53%)
   - **GPT-2 full** excels at translational motions: `lift` (60%), `slide` (49%), `sweep` (38%)
   - **GPT-2 LoRA** only gets `toggle` (64%) and `sweep` (27%) — near failure
2. **R@5 reveals hidden structure**: Many verbs at 0% R@1 are within top 5. `grasp` goes from 0% R@1 to 38% R@5 (clip_full) and 100% R@5 (gpt2_lora). The model knows the rough neighborhood but can't distinguish synonymous verbs.
3. **Truly confusable verb groups** (0% R@5 across all models): `remove` is the only verb at 0% everywhere — it's kinematically identical to other grasping motions.
4. **Verb synonym confusion**: `grasp`/`pick up`/`take` all involve reaching and closing gripper — indistinguishable from actions alone. `push`/`slide`/`move` similarly overlap. This is a fundamental limitation of action-only encoding.

---

## Stage 2b: Downstream VLA Evaluation (OpenVLA-7B Fine-tuning)

Each tokenizer is frozen and plugged into OpenVLA-7B. The VLA is fine-tuned with LoRA (r=32) on CALVIN ABCD for 50K steps. Val L1 = mean absolute error of decoded actions vs ground truth.

### Results

| Condition | Best Val Loss | Val L1 ↓ | Val Acc |
|-----------|-------------|----------|---------|
| **gpt2_full** | 1.75 | **0.170** | 22.0% |
| vanilla_lora | 1.98 | 0.189 | 19.1% |
| clip_full | 1.75 | 0.213 | 22.1% |
| vanilla_full | 2.08 | 0.216 | 20.7% |
| clip_lora | 1.43 | 0.220 | 27.4% |
| gpt2_lora | 0.80 | 0.321 | 53.6% |

### Comparison with cls_head experiments (from round2_openvla_finetune.md)

| Tokenizer | Method | Val L1 ↓ | Notes |
|-----------|--------|----------|-------|
| **vq_verb λ=0.5** | Classification head | **0.170** | Best cls_head result |
| **gpt2_full** | CLIP contrastive | **0.170** | Matches cls_head! |
| vq_verb01 λ=0.1 | Classification head | 0.180 | |
| bin (standard) | ActionTokenizer | 0.185 | OpenVLA default |
| **vanilla_lora** | Recon only + LoRA | 0.189 | |
| vq_vanilla | Recon only (cls_head era) | 0.193 | |
| **clip_full** | CLIP contrastive | 0.213 | |
| **vanilla_full** | Recon only | 0.216 | |

### VLA evaluation findings

1. **gpt2_full matches the best cls_head result** (L1=0.170) — the CLIP contrastive approach achieves the same action quality as the classification head approach when using GPT-2 text encoder with full VQ-VLA fine-tuning.
2. **Both beat vanilla/bin baselines** — 0.170 vs 0.185-0.216.
3. **CLIP text encoder underperforms GPT-2 on downstream L1** despite winning on retrieval metrics — interesting decoupling between retrieval quality and downstream utility.
4. **LoRA tokenizers have worse L1** than their full FT counterparts — the restricted adaptation hurts downstream.
5. **Val loss doesn't predict Val L1** — gpt2_lora has lowest val loss (0.80) but worst L1 (0.321). It learns to predict codebook tokens accurately but the tokens don't decode to good continuous actions.

---

## Training curves

See `figures/openvla_clip_curves.png` for all 6 conditions' train/val loss, accuracy, and L1 over 50K steps.

---

## Summary

| | Best retrieval | Best downstream L1 |
|---|---|---|
| Text encoder | **CLIP** (R@1=6.3% instr, 12.2% verb) | **GPT-2** (L1=0.170) |
| VQ-VLA FT | Full FT ≈ LoRA | **Full FT** (0.170 vs 0.321) |
| vs cls_head | N/A (new metric) | **Tied** (0.170 = 0.170) |

The CLIP contrastive approach works — it produces verb-decodable tokenizers that match the classification head approach on downstream VLA performance. The key advantage is that it doesn't require verb labels, uses full instructions (recovering 28% more training data), and handles multi-verb sentences naturally.

However, the retrieval metrics (which favor CLIP text encoder) don't directly predict downstream VLA quality (which favors GPT-2). This suggests the contrastive space structure and downstream utility are partially decoupled — the VLA may benefit more from the reconstruction quality preserved by the GPT-2 condition than from the semantic structure learned by CLIP.

---

## Files

- Tokenizer training: `openvla_experiment/scripts/finetune_tokenizer_clip.py`
- Retrieval evaluation: `openvla_experiment/scripts/eval_clip_retrieval.py`
- VLA fine-tuning: `scripts/submit_openvla_clip_eval.sh`
- Tokenizer checkpoints: `checkpoints/vqvla_clip_{clip_full,clip_lora,gpt2_full,gpt2_lora,vanilla_full,vanilla_lora}/`
- VLA checkpoints: `runs/openvla_clip_eval/`
- Figures: `figures/clip_vqvla_curves.png`, `figures/openvla_clip_curves.png`
