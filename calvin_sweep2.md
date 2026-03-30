# CALVIN D→D Tokenizer Sweep V2: Fixed QueST + CLS-Token Aux Heads

## Overview

This is the second iteration of the CALVIN tokenizer sweep, fixing two critical issues
from V1 (`calvin_sweep.md`):

1. **QueST horizon bug**: `build_quest()` used `args.chunk_size` (=4, VQ-BeT default)
   instead of `args.horizon` (=16/32) when reconstructing the model at inference time.
   This caused wrong positional embeddings and decoder output shape → garbage actions → 0% SR.

2. **Aux head pooling**: V1 used mean-pooling over all registered/quantized tokens before
   feeding to verb/CLIP heads. With long horizons (16-32 steps → 4-8 tokens after downsampling),
   mean-pooling collapses too much temporal structure. V2 uses a CLS-token Transformer
   (`tokenization/aux_heads.py`) that prepends a learnable [CLS] token, adds sinusoidal PE,
   and runs a TransformerEncoder before classification/projection.

**Dataset**: CALVIN D→D (task_D_D), 3,398 training / 671 validation episodes
**Verb classes**: 21 sparse classes (min_class_count=30, weighted CE)
**Date**: March 2026

---

## 1. Changes from V1

### Bug Fix: QueST Horizon

In `tokenization/train_tokenizer.py`, `build_quest()` now uses:
```python
horizon = getattr(args, 'horizon', getattr(args, 'chunk_size', 32))
```
Previously it used `args.chunk_size` which defaulted to 4 (VQ-BeT's default), even when
the QueST config specified `horizon=16` or `horizon=32`.

Additionally, `chunk_size` is now synced to `horizon` for QueST in `parse_args()`:
```python
if args.tokenizer == 'quest' and hasattr(args, 'horizon'):
    args.chunk_size = args.horizon
```

### New Aux Heads: CLS-Token Transformer

V1 aux heads:
- Mean-pool all encoder tokens → single vector → linear classifier/projector
- With horizon=16, downsample=2 → 8 tokens → mean to 1 → too lossy

V2 aux heads (`tokenization/aux_heads.py`):
- **ActionTransformer**: prepend learnable [CLS] token + sinusoidal PE → TransformerEncoder → CLS output
- **VerbHead**: ActionTransformer → linear classifier (num_verbs classes)
- **ContrastiveHead**: ActionTransformer → projection → InfoNCE loss
- **TextEncoderWrapper**: CLIP/GPT-2 text encoder with optional LoRA

### Sweep Infrastructure

New `run_sweep.sh` handles the full pipeline:
- Cartesian product over TOKENIZER x AUX_HEAD x TOK_SET
- SLURM dependency chaining: tokenizer → {probe, policy} (concurrent)
- `--set` overrides for per-config hyperparameters (horizon, fsq_levels, downsample_factor)

---

## 2. Sweep Configuration

### Tokenizer Configs (QueST only)

| Config | Horizon | FSQ Levels | Codebook | Downsample | Tokens/chunk |
|--------|---------|-----------|----------|------------|-------------|
| h16/f256/d2 | 16 | [4,4,4,4] | 256 | 2 | 8 |
| h32/f1000/d4 | 32 | [8,5,5,5] | 1000 | 4 | 8 |
| h16/f256/d4 | 16 | [4,4,4,4] | 256 | 4 | 4 |

### Aux Head Variants

| Aux | Description | Lambda |
|-----|-------------|--------|
| none | Vanilla (recon only) | 0 |
| verb:0.1 | CLS-token verb classifier | 0.1 |
| clip:0.1 | CLS-token contrastive | 0.1 |

### Total: 3 configs x 3 aux = 9 conditions

---

## 3. Scripts & File Locations

| Resource | Path |
|----------|------|
| Sweep launcher | `run_sweep.sh` |
| Tokenizer training | `tokenization/train_tokenizer.py` |
| Aux heads | `tokenization/aux_heads.py` |
| QueST model | `tokenization/oat/tokenizer/quest/tokenizer.py` |
| Verb probe | `verb_probe/train_verb_probe.py` |
| Policy train | `<openvla-mini>/vla-scripts/train.py` |
| Policy adapter | `<openvla-mini>/prismatic/vla/calvin_sweep_action_tokenizer.py` |
| Rollout eval | `policy/scripts/evaluate_openvla_rollout.py` |

Where `<openvla-mini>` = `/data/user_data/wenjiel2/Code/openvla-mini`

### Checkpoints

All V2 tokenizer checkpoints are saved under:
```
/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/calvin_sweep/tokenizers/
```

Each directory contains:
- `config.json` — training config (horizon, fsq_levels, downsample, aux type, etc.)
- `tokenizer_weights.pth` — tokenizer-only weights (for policy adapter loading)
- `full.pth` — full checkpoint (tokenizer + aux heads + optimizer state)
- `metrics.csv` — training/validation metrics per epoch
- `probe_native_best.pth` / `probe_native_best_best.pth` — verb probe on raw actions (where available)
- `probe_latent_best.pth` / `probe_latent_best_best.pth` — verb probe on encoder latents (where available)

```
checkpoints/calvin_sweep/tokenizers/
├── quest_16_4444_2/              # h16/f256/d2, vanilla  
├── quest_32_8555_4/              # h32/f1000/d4, vanilla 
├── quest_16_4444_4/              # h16/f256/d4, vanilla
├── quest_verb0.1_16_4444_2/      # h16/f256/d2 + verb    
├── quest_verb0.1_32_8555_4/      # h32/f1000/d4 + verb   
├── quest_verb0.1_16_4444_4/      # h16/f256/d4 + verb   
├── quest_clip0.1_16_4444_2/      # h16/f256/d2 + clip  
├── quest_clip0.1_32_8555_4/      # h32/f1000/d4 + clip
└── quest_clip0.1_16_4444_4/      # h16/f256/d4 + clip
```

(V1 checkpoints remain in `checkpoints/calvin_sweep/tokenizers/quest/` for reference.)

---

## 4. Tokenizer Training Results

All 9 tokenizers trained successfully. Early stopping with patience=15 on total val loss
(recon + lambda * aux_loss).

### Vanilla (no aux)

| Config | Val Recon | Epochs | Codebook Util |
|--------|-----------|--------|---------------|
| h16/f256/d2 | 0.0090 | 184 | 674/— |
| h32/f1000/d4 | 0.0109 | 178 | 674/— |
| h16/f256/d4 | 0.0111 | 160 | 674/— |

### Verb aux (lambda=0.1)

| Config | Val Recon | Val Verb Acc | Val mF1 | Epochs |
|--------|-----------|-------------|---------|--------|
| h16/f256/d2 | 0.0145 | 40.5% | 41.7% | 42 |
| h32/f1000/d4 | 0.0210 | 41.7% | 43.7% | 37 |
| h16/f256/d4 | 0.0160 | 40.1% | 42.1% | 57 |

### CLIP aux (lambda=0.1)

| Config | Val Recon | Val R@1 | Val R@5 | Epochs |
|--------|-----------|---------|---------|--------|
| h16/f256/d2 | 0.0178 | 2.8% | 14.1% | 32 |
| h32/f1000/d4 | 0.0225 | 3.0% | 13.6% | 29 |
| h16/f256/d4 | 0.0192 | 3.4% | 15.4% | 33 |

### Summary

- **Best recon**: h16/f256/d2 vanilla (0.0090) — matches V1 finding that this config wins
- **Best verb acc**: h32/f1000/d4 + verb (41.7% / 43.7% mF1) — larger horizon helps verb classification
- **Aux heads hurt recon**: +60-100% recon loss, but much less than V1 post-FSQ (which doubled recon)
- **CLS-token verb head much better than V1 mean-pool**: V1 post-FSQ verb acc was 13.5%; V2 gets 40-42%
- **Early stopping triggered early with aux**: 29-57 epochs vs 160-184 for vanilla

---

## 5. Comparison with V1

### Tokenizer Recon (same configs, V1 vs V2)

| Config | V1 Recon | V2 Recon | Notes |
|--------|----------|----------|-------|
| h16/f256/d2 vanilla | 0.0121 | 0.0090 | V2 better (horizon fix) |
| h32/f1000/d4 vanilla | 0.0138 | 0.0109 | V2 better |
| h16/f256/d4 vanilla | 0.0158 | 0.0111 | V2 better |

Reconstruction improved across the board because the `chunk_size` bug also affected
**training**, not just inference. V1 checkpoints show `chunk_size=4` (VQ-BeT default)
while `horizon=16/32` — meaning the dataset fed 4-step chunks to a model expecting
16/32-step inputs. V2 correctly syncs `chunk_size=horizon`, so training data matches
the model architecture. V1 also used batch_size=64 vs V2's 32 (minor factor).

### Aux Head Verb Accuracy (V1 mean-pool vs V2 CLS-token)

| Approach | Config | Val Verb Acc | Val mF1 | Val Recon |
|----------|--------|-------------|---------|-----------|
| V1 post-FSQ mean-pool | h16/f256/d2 | 13.5% | 15.8% | 0.0249 |
| V1 pre-FSQ mean-pool | h16/f256/d2 | 18.3% | 18.6% | 0.0208 |
| **V2 CLS-token** | **h16/f256/d2** | **40.5%** | **41.7%** | **0.0145** |

The CLS-token Transformer aux head is dramatically better: +22pp verb accuracy and
lower recon loss than both V1 approaches.

### Aux Head CLIP (V1 mean-pool vs V2 CLS-token)

| Approach | Config | Val R@1 | Val R@5 | Val R@10 | Val Recon |
|----------|--------|---------|---------|----------|-----------|
| V1 post-FSQ mean-pool | h16/f256/d2 | 1.7% | 7.1% | 11.7% | 0.0302 |
| V1 pre-FSQ mean-pool | h16/f256/d2 | 2.1% | 7.6% | 13.6% | 0.0262 |
| **V2 CLS-token** | **h16/f256/d2** | **2.8%** | **14.1%** | **—** | **0.0178** |

CLIP retrieval also improves: R@5 nearly doubles (7.1% → 14.1%) and recon loss drops 41%
(0.0302 → 0.0178). The CLS-token approach benefits both aux head types.

### V1 Rollout Results (for reference — QueST broken due to horizon bug)

| Condition | Tokenizer | Real L1 | SR1 | SR2 | SR3 |
|-----------|-----------|---------|-----|-----|-----|
| vb_c5e16g4_verb01 | VQ-BeT + verb | 0.323 | **35.7%** | 5.3% | 0.4% |
| vb_c10e16g4 | VQ-BeT c10/e16/g4 | 0.337 | 34.7% | **6.5%** | **1.1%** |
| bin_baseline | Bin 256 | 0.630 | 32.4% | 2.4% | 0.4% |
| vb_c5e64g2 | VQ-BeT c5/e64/g2 | 0.348 | 31.0% | 5.6% | 0.6% |
| vb_c5e16g4 | VQ-BeT c5/e16/g4 | 0.357 | 30.0% | 5.7% | 1.0% |
| vb_c5e16g4_clip01 | VQ-BeT + clip | 0.348 | 29.4% | 4.0% | 0.2% |
| quest_h32f1000d4 | QueST (BROKEN) | 0.315 | 0.3% | 0.0% | 0.0% |
| quest_h16f256d2 | QueST (BROKEN) | 0.304 | 0.0% | 0.0% | 0.0% |
| quest_h16f256d4 | QueST (BROKEN) | 0.318 | 0.0% | 0.0% | 0.0% |
| quest_h16d2_verb01 | QueST (BROKEN) | 0.303 | 0.0% | 0.0% | 0.0% |
| quest_h16d2_clip01 | QueST (BROKEN) | 0.317 | 0.0% | 0.0% | 0.0% |

**V1 key finding**: VQ-BeT verb01 had best SR1 (35.7%). All QueST conditions got 0% due to
the horizon bug — the model was reconstructed with horizon=4 instead of 16/32 at inference,
producing garbage actions despite good Real L1 (which only measured teacher-forced predictions
before the decode bug manifested in autoregressive generation).

---

## 6. Verb Probe Results

Each tokenizer is evaluated with 3 probe types:
- **native**: raw continuous actions (baseline)
- **tokid**: discrete token IDs through learned embedding
- **latent**: continuous encoder latents

*All probes retrained with 100 epochs (jobs 6822712–6822730). Native baseline: 30.3% Acc / 28.1% MF1.*

### QueST Probes

Native baseline: 30.3% Acc / 28.1% MF1

| Condition | tokid (Acc / MF1) | latent (Acc / MF1) |
|-----------|-------------------|-------------------|
| quest_16_4444_2 | 32.0% / 28.7% | 29.9% / 29.5% |
| quest_32_8555_4 | 30.2% / 27.1% | 31.4% / 28.9% |
| quest_16_4444_4 | 31.7% / 27.0% | 31.7% / 27.3% |
| quest_verb0.1_16_4444_2 | 30.5% / 29.3% | **38.6% / 40.9%** |
| quest_verb0.1_32_8555_4 | 30.0% / 27.1% | **40.7% / 42.6%** |
| quest_verb0.1_16_4444_4 | 30.6% / 26.1% | **36.5% / 38.5%** |
| quest_clip0.1_16_4444_2 | 32.9% / 29.8% | **39.0% / 41.1%** |
| quest_clip0.1_32_8555_4 | 27.5% / 24.7% | **40.1% / 40.4%** |
| quest_clip0.1_16_4444_4 | 32.3% / 30.7% | **37.2% / 37.0%** |

**Findings:** Aux heads (verb/clip) dramatically improve latent verb decodability (37–43% MF1 vs ~28% vanilla), but tokid probes show no benefit (~27–31% MF1 across the board). The discrete FSQ codes do not retain the semantic structure that the aux loss injects into the continuous latents.

### VQ-BeT Probes

Native baseline: 30.9% Acc / 26.5% MF1

| Condition | tokid (Acc / MF1) | latent (Acc / MF1) |
|-----------|-------------------|-------------------|
| vq_bet_5_16_2 | 30.9% / 27.8% | 23.4% / 18.1% |
| vq_bet_5_16_4 | 30.2% / 28.8% | 18.5% / 12.5% |
| vq_bet_5_16_4_verb0.1 | 29.9% / 25.5% | 29.0% / 25.7% |
| vq_bet_5_16_4_clip0.1 | **32.0% / 29.8%** | 30.2% / 26.6% |
| vq_bet_5_64_2 | 32.9% / 28.4% | 21.9% / 18.4% |
| vq_bet_10_16_2 | 28.4% / 25.9% | 20.7% / 16.4% |
| vq_bet_10_16_4 | 31.5% / 27.1% | 15.9% / 11.5% |
| vq_bet_10_64_2 | 31.7% / 26.2% | 20.6% / 16.9% |

**VQ-BeT findings:**
- **Latent probes are very weak** (12–18% MF1 for vanilla) — much worse than QueST latents. VQ-BeT's ResidualVQ bottleneck compresses latents more aggressively.
- **Aux heads help VQ-BeT latents** (verb: 25.7%, clip: 26.6%) but the gain is smaller than for QueST (~+13pp vs ~+14pp).
- **Tokid probes are the better representation for VQ-BeT** — consistently outperform latents, opposite to QueST.
- **Best VQ-BeT tokid**: vq_bet_5_64_2 at 32.9% acc / 28.4% MF1.

---

## 7. Policy Training (MiniVLA from scratch)

Same setup as V1: MiniVLA (Qwen2.5-0.5B + DINOv2/SigLIP), full FSDP, frozen vision backbone,
50K steps, batch_size=16.

*Status: 6/9 completed, 3 timed out (12h SLURM limit)*

| Condition | Tag | Final Step | Final Loss | Token Acc | Best Loss (step) | Notes |
|-----------|-----|-----------|-----------|-----------|-----------------|-------|
| quest vanilla h16/d2 | quest_16_4444_2 | 35250 | 0.5661 | 75.0% | 0.5973 (30K) | TIMEOUT at 12h |
| quest vanilla h32/d4 | quest_32_8555_4 | 45740 | 0.8903 | 71.9% | 0.6680 (40K) | TIMEOUT at 12h |
| quest vanilla h16/d4 | quest_16_4444_4 | 50000 | 0.5340 | 65.6% | 0.5316 (30K) | completed |
| quest verb h16/d2 | quest_verb0.1_16_4444_2 | 50000 | 0.5185 | 80.5% | 0.4479 (15K) | completed |
| quest verb h32/d4 | quest_verb0.1_32_8555_4 | 40505 | 0.6128 | 73.8% | 0.5069 (25K) | TIMEOUT at 12h |
| quest verb h16/d4 | quest_verb0.1_16_4444_4 | 50000 | 0.4308 | 73.4% | 0.3954 (45K) | completed |
| quest clip h16/d2 | quest_clip0.1_16_4444_2 | 50000 | 0.4930 | 79.7% | 0.3684 (45K) | completed |
| quest clip h32/d4 | quest_clip0.1_32_8555_4 | 50000 | 0.5363 | 80.0% | 0.5363 (50K) | completed |
| quest clip h16/d4 | quest_clip0.1_16_4444_4 | 50000 | 0.3514 | 78.1% | 0.3514 (50K) | completed |

Notes:
- All 3 timeouts are h32/d4 or h16/d2 vanilla configs (longer sequences = slower steps)
- Best token accuracy: verb h16/d2 (80.5%), clip h32/d4 (80.0%), clip h16/d2 (79.7%)
- Best final loss: clip h16/d4 (0.3514), verb h16/d4 (0.4308), clip h16/d2 (0.4930)
- h32/d4 configs consistently have higher loss — larger codebook (1000) is harder to predict

---

## 8. Real L1 Evaluation (Teacher-Forced)

*Partial results from 100 batches (TF jobs timed out at 4h due to slow RLDS streaming; metrics stable by batch 100).*

| Condition | Real L1 | V1 Real L1 |
|-----------|---------|------------|
| quest_16_4444_2 | 0.0520 | 0.304 |
| quest_16_4444_2_verb0.1 | 0.0279 | 0.303 |
| quest_16_4444_2_clip0.1 | **0.0213** | 0.317 |
| quest_16_4444_4 | 0.0831 | 0.318 |
| quest_16_4444_4_verb0.1 | 0.0322 | — |
| quest_16_4444_4_clip0.1 | 0.0310 | — |
| quest_32_8555_4 | 0.1200 | 0.315 |
| quest_32_8555_4_verb0.1 | 0.0394 | — |
| quest_32_8555_4_clip0.1 | 0.0666 | — |

Notes:
- V2 Real L1 is much lower than V1 across the board — the horizon fix drastically improved reconstruction
- **Aux heads reduce Real L1**: verb/clip conditions have 2–3× lower L1 than vanilla within each config
- **Best Real L1**: clip h16/d2 (0.0213), followed by verb h16/d2 (0.0279)
- h32/d4 configs have the highest L1 (larger codebook = harder to predict)
- V1 Real L1 values are not directly comparable (different eval methodology: V1 used evaluate_openvla.py)

---

## 9. Rollout Evaluation

*All 9 QueST V2 conditions evaluated (1000 sequences each).*

| Condition | SR1 | SR2 | SR3 | V1 SR1 |
|-----------|-----|-----|-----|--------|
| quest_32_8555_4 | **6.5%** | 0.0% | 0.0% | 0.3% (broken) |
| quest_32_8555_4_verb0.1 | 4.5% | 0.0% | 0.0% | — |
| quest_16_4444_4 | 3.8% | 0.1% | 0.0% | 0.0% (broken) |
| quest_32_8555_4_clip0.1 | 2.1% | 0.1% | 0.0% | — |
| quest_16_4444_2_verb0.1 | 1.0% | 0.0% | 0.0% | 0.0% (broken) |
| quest_16_4444_2_clip0.1 | 0.5% | 0.0% | 0.0% | 0.0% (broken) |
| quest_16_4444_2 | 0.1% | 0.0% | 0.0% | 0.0% (broken) |
| quest_16_4444_4_verb0.1 | 0.0% | 0.0% | 0.0% | — |
| quest_16_4444_4_clip0.1 | 0.0% | 0.0% | 0.0% | — |

### V1 Best (VQ-BeT, for comparison target)

| Condition | SR1 | SR2 |
|-----------|-----|-----|
| vb_c5e16g4_verb01 | 35.7% | 5.3% |
| vb_c10e16g4 | 34.7% | 6.5% |

**Findings:**
- QueST V2 rollout performance is very poor (best 6.5% SR1), far below VQ-BeT V1 (35.7%).
- The horizon fix improved V1→V2 (0% → 0.1–6.5%), confirming the bug was real, but QueST still underperforms.
- **Lower Real L1 does not predict higher SR**: clip h16/d2 has the best L1 (0.021) but only 0.5% SR1, while vanilla h32/d4 has the worst L1 (0.120) but the best SR1 (6.5%).
- **Aux heads hurt rollout for h16/d4**: vanilla gets 3.8% SR1, but verb and clip get 0.0%. The aux loss may overfit the tokenizer to semantic structure at the expense of action fidelity.
- h32/d4 configs do best in rollout despite having the highest policy loss and Real L1 — longer horizons may be more forgiving of per-step errors.
