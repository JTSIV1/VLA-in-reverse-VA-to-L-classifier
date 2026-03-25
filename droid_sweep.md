# DROID Tokenizer Sweep: Experiment Documentation

## Overview

This experiment compares three action tokenizers (VQ-BeT, OAT, QueST) on the DROID
action dataset. Each tokenizer is trained with optional auxiliary heads (verb classification,
CLIP-style action-language contrastive loss) across a small lambda sweep, following the same
experimental template used for CALVIN.

This document covers the **base tokenizer sweep only**. A follow-up DROID hyperparameter sweep
(varying tokenizer architecture/configs per tokenizer) and downstream MiniVLA / OpenVLA policy
training have **not** been run yet.

Unlike CALVIN, DROID is a larger, noisier real-robot dataset with free-form language,
longer trajectories, and substantially more diverse verbs. The sweep therefore measures both
reconstruction quality and whether auxiliary supervision provides any useful semantic signal
without destroying the tokenizer's reconstruction objective.

**Dataset**: DROID action shards (`droid_actions/`), 48,015 filtered single-verb episodes
**Split used here**: 43,214 train / 4,801 val (random 90/10 split, seed 42)
**Average trajectory length**: 238.3 steps (filtered set)
**Sparse verb classes**: 44 classes with `min_class_count >= 30` on train split
**Date**: March 2026

---

## 1. Tokenizer Architecture Summary

### VQ-BeT (Vector-Quantized Behavior Transformer)
- **Architecture**: MLP encoder -> ResidualVQ -> MLP decoder
- **Paper-aligned hyperparameters**: latent_dim=512, num_codes=16, vq_groups=2, chunk_size=5
- **Tokens per chunk**: 2 (one per VQ group), total code combinations = 256
- **Action normalization**: LinearNormalizer (limits mode, `[-1, 1]`)
- **Note**: Uses a learned commitment/codebook loss, so `vq` is nonzero during training

### OAT (One-step Action Tokenizer)
- **Architecture**: RegisterEncoder + FSQ `[8,5,5,5]`
- **Hyperparameters**: num_registers=8, horizon=32
- **Tokens per chunk**: 8, codebook size = 1000
- **Action normalization**: LinearNormalizer (limits mode, `[-1, 1]`)
- **Note**: FSQ has no commitment loss, so the logged `vq` term is expected to be 0

### QueST (Quantized Environment-State Tokenizer)
- **Architecture**: Causal temporal conv + TransformerEncoder + FSQ `[8,5,5,5]`
- **Hyperparameters**: horizon=32, downsample_factor=4
- **Tokens per chunk**: 8 (`32 / 4`), codebook size = 1000
- **Action normalization**: LinearNormalizer (limits mode, `[-1, 1]`)
- **Note**: The DROID sweep used the default `vq_type=fsq`, so logged `vq` is also expected to be 0

---

## 2. Auxiliary Heads

### Verb Classification Head
- Transformer-pooled action latents -> linear verb classifier
- Weighted CE loss over sparse verbs
- Lambda sweep: `{0, 0.01, 0.1, 0.5, 1.0}`

### CLIP Contrastive Head
- Transformer-pooled action embedding aligned to instruction embedding
- Symmetric InfoNCE over action/text pairs
- Lambda sweep: `{0, 0.1, 0.5, 1.0, 2.0}`

---

## 3. File Locations

### Scripts

| Script | Path | Description |
|--------|------|-------------|
| Tokenizer training | `tokenization/train_tokenizer.py` | Main training loop for all 3 tokenizers + aux heads |
| VQ-BeT model | `tokenization/vqbet_tokenizer.py` | VQ-BeT tokenizer implementation |
| OAT model | `tokenization/oat/tokenizer/oat/tokenizer.py` | OAT tokenizer |
| QueST model | `tokenization/oat/tokenizer/quest/tokenizer.py` | QueST tokenizer |
| Sweep submission | `scripts/submit_droid_sweep.sh` | SLURM launcher for DROID vanilla / verb / clip sweeps |
| HP sweep submission | `scripts/submit_droid_hp_sweep.sh` | SLURM launcher for the 18-config DROID HP sweep |
| HP aux retraining | `scripts/submit_droid_hp_aux.sh` | Retrain chosen HP winners with verb / CLIP losses |
| DROID TFDS build | `policy/scripts/build_droid_tfds.sh` | Builds filtered DROID TFDS cache for policy training |
| Base OpenVLA policy | `scripts/submit_droid_policy.sh` | OpenVLA fine-tuning on the base-sweep winners |
| HP OpenVLA policy | `scripts/submit_droid_hp_policy.sh` | OpenVLA fine-tuning on HP-sweep shortlist checkpoints |
| MiniVLA scratch policy | `scripts/submit_droid_scratch.sh` | From-scratch MiniVLA training on DROID |
| Verb probe notes | `lab_notebooks/droid_verb_decodability/round1_setup.md` | DROID dataset setup and earlier action-only probing context |

### Tokenizer Checkpoints

All under `checkpoints/droid_sweep/`:

```
checkpoints/droid_sweep/
├── {vq_bet,oat,quest}_vanilla/
├── {vq_bet,oat,quest}_verb{0.01,0.1,0.5,1.0}/
└── {vq_bet,oat,quest}_clip{0.1,0.5,1.0,2.0}/
```

Each run directory contains:
- `full.pth` - full checkpoint (model + optimizer + args)
- `tokenizer_weights.pth` - tokenizer weights only
- `metrics.csv` - per-epoch metrics
- `config.json` - run summary

### Logs

All under `logs/`, e.g.:
- `logs/dr_vq_bet_vanilla_*.out`
- `logs/dr_oat_verb0.1_*.out`
- `logs/dr_quest_clip0.5_*.out`

### Data

| Resource | Path |
|----------|------|
| DROID action shards | `/data/user_data/wenjiel2/datasets/droid_actions/` |
| DROID RLDS shards | `/data/user_data/wenjiel2/datasets/droid_rlds/` |
| DROID annotations | `/data/user_data/wenjiel2/datasets/droid_annotations/` |
| Cached tokenizer metadata | `data/droid_tokenizer_metadata.csv` |

---

## 4. Tokenizer Sweep Results

### Best Val Recon Loss (per tokenizer, vanilla λ=0)

| Tokenizer | Val Recon | Epochs |
|-----------|-----------|--------|
| VQ-BeT | 0.0320 | 22 (early stop) |
| OAT | **0.0070** | 100 |
| QueST | 0.0201 | 100 |

### Winners

For each tokenizer, the chosen winner is the lambda that gave the strongest auxiliary metric
without an obviously disproportionate reconstruction penalty.

- **VQ-BeT**: verb λ=0.01 (best mF1 among verb runs, recon improves vs vanilla), clip λ=0.5 (best retrieval while still improving recon vs vanilla)
- **OAT**: verb λ=0.01 (recon improves substantially with nontrivial verb signal), clip λ=0.1 (smallest recon cost, retrieval close to stronger lambdas)
- **QueST**: verb λ=0.01 (best recon and best overall balance), clip λ=0.1 (recon slightly better than vanilla; larger λ values hurt recon sharply)

### Verb Sweep Summary

| Tokenizer | Winner | Val Recon | Val Acc | Val mF1 | Recon vs vanilla |
|-----------|--------|-----------|---------|---------|------------------|
| VQ-BeT | verb λ=0.01 | 0.0237 | 6.94% | 1.42% | -25.8% |
| OAT | verb λ=0.01 | 0.0053 | 7.68% | 2.52% | -24.0% |
| QueST | verb λ=0.01 | 0.0144 | 4.33% | 1.63% | -28.6% |

Observations:
- All three tokenizers improved reconstruction under a very small verb loss (`λ=0.01`)
- Absolute verb metrics are low across the board; DROID remains much harder than CALVIN
- OAT achieved the strongest verb macro-F1 on DROID overall at `λ=0.1` (3.42%), but with a worse recon tradeoff than `λ=0.01`
- VQ-BeT verb runs converged quickly and early-stopped after 15-21 epochs, unlike OAT/QueST which trained much longer

### CLIP Sweep Summary

| Tokenizer | Winner | Val Recon | R@1 | R@5 | R@10 | Recon vs vanilla |
|-----------|--------|-----------|-----|-----|------|------------------|
| VQ-BeT | clip λ=0.5 | 0.0254 | 0.42% | 1.62% | 2.73% | -20.5% |
| OAT | clip λ=0.1 | 0.0078 | 0.54% | 2.23% | 4.01% | +11.5% |
| QueST | clip λ=0.1 | 0.0188 | 0.29% | 1.53% | 2.83% | -6.4% |

Observations:
- Retrieval metrics are uniformly low; none of the tokenizers achieve strong action-language alignment on DROID from actions alone
- OAT has the best CLIP retrieval on DROID, reaching 4.21% R@10 at λ=0.5 and 4.01% at λ=0.1
- QueST is the most sensitive to larger CLIP lambdas: λ >= 0.5 more than doubles reconstruction error relative to vanilla
- VQ-BeT again benefits from auxiliary supervision without harming reconstruction, but retrieval remains weak in absolute terms

### Cross-Tokenizer Comparison

Best config per tokenizer in this sweep:

| Tokenizer | Best Vanilla Recon | Best Verb Winner | Best CLIP Winner |
|-----------|--------------------|------------------|------------------|
| VQ-BeT | 0.0320 | verb0.01 | clip0.5 |
| OAT | **0.0070** | verb0.01 | clip0.1 |
| QueST | 0.0201 | verb0.01 | clip0.1 |

Key surprise relative to CALVIN: **OAT is the best recon model on DROID** under the current hyperparameters, while
VQ-BeT is worst of the three. This is the opposite of the CALVIN ranking and suggests DROID's longer,
real-robot trajectories favor the register-based OAT bottleneck more than CALVIN did.

---

## 5. Codebook Utilization

Approximate final code usage from the logged `codes=` metric in the final training epoch.

| Tokenizer | Unique codes | Codebook size | Utilization |
|-----------|-------------|---------------|-------------|
| VQ-BeT vanilla | 201 | 256 | 78.5% |
| OAT vanilla | 999 | 1000 | 99.9% |
| QueST vanilla | 899 | 1000 | 89.9% |

Findings:
- **OAT does not collapse on DROID**. This is a major difference from CALVIN, where OAT used only ~5% of its codebook.
- QueST also maintains healthy utilization on DROID, though slightly below OAT.
- VQ-BeT uses most of its 256 possible code tuples, but not as exhaustively as the FSQ-based models.
- Overall, DROID appears to support much richer code usage than CALVIN under the same default tokenizer settings.

---

## 6. Main Takeaways

1. DROID is substantially harder than CALVIN for both verb and CLIP auxiliary objectives.
2. OAT is the strongest reconstruction model on DROID in this sweep, with near-complete codebook usage.
3. Very small auxiliary weights (`λ=0.01` for verb, `λ=0.1` for CLIP) are the only safe regime; larger lambdas often hurt reconstruction sharply.
4. OAT and QueST train stably for the full 100 epochs, while VQ-BeT tends to early-stop much earlier.
5. Unlike CALVIN, DROID shows **no evidence of OAT codebook collapse** under the current setup.

---

## 7. What Has Not Been Run Yet

### Hyperparameter Sweep

No DROID analogue of the CALVIN HP sweep has been run yet.

The launch code now exists:
- `scripts/submit_droid_hp_sweep.sh`
- `scripts/submit_droid_hp_aux.sh`

That means the report does **not** contain per-tokenizer tables over varying architectural
configs such as:
- VQ-BeT chunk size / codebook size / quantizer groups
- OAT FSQ levels / number of registers / horizon
- QueST FSQ levels / downsample factor / horizon

All DROID results here use the same default base configs that were used in the first sweep.

### Retraining HP Winners with Aux Heads

The CALVIN report includes a second stage where the chosen HP winners are retrained with
`verb=0.1` and `clip=0.1`-style auxiliary losses and summarized again per tokenizer.

That has **not** been done for DROID, because there is currently no completed DROID HP sweep
from which to select winners.

The retraining launcher now exists at `scripts/submit_droid_hp_aux.sh`.

### MiniVLA / OpenVLA Policy Training

No downstream DROID MiniVLA or OpenVLA policy training has been run yet.

The build / launch code now exists at:
- `policy/scripts/build_droid_tfds.sh`
- `scripts/submit_droid_policy.sh`
- `scripts/submit_droid_hp_policy.sh`
- `scripts/submit_droid_scratch.sh`

Evidence:
- No DROID policy run directories were found under `runs/`
- No DROID-specific policy checkpoints or report sections exist analogous to the CALVIN policy stage

So this report currently stops at tokenizer-level reconstruction / verb / CLIP metrics.

---

## 8. Caveats

- `vq_bet_vanilla/metrics.csv` was empty; its metrics were recovered from the corresponding stdout log.
- The DROID metadata used by this sweep is the cached filtered file `data/droid_tokenizer_metadata.csv`, not the full raw 75K-episode annotation set.
- No sweep figure has been generated yet for DROID; this document is based on checkpoint metrics and training logs only.