# CALVIN D→D Tokenizer Sweep: Experiment Documentation

## Overview

This experiment compares three action tokenizers (VQ-BeT, OAT, QueST) on the CALVIN D→D
benchmark. Each tokenizer is trained with optional auxiliary heads (verb classification, CLIP
contrastive) at varying lambda values. Pareto-optimal winners are then used for downstream
policy training with OpenVLA-mini.

**Dataset**: CALVIN D→D (task_D_D), 3,398 training / 671 validation episodes
**Verb classes**: 20 sparse classes (weighted CE loss)
**Date**: March 2026

---

## 1. Tokenizer Architecture Summary

### VQ-BeT (Vector-Quantized Behavior Transformer)
- **Architecture**: MLP encoder → ResidualVQ → MLP decoder
- **Paper-aligned hyperparameters**: latent_dim=512, num_codes=16, vq_groups=2, chunk_size=5
- **Tokens per chunk**: 2 (one per VQ group), codebook size = 16
- **Action normalization**: LinearNormalizer (limits mode, [-1, 1])
- **Reference**: Behavior Generation with Latent Actions (Lee et al., 2024)

### OAT (One-step Action Tokenizer)
- **Architecture**: RegisterEncoder + FSQ [8,5,5,5]
- **Hyperparameters**: num_registers=8, horizon=32
- **Tokens per chunk**: 8 (one per register), codebook size = 1000 (8x5x5x5)
- **Action normalization**: LinearNormalizer (limits mode, [-1, 1])
- **Note**: FSQ has no commitment loss (deterministic rounding)

### QueST (Quantized Environment-State Tokenizer)
- **Architecture**: Causal conv + TransformerEncoder + FSQ [8,5,5,5]
- **Hyperparameters**: horizon=32, downsample_factor=4
- **Tokens per chunk**: 8 (horizon / downsample_factor), codebook size = 1000
- **Action normalization**: LinearNormalizer (limits mode, [-1, 1])

---

## 2. Auxiliary Heads

### Verb Classification Head
- Linear probe on mean-pooled encoder latents → 20-class verb prediction
- Weighted CE loss (inverse class frequency)
- Lambda sweep: {0, 0.01, 0.1, 0.5, 1.0}

### CLIP Contrastive Head
- Instruction-level CLIP loss between encoder latents and language embeddings
- Lambda sweep: {0, 0.1, 0.5, 1.0, 2.0}

---

## 3. File Locations

### Scripts

| Script | Path | Description |
|--------|------|-------------|
| Tokenizer training | `tokenization/train_tokenizer.py` | Main training loop for all 3 tokenizers + aux heads |
| VQ-BeT model | `tokenization/vqbet_tokenizer.py` | VQBeTTokenizer class with normalizer |
| OAT model | `tokenization/oat/tokenizer/oat/tokenizer.py` | OAT tokenizer (RegisterEncoder + FSQ) |
| QueST model | `tokenization/oat/tokenizer/quest/tokenizer.py` | QueST tokenizer (CausalConv + Transformer + FSQ) |
| Sweep submission | `scripts/submit_calvind_sweep.sh` | SLURM sbatch for 27-job tokenizer sweep |
| Policy submission | `scripts/submit_calvind_policy.sh` | SLURM sbatch for 10 policy training jobs |
| Sweep figure | `figures/plot_calvind_sweep.py` | Generates `figures/calvind_sweep.png` |
| Policy adapter | `<openvla-mini>/prismatic/vla/calvin_sweep_action_tokenizer.py` | CalvinSweepActionTokenizer for OpenVLA-mini |
| Policy finetune | `<openvla-mini>/vla-scripts/finetune.py` | OpenVLA-mini LoRA fine-tuning script |

Where `<openvla-mini>` = `/data/user_data/wenjiel2/Code/openvla-mini`

### Tokenizer Checkpoints

All under `checkpoints/calvind_sweep/`:

```
checkpoints/calvind_sweep/
├── {vq_bet,oat,quest}_vanilla/           # lambda=0 baselines
│   ├── full.pth                          # full checkpoint (model + optimizer + args)
│   ├── tokenizer_weights.pth             # model weights only
│   ├── metrics.csv                       # per-epoch training/val metrics
│   └── config.json                       # run config summary
├── {vq_bet,oat,quest}_verb{0.01,0.1,0.5,1.0}/   # verb lambda sweep
│   └── ... (same structure)
├── {vq_bet,oat,quest}_clip{0.1,0.5,1.0,2.0}/     # clip lambda sweep
│   └── ... (same structure)
└── *_*_*/ (e.g. oat_verb0.1_verb0.1)     # OLD naming, pre-rerun (ignore)
```

**Note**: Directories with double-tag names (e.g., `oat_verb0.1_verb0.1`) are from the first
run before VQ-BeT normalization was added. The single-tag directories (e.g., `oat_verb0.1`)
are the current/correct checkpoints.

### Policy Checkpoints

All under `runs/calvind_policy/`:

```
runs/calvind_policy/
├── openvla-7b+...--bin_baseline--image_aug/       # bin-based ActionTokenizer baseline
├── openvla-7b+...+sweep-vq_bet--vqbet_vanilla/   # VQ-BeT vanilla
├── openvla-7b+...+sweep-vq_bet--vqbet_verb01/    # VQ-BeT verb λ=0.1
├── openvla-7b+...+sweep-vq_bet--vqbet_clip01/    # VQ-BeT clip λ=0.1
├── openvla-7b+...+sweep-oat--oat_vanilla/         # OAT vanilla
├── openvla-7b+...+sweep-oat--oat_verb01/          # OAT verb λ=0.1
├── openvla-7b+...+sweep-oat--oat_clip01/          # OAT clip λ=0.1
├── openvla-7b+...+sweep-quest--quest_vanilla/     # QueST vanilla
├── openvla-7b+...+sweep-quest--quest_verb001/     # QueST verb λ=0.01
└── openvla-7b+...+sweep-quest--quest_clip01/      # QueST clip λ=0.1
```

Each policy dir contains LoRA adapter weights + `--best` variant (best val loss checkpoint).

### Figures

| File | Description |
|------|-------------|
| `figures/calvind_sweep.png` | Full sweep results: training curves + tables |
| `lab_notebooks/calvind_sweep/vanilla_training.png` | Vanilla-only training curves |
| `lab_notebooks/calvind_sweep/verb_sweep.png` | Verb lambda sweep curves |
| `lab_notebooks/calvind_sweep/clip_sweep.png` | CLIP lambda sweep curves |
| `lab_notebooks/calvind_sweep/recon_vs_verb_tradeoff.png` | Pareto frontier |

### Data

| Resource | Path |
|----------|------|
| CALVIN D training | `/data/user_data/yashagar/task_D_D/training/` |
| CALVIN D validation | `/data/user_data/yashagar/task_D_D/validation/` |
| CALVIN RLDS (for policy) | `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/` |

---

## 4. Tokenizer Sweep Results

### Full Sweep Figure

![CALVIN-D Tokenizer Sweep](figures/calvind_sweep.png)

The figure shows training curves (recon loss, aux loss, metrics) for all 27 conditions
across the verb and CLIP lambda sweeps, plus summary tables with per-tokenizer-group
highlighting of Pareto-optimal winners.

### Best Val Recon Loss (per tokenizer, vanilla λ=0)

| Tokenizer | Val Recon | Epochs |
|-----------|-----------|--------|
| VQ-BeT | 0.0141 | 63 (early stop) |
| OAT | 0.0437 | 48 (early stop) |
| QueST | 0.0127 | 200 (full) |

### Winners

For each tokenizer, the lambda that achieves strong aux metric without
disproportionately hurting reconstruction:

- **VQ-BeT**: verb λ=0.1 (mF1 37.8%, recon +16%), clip λ=0.1 (R@1 3.6%, recon +16%)
- **OAT**: verb λ=0.1 (mF1 29.1%, recon -25%), clip λ=0.1 (R@1 2.7%, recon -24%)
- **QueST**: verb λ=0.01 (mF1 26.9%, recon +43%), clip λ=0.1 (R@1 2.2%, recon +231%)

Note: OAT aux heads actually *improved* recon (negative %). This may reflect regularization
from the auxiliary task.

---

## 5. Codebook Utilization

Measured across all training chunks (288K for VQ-BeT, 74K–149K for OAT/QueST) using
`scripts/codebook_utilization_hp.py` with `codes_to_indices` for correct FSQ code counting.

| Tokenizer | Unique codes | Codebook size | Utilization |
|-----------|-------------|---------------|-------------|
| VQ-BeT | 226 | 256 (16×16) | 88.3% |
| OAT | 50 | 1000 | 5.0% |
| QueST | 928 | 1000 | 92.8% |

**Finding**: OAT has genuine codebook collapse on CALVIN (5% utilization). QueST
has healthy utilization (93%). Earlier measurements showed QueST at 5.4% due to a bug
(`.astype(int)` on FSQ float codes truncated values like -0.75 to 0, collapsing
distinct codes). Fixed by using `FSQ.codes_to_indices()` to convert quantized float
vectors to scalar code indices.

---

## 6. Hyperparameter Sweep (18 configs)
Because the codebook utilization was so low. We decided to redo a sweep for the hyperparameters.
Swept **codebook size**, **tokens per chunk**, and **horizon** to find each tokenizer's
optimal configuration for CALVIN. All vanilla (λ=0, no aux heads).

### Scripts & Checkpoints

| Resource | Path |
|----------|------|
| Sweep submission | `scripts/submit_calvind_hp_sweep.sh` |
| Checkpoints | `checkpoints/calvind_hp_sweep/` |

### VQ-BeT Results

Fixed: latent_dim=512, hidden_dim=128, num_mlp_layers=1

| Config | chunk | n_embed | groups | tokens | combos | val_recon | CB util | early_stop |
|--------|-------|---------|--------|--------|--------|-----------|---------|-----------|
| c5/e16/g2 | 5 | 16 | 2 | 2 | 256 | 0.0177 | 92.6% (237/256) | ep 53 |
| **c5/e16/g4** | **5** | **16** | **4** | **4** | **65K** | **0.0085** | **33.4% (21.9K/65K)** | **ep 73** |
| c5/e64/g2 | 5 | 64 | 2 | 2 | 4,096 | 0.0089 | 90.6% (3713/4096) | ep 155 |
| c10/e16/g2 | 10 | 16 | 2 | 2 | 256 | 0.0216 | 91.8% (235/256) | ep 53 |
| c10/e16/g4 | 10 | 16 | 4 | 4 | 65K | 0.0119 | 24.2% (15.8K/65K) | ep 110 |
| c10/e64/g2 | 10 | 64 | 2 | 2 | 4,096 | 0.0127 | 77.1% (3159/4096) | ep 109 |

**Findings**:
- More groups (4 vs 2) halves recon loss — the dominant factor
- Larger codebook (64 vs 16) also helps but less so
- chunk=5 consistently beats chunk=10 (shorter chunks are easier to reconstruct)
- groups=2 configs have ~90% utilization; groups=4 configs drop to 24-33% (65K combos is overkill)
- Best: **c5/e16/g4** (0.0085)

### OAT Results

Fixed: emb_dim=256, enc_depth=2, dec_depth=4, head_dim=64

| Config | horizon | FSQ levels | codebook | regs | tokens | val_recon | CB util | early_stop |
|--------|---------|-----------|----------|------|--------|-----------|---------|-----------|
| h32/f1000/r8 | 32 | [8,5,5,5] | 1000 | 8 | 8 | 0.0461 | 5.0% (50/1000) | ep 60 |
| **h32/f256/r8** | **32** | **[4,4,4,4]** | **256** | **8** | **8** | **0.0452** | **10.2% (26/256)** | **ep 64** |
| h32/f256/r4 | 32 | [4,4,4,4] | 256 | 4 | 4 | 0.0484 | 7.0% (18/256) | ep 51 |
| h32/f64/r4 | 32 | [4,4,4] | 64 | 4 | 4 | 0.0506 | 23.4% (15/64) | ep 43 |
| h16/f256/r4 | 16 | [4,4,4,4] | 256 | 4 | 4 | 0.0558 | 5.1% (13/256) | ep 26 |
| h16/f256/r8 | 16 | [4,4,4,4] | 256 | 8 | 8 | 0.0548 | 5.1% (13/256) | ep 43 |

**Findings**:
- OAT has **low codebook utilization** across all configs (5–23%), though not as extreme as initially reported
- Smaller codebook (64) achieves higher utilization (23.4%) but fewer unique codes (15)
- 8 registers > 4 registers (more tokens = better reconstruction)
- Horizon 32 > 16 (longer context helps OAT's register-based compression)
- Best: **h32/f256/r8** (0.0452)

### QueST Results

Fixed: encoder_dim=256, decoder_dim=256, enc_layers=2, dec_layers=4

| Config | horizon | FSQ levels | codebook | ds | tokens | val_recon | CB util | early_stop |
|--------|---------|-----------|----------|----|--------|-----------|---------|-----------|
| h32/f1000/d4 | 32 | [8,5,5,5] | 1000 | 4 | 8 | 0.0138 | 92.8% (928/1000) | ep 194 |
| h32/f256/d4 | 32 | [4,4,4,4] | 256 | 4 | 8 | 0.0181 | 94.1% (241/256) | ep 188 |
| h32/f256/d8 | 32 | [4,4,4,4] | 256 | 8 | 4 | 0.0193 | 92.2% (236/256) | ep 177 |
| h32/f64/d4 | 32 | [4,4,4] | 64 | 4 | 8 | 0.0195 | 100% (64/64) | ep 197 |
| h16/f256/d4 | 16 | [4,4,4,4] | 256 | 4 | 4 | 0.0158 | 83.6% (214/256) | ep 200 |
| **h16/f256/d2** | **16** | **[4,4,4,4]** | **256** | **2** | **8** | **0.0121** | **98.8% (253/256)** | **ep 186** |

**Findings**:
- QueST has **excellent codebook utilization** (83–100%) across all configs — no collapse
- Earlier reports of low QueST utilization were due to a measurement bug (`.astype(int)` on FSQ floats)
- QueST benefits from more codes (1000 > 256 > 64) with near-full utilization at all sizes
- **h16/f256/d2 is the surprise winner** — shorter horizon with less downsampling
- ds=2 produces 8 tokens from 16 steps, giving fine-grained temporal resolution
- Best: **h16/f256/d2** (0.0121)

### Cross-Tokenizer Comparison

Best config per tokenizer (all recon MSE in normalized [-1,1] action space):

| Tokenizer | Best Config | Val Recon | Tokens | Codebook |
|-----------|------------|-----------|--------|----------|
| **VQ-BeT** | c5/e16/g4 | **0.0085** | 4 | 16 (×4 groups) |
| **QueST** | h16/f256/d2 | 0.0121 | 8 | 256 |
| **OAT** | h32/f256/r8 | 0.0452 | 8 | 256 |

VQ-BeT achieves the lowest recon loss despite being the simplest architecture (MLP only).
OAT's register-based compression is the most aggressive bottleneck, resulting in 4-5×
higher recon loss than VQ-BeT/QueST.

### Aux Loss Retraining (HP Sweep Winners)

Retrain the best VQ-BeT and QueST configs with verb classification and CLIP contrastive
heads (λ=0.1 for both, matching the original sweep's winning lambda).

| Job ID | Tokenizer | Config | Aux Head | Lambda |
|--------|-----------|--------|----------|--------|
| 6724609 | VQ-BeT | c5/e16/g4 | verb | 0.1 |
| 6724610 | VQ-BeT | c5/e16/g4 | clip | 0.1 |
| 6724611 | QueST | h16/f256/d2 | verb | 0.1 |
| 6724612 | QueST | h16/f256/d2 | clip | 0.1 |

Checkpoints saved to `checkpoints/calvind_hp_sweep/{c5e16g4_verb01,c5e16g4_clip01,h16d2_verb01,h16d2_clip01}/`.

---


## 7. Policy Training: From-Scratch MiniVLA (Qwen2.5-0.5B)

Second round using the HP sweep winners (Section 6). Trains MiniVLA from scratch
on CALVIN-D only (no OXE pretraining).

### Model
- **Base VLM**: `prism-qwen25-extra-dinosiglip-224px+0_5b` (Qwen2.5-0.5B + DINOv2/SigLIP)
- **Training**: Full-parameter FSDP (no LoRA), batch_size=16, 50K steps
- **Action tokenizer**: `extra_action_tokenizer` (uses 256 `<extra_i>` tokens added to Qwen vocab)
- **Flash attention**: v2.7.4 (required for memory efficiency)

### Scripts & Checkpoints

| Resource | Path |
|----------|------|
| Submission script | `scripts/submit_calvind_scratch.sh` |
| VLA train script | `<openvla-mini>/vla-scripts/train.py` |
| materialize.py (sweep: prefix) | `<openvla-mini>/prismatic/vla/materialize.py` |
| Checkpoints | `runs/calvind_scratch/` |

### Conditions (10 → 7 active)
Uses HP sweep checkpoints (`checkpoints/calvind_hp_sweep/`):

| # | Tag | Tokenizer | Config | Tokens | Codebook | Tok Recon | Status |
|---|-----|-----------|--------|--------|----------|-----------|--------|
| 1 | sc_bin_baseline | Bin | 256 bins | 7 | 256 | N/A | **Done** |
| 2 | sc_vb_c5e16g4 | VQ-BeT | c5/e16/g4 | 4 | 16 (×4) | 0.0085 | **Done** |
| 3 | sc_vb_c5e64g2 | VQ-BeT | c5/e64/g2 | 2 | 64 (×2) | 0.0089 | **Done** |
| 4 | sc_vb_c10e16g4 | VQ-BeT | c10/e16/g4 | 4 | 16 (×4) | 0.0119 | **Done** |
| 5 | ~~sc_oat_h32f256r8~~ | ~~OAT~~ | ~~h32/f256/r8~~ | ~~8~~ | ~~256~~ | ~~0.0452~~ | Cancelled |
| 6 | ~~sc_oat_h32f1000r8~~ | ~~OAT~~ | ~~h32/f1000/r8~~ | ~~8~~ | ~~1000~~ | ~~0.0461~~ | Cancelled |
| 7 | ~~sc_oat_h32f256r4~~ | ~~OAT~~ | ~~h32/f256/r4~~ | ~~4~~ | ~~256~~ | ~~0.0484~~ | Cancelled |
| 8 | sc_quest_h16f256d2 | QueST | h16/f256/d2 | 8 | 256 | 0.0121 | Running |
| 9 | sc_quest_h32f1000d4 | QueST | h32/f1000/d4 | 8 | 1000 | 0.0138 | Running |
| 10 | sc_quest_h16f256d4 | QueST | h16/f256/d4 | 4 | 256 | 0.0158 | Running |

All 3 OAT conditions were cancelled due to poor tokenizer reconstruction (0.045–0.048,
4-5× worse than VQ-BeT/QueST) and severe codebook collapse (5–10% utilization).

### Training Progress (as of 2026-03-23)

| Condition | Steps | Train Loss | Token Acc | Notes |
|-----------|-------|-----------|-----------|-------|
| bin_baseline | 50000 | 2.52 | 29.5% | Done |
| vb_c5e16g4 | 50000 | 1.32 | 28.1% | Done |
| vb_c5e64g2 | 50000 | 1.33 | 21.9% | Done |
| vb_c10e16g4 | 50000 | 1.19 | 43.8% | Done |
| quest_h16f256d2 | 44203 | 0.51 | 77.3% | Running |
| quest_h32f1000d4 | 2639 | 1.00 | 68.1% | Running |
| quest_h16f256d4 | 18784 | 0.67 | 67.2% | Running |

**Important caveat on token accuracy**: Token accuracy is not directly comparable across
tokenizers. A tokenizer with fewer effective codes (collapsed codebook) will have artificially
high token accuracy because prediction is easier. The training L1 metric is also misleading:
it computes `L1(decode(pred_tokens), decode(gt_tokens))` rather than
`L1(decode(pred_tokens), raw_continuous_gt)`. When both pred and GT decode to the same
degenerate code, L1 is zero even though the actual action error is large.

### Evaluation Plan

A proper **Real L1** eval script has been added to `policy/scripts/evaluate_openvla.py`
(`--eval_real_l1` flag). This computes `L1(decode(pred_tokens), original_continuous_actions)`
by loading raw GT actions from the RLDS val split, giving a fair metric across all tokenizers.
Will be run on best checkpoints once training completes.

Where `<openvla-mini>` = `/data/user_data/wenjiel2/Code/openvla-mini`

---

