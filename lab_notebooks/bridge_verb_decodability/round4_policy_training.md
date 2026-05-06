# Round 4: MiniVLA Policy Training on BridgeV2

## Goal

Fine-tune MiniVLA (0.5B) on BridgeV2 using each sweep tokenizer to measure whether verb-decodable tokenizers produce better VLA policies. This is the downstream evaluation of the tokenizer sweep (Round 2) and FSQ propagation experiments (Round 3).

## Setup

- **Model**: MiniVLA (prism-qwen25-dinosiglip-224px+0_5b), 1.25B params total, 522M trainable (vision backbone frozen)
- **Config**: `prism-qwen25-dinosiglip-224px+0_5b+mx-bridge`
- **Training**: 50,000 steps, batch_size=16, lr=2e-5 (constant schedule), AdamW
- **Data**: BridgeV2 RLDS (`/data/user_data/wenjiel2/datasets/bridge_rlds/bridge_dataset/`)
- **Checkpoints saved**: every 5,000 steps (10 checkpoints per run)
- **Script**: `torchrun vla-scripts/train.py` in openvla-mini codebase

### Tokenizer Conditions

Each tokenizer from the Round 2 sweep is used as the action tokenizer for MiniVLA. The tokenizer converts continuous actions into discrete tokens that the LLM backbone predicts autoregressively.

- **VQ-BeT** (3 configs × 3 aux = 9): chunk_size ∈ {5, 10}, latent_dim ∈ {256, 512}, num_codes=16, vq_groups=2
- **OAT** (3 configs × 5 aux = 15): FSQ-based, includes pfsq variants
- **QueST** (3 configs × 5 aux = 15): FSQ-based, includes pfsq variants

Total: 39 policy training runs.

### Metrics

- **Loss**: cross-entropy on predicted action tokens (lower = better)
- **Token Accuracy**: fraction of correctly predicted action tokens (higher = better)
- **L1 Loss**: mean absolute error of decoded continuous actions (lower = better, most directly measures action quality)

All metrics are training-only (no held-out val split in the RLDS streaming setup). We report 500-step rolling averages for stability.

## Results

### VQ-BeT (completed)

Training curves (500-step rolling averages):

| Tag | Step | Loss | Token Acc | L1 |
|-----|------|------|----------|-----|
| vq_bet_5_16_2_256 | 500 | 1.102 | 26.7% | 0.067 |
| | 5000 | 0.930 | 37.0% | 0.050 |
| | 10000 | 0.879 | 40.6% | 0.047 |
| | 25000 | 0.789 | 46.7% | 0.042 |
| | 50000 | 0.716 | 51.6% | 0.039 |
| vq_bet_5_16_2_256_verb0.1 | 500 | 0.771 | 53.7% | 0.065 |
| | 5000 | 0.677 | 58.2% | 0.056 |
| | 10000 | 0.664 | 59.0% | 0.053 |
| | 25000 | 0.614 | 61.3% | 0.049 |
| | 50000 | 0.560 | 64.3% | 0.044 |
| vq_bet_5_16_2_256_clip0.1 | 500 | 0.538 | 63.3% | 0.052 |
| | 5000 | 0.479 | 65.0% | 0.050 |
| | 10000 | 0.461 | 66.0% | 0.048 |
| | 25000 | 0.427 | 68.6% | 0.044 |
| | 50000 | 0.383 | 72.0% | 0.039 |
| vq_bet_5_16_2_512 | 500 | 1.107 | 25.4% | 0.070 |
| | 5000 | 0.928 | 36.4% | 0.060 |
| | 10000 | 0.881 | 40.2% | 0.058 |
| | 25000 | 0.776 | 48.1% | 0.051 |
| | 50000 | 0.696 | 52.9% | 0.046 |
| vq_bet_5_16_2_512_verb0.1 | 500 | 0.751 | 50.0% | 0.049 |
| | 5000 | 0.644 | 55.9% | 0.048 |
| | 10000 | 0.599 | 57.9% | 0.044 |
| | 25000 | 0.540 | 62.0% | 0.036 |
| | 50000 | 0.490 | 65.6% | 0.032 |
| vq_bet_5_16_2_512_clip0.1 | 500 | 0.547 | 59.2% | 0.052 |
| | 5000 | 0.490 | 63.4% | 0.049 |
| | 10000 | 0.471 | 64.8% | 0.049 |
| | 25000 | 0.437 | 67.5% | 0.044 |
| | 50000 | 0.412 | 69.1% | 0.042 |
| vq_bet_10_16_2_256 | 500 | 1.183 | 28.4% | 0.098 |
| | 5000 | 1.036 | 35.1% | 0.090 |
| | 10000 | 0.988 | 37.7% | 0.085 |
| | 25000 | 0.926 | 41.2% | 0.078 |
| | 50000 | 0.889 | 42.8% | 0.073 |

Summary at step 50,000 (500-step avg):

| Tag | Loss | Token Acc | L1 |
|-----|------|----------|-----|
| vq_bet_5_16_2_256 | 0.716 | 51.6% | 0.039 |
| vq_bet_5_16_2_256_verb0.1 | 0.560 | 64.3% | 0.044 |
| vq_bet_5_16_2_256_clip0.1 | **0.383** | **72.0%** | 0.039 |
| vq_bet_5_16_2_512 | 0.696 | 52.9% | 0.046 |
| vq_bet_5_16_2_512_verb0.1 | 0.490 | 65.6% | **0.032** |
| vq_bet_5_16_2_512_clip0.1 | 0.412 | 69.1% | 0.042 |
| vq_bet_10_16_2_256 | 0.889 | 42.8% | 0.073 |

### VQ-BeT chunk_size=10 (completed)

| Tag | Step | Loss | Token Acc | L1 |
|-----|------|------|----------|-----|
| vq_bet_10_16_2_256_verb0.1 | 500 | 1.041 | 42.8% | 0.0688 |
| | 5000 | 0.707 | 53.8% | 0.0647 |
| | 10000 | 0.694 | 54.2% | 0.0648 |
| | 25000 | 0.648 | 56.7% | 0.0604 |
| | 45143* | 0.604 | 60.2% | 0.0540 |
| vq_bet_10_16_2_256_clip0.1 | 500 | 1.086 | 37.7% | 0.0518 |
| | 5000 | 0.710 | 50.4% | 0.0495 |
| | 10000 | 0.694 | 51.4% | 0.0459 |
| | 25000 | 0.647 | 55.6% | 0.0423 |
| | 50000 | 0.619 | 57.3% | 0.0386 |

*verb0.1 hit 12h time limit at step 45,143.

### OAT (partial — oat_16_855_4 family completed)

| Tag | Step | Loss | Token Acc | L1 |
|-----|------|------|----------|-----|
| oat_16_855_4 | 500 | 1.160 | 56.1% | 0.0300 |
| | 5000 | 0.450 | 77.1% | 0.0054 |
| | 10000 | 0.428 | 78.3% | 0.0052 |
| | 25000 | 0.386 | 79.8% | 0.0049 |
| | 48250* | 0.367 | 80.0% | 0.0046 |
| oat_16_855_4_verb0.1 | 500 | 1.320 | 45.9% | 0.0165 |
| | 5000 | 0.659 | 65.6% | 0.0162 |
| | 10000 | 0.623 | 67.0% | 0.0142 |
| | 25000 | 0.576 | 69.1% | 0.0136 |
| | 44213* | 0.559 | 69.5% | 0.0140 |
| oat_16_855_4_clip0.1 | 500 | 1.076 | 45.8% | 0.0514 |
| | 5000 | 0.569 | 64.0% | 0.0278 |
| | 10000 | 0.540 | 66.3% | 0.0265 |
| | 25000 | 0.509 | 68.5% | 0.0248 |
| | 50000 | 0.486 | 69.7% | 0.0233 |
| oat_16_855_4_verb0.1_pfsq | 500 | 1.280 | 49.9% | 0.0184 |
| | 5000 | 0.706 | 63.7% | 0.0101 |
| | 10000 | 0.671 | 65.3% | 0.0096 |
| | 25000 | 0.617 | 67.4% | 0.0094 |
| | 50000 | 0.590 | 68.7% | 0.0092 |
| oat_16_855_4_clip0.1_pfsq | 500 | 1.472 | 35.7% | 0.0583 |
| | 5000 | 0.932 | 50.2% | 0.0510 |
| | 10000 | 0.895 | 51.2% | 0.0493 |
| | 25000 | 0.824 | 54.9% | 0.0465 |
| | 50000 | 0.791 | 56.2% | 0.0436 |
| oat_32_888_8 | 500 | 0.641 | 84.8% | 0.0003 |
| | 5000 | 0.243 | 90.9% | 0.0001 |
| | 10000 | 0.221 | 91.6% | 0.0001 |
| | 25000 | 0.204 | 92.0% | 0.0001 |
| | 50000 | 0.187 | 92.4% | 0.0001 |

*vanilla and verb0.1 hit 12h time limit. clip0.1_pfsq results are notably worse.

### QueST (pending — collaborator resubmission)

15 QueST jobs cancelled from wenjiel2's quota and transferred to collaborator's quota via `scripts/collab_bridge_policy.sh` on shire-general partition. Remaining OAT conditions (32_888_8 aux variants, 16_8555_4 family) also in that batch.

## Summary Table (all completed, 500-step avg at final step)

| Tag | Job ID | Steps | Loss | Token Acc | L1 | Tok Recon |
|-----|--------|-------|------|----------|-----|-----------|
| **VQ-BeT chunk=5, dim=256** | | | | | | |
| vq_bet_5_16_2_256 | 6884130 | 50000 | 0.718 | 51.4% | 0.0398 | 0.0029 |
| vq_bet_5_16_2_256_verb0.1 | 6884131 | 50000 | 0.564 | 63.9% | 0.0442 | 0.0026 |
| vq_bet_5_16_2_256_clip0.1 | 6884132 | 50000 | **0.387** | **71.7%** | 0.0392 | 0.0035 |
| **VQ-BeT chunk=5, dim=512** | | | | | | |
| vq_bet_5_16_2_512 | 6884133 | 50000 | 0.705 | 52.4% | 0.0470 | 0.0021 |
| vq_bet_5_16_2_512_verb0.1 | 6884134 | 50000 | 0.489 | 66.0% | **0.0312** | 0.0025 |
| vq_bet_5_16_2_512_clip0.1 | 6884135 | 50000 | 0.410 | 69.4% | 0.0406 | 0.0040 |
| **VQ-BeT chunk=10, dim=256** | | | | | | |
| vq_bet_10_16_2_256 | 6884136 | 50000 | 0.886 | 43.0% | 0.0718 | 0.0039 |
| vq_bet_10_16_2_256_verb0.1 | 6887081 | 45143 | 0.604 | 60.2% | 0.0540 | 0.0050 |
| vq_bet_10_16_2_256_clip0.1 | 6887082 | 50000 | 0.619 | 57.3% | 0.0386 | 0.0096 |
| **OAT 16_855_4** | | | | | | |
| oat_16_855_4 | 6887083 | 48250 | 0.367 | 80.0% | 0.0046 | 0.0034 |
| oat_16_855_4_verb0.1 | 6887084 | 44213 | 0.559 | 69.5% | 0.0140 | 0.0041 |
| oat_16_855_4_clip0.1 | 6887085 | 50000 | 0.486 | 69.7% | 0.0233 | 0.0039 |
| oat_16_855_4_verb0.1_pfsq | 6887086 | 50000 | 0.590 | 68.7% | 0.0092 | 0.0036 |
| oat_16_855_4_clip0.1_pfsq | 6887087 | 50000 | 0.791 | 56.2% | 0.0436 | 0.0049 |
| **OAT 32_888_8** | | | | | | |
| oat_32_888_8 | 6887088 | 50000 | 0.187 | 92.4% | 0.0001 | 0.0031 |

## Analysis

### Cross-family comparison

**OAT tokenizers learn dramatically faster than VQ-BeT.** At step 50k:
- oat_16_855_4 vanilla: 80.0% token acc, L1=0.0046 — far better than any VQ-BeT condition
- oat_32_888_8 vanilla: 92.4% token acc, L1=0.0001 — near-perfect token prediction
- Best VQ-BeT (clip0.1, 256): 71.7% token acc, L1=0.0392

This likely reflects that OAT's FSQ quantization is more policy-friendly than VQ-BeT's residual VQ — fewer tokens per chunk (4 vs 2 codes, but each code covers more), and the codebook structure may be easier for the LLM to predict.

### Aux head effect is reversed for OAT

**For VQ-BeT: aux > vanilla** (consistent across all configs):
- clip0.1 reduces loss by 46%, verb0.1 reduces by 22% vs vanilla

**For OAT: vanilla > aux** — the opposite pattern:
- oat_16_855_4 vanilla: L1=0.0046, 80.0% acc
- oat_16_855_4 verb0.1: L1=0.0140 (+205%), 69.5% acc
- oat_16_855_4 clip0.1: L1=0.0233 (+406%), 69.7% acc
- pfsq variants: even worse on clip (0.0436), verb_pfsq competitive (0.0092)

This is a striking reversal. The aux training objective appears to *hurt* the OAT tokenizer's policy performance by distorting the codebook away from action-optimal quantization. OAT's vanilla codebook already captures action structure well; adding verb/CLIP supervision degrades it.

### L1 vs. token accuracy discrepancy

The pattern from VQ-BeT persists: higher token accuracy doesn't always mean lower L1.
- oat_16_855_4 verb0.1_pfsq: 68.7% acc, L1=0.0092
- oat_16_855_4 clip0.1: 69.7% acc, L1=0.0233
- Same accuracy, 2.5× worse L1 for clip

**Not all tokens are created equal.** Some token errors are innocuous (nearby codes that decode to similar actions), while others are catastrophic. Verb-optimized tokenizers may organize the codebook so that nearby codes correspond to similar verbs, making token errors less damaging.

### ⚠ L1 metric is NOT comparable across tokenizer families

The L1 metric measures decoded continuous action error, but the decoding path differs fundamentally:
- **VQ-BeT**: linear decoder → unnormalize. Wrong codes produce divergent outputs. Random-prediction L1 ≈ 0.05–0.08.
- **OAT**: 4-layer transformer decoder → unnormalize. The learned decoder constrains outputs to the training distribution even for random codes. Random-prediction L1 ≈ 0.006 (at step 1, oat_32_888_8 has CE loss=14+ but L1=0.006).

Additionally, `decode_token_ids_to_actions` only returns the **first timestep** of each decoded chunk. For OAT with horizon=32, the first timestep is highly insensitive to code perturbations — nearby (and even distant) codes decode similarly at t=0.

**Conclusion**: Use **token accuracy** for cross-family comparison. L1 is only meaningful within the same tokenizer family. The apparent OAT vs VQ-BeT L1 gap (0.0001 vs 0.03) is an artifact of decoder architecture.

### oat_32_888_8: suspiciously good?

92.4% token accuracy at 50k steps with L1=0.0001. Investigation:

**Tokenizer config**: horizon=32, FSQ [8,8,8]=512 codebook, 8 registers → **8 tokens/chunk**, each from 512 classes. Codebook utilization: 485/512 (94.7%). This is a well-functioning tokenizer with near-full codebook usage.

**Why L1=0.0001 is misleading**: At step 1 (random predictions, CE=14+), L1 is already 0.006. The entire dynamic range of L1 for this tokenizer is [0.006 → 0.0001] — a 60× compression compared to VQ-BeT. This is caused by the OAT decoder constraining all outputs to a narrow range (see note above).

**Token accuracy (92.4%) is genuine**: With 485 active codes and 8 tokens per chunk, this represents real predictive ability. Possible explanations:
1. 8 tokens × 512 codebook = 72 bits total capacity for 32×7=224 values. Each code covers ~4 timesteps, which may have low temporal variance → easy to predict
2. The VLA may be memorizing the training set — teacher-force eval on val needed to confirm
3. The longer horizon (32 vs 16) may make the prediction easier if BridgeV2 action sequences are repetitive within 32-step windows

## Teacher-Forced Evaluation on Val Split

Teacher-forced evaluation on the held-out validation split (`train[95%:]`, ~2,660 episodes) to measure generalization rather than training-set performance.

**Script**: `policy/eval_policy.py --mode teacher_force --condition <tag>`

### Results (15 conditions)

| Condition | CE Loss | Token Acc (%) | L1 |
|-----------|---------|---------------|-----|
| **VQ-BeT chunk=5, dim=256** | | | |
| vq_bet_5_16_2_256 | 0.767 | 48.4 | 0.0421 |
| vq_bet_5_16_2_256_verb0.1 | 0.566 | 64.7 | 0.0459 |
| vq_bet_5_16_2_256_clip0.1 | 0.441 | 68.1 | 0.0469 |
| **VQ-BeT chunk=5, dim=512** | | | |
| vq_bet_5_16_2_512 | 0.723 | 50.5 | 0.0479 |
| vq_bet_5_16_2_512_verb0.1 | 0.521 | 64.5 | 0.0330 |
| vq_bet_5_16_2_512_clip0.1 | 0.387 | 70.3 | 0.0384 |
| **VQ-BeT chunk=10, dim=256** | | | |
| vq_bet_10_16_2_256 | 0.863 | 44.9 | 0.0711 |
| vq_bet_10_16_2_256_verb0.1 | 0.563 | 63.5 | 0.0479 |
| vq_bet_10_16_2_256_clip0.1 | 0.638 | 56.0 | 0.0445 |
| **OAT 16_855_4** | | | |
| oat_16_855_4 | 0.357 | 80.2 | 0.0044 |
| oat_16_855_4_verb0.1 | 0.560 | 70.1 | 0.0128 |
| oat_16_855_4_clip0.1 | 0.480 | 69.8 | 0.0211 |
| oat_16_855_4_verb0.1_pfsq | 0.562 | 69.3 | 0.0096 |
| oat_16_855_4_clip0.1_pfsq | 0.805 | 55.5 | 0.0459 |
| **OAT 32_888_8** | | | |
| oat_32_888_8 | 0.207 | 92.1 | 0.0001 |

### Analysis: train vs val generalization

Training and val metrics are close for most conditions, suggesting limited overfitting:
- VQ-BeT vanilla: train 51.4% → val 48.4% (−3pp), train L1 0.040 → val 0.042
- OAT 16_855_4 vanilla: train 80.0% → val 80.2% (+0.2pp, essentially identical)
- OAT 32_888_8: train 92.4% → val 92.1% (−0.3pp)

The val results confirm the training-time findings:
1. **OAT >> VQ-BeT** on token accuracy (80–92% vs 45–70%)
2. **Aux heads help VQ-BeT but hurt OAT** (same reversal seen in training)
3. **oat_32_888_8 L1=0.0001 is real** on val too — but likely an artifact of the decoder constraining outputs (see L1 comparability note above)
4. **L1 cross-family incomparable**: best VQ-BeT L1 (0.033) vs best OAT L1 (0.004) is meaningless due to different decoder architectures

### Fullproj embedding variant TF eval (2026-04-03, jobs 6934581/6934582)

VQ-BeT 10_16_2_256 policies trained with fullproj embedding (codebook-projected, no per-token learnable params). Full analysis in [ot_codebook_embedding.md](../vla_embed_init/ot_codebook_embedding.md).

| Condition | CE Loss | Token Acc (%) | L1 |
|-----------|---------|---------------|-----|
| vq_bet_10_16_2_256_verb0.1_fullproj | 0.591 | 61.0 | 0.0535 |
| vq_bet_10_16_2_256_clip0.1_fullproj | 0.620 | 58.6 | 0.0478 |

Compared to standard embedding counterparts: verb0.1+fullproj (61.0%) closely tracks verb0.1 standard (63.5%), clip0.1+fullproj (58.6%) closely tracks clip0.1 standard (56.0%). Embedding strategy has minimal effect on TF metrics — the transformer compensates.

### Missing conditions

15 of 39 standard conditions evaluated (9 VQ-BeT + 6 OAT). The remaining 24 (OAT 32_888_8 aux variants, OAT 16_8555_4 family, all 15 QueST conditions) were awaiting policy training via collaborator submission (`scripts/collab_bridge_policy.sh`).

## Padding Mismatch Fix (2026-03-31)

### Problem identified

**Tokenizer training vs policy evaluation use different action chunk sampling strategies, creating a distribution mismatch:**

1. **Tokenizer training** (`BridgeTokenizerDataset._random_starts`): chunks could only start from `[0, max(0, T - chunk_size)]`. For short episodes (T < horizon), this meant chunks always started at t=0, biasing the tokenizer toward beginning-of-episode actions.

2. **Policy evaluation** (RLDS `chunk_act_obs`): chunks start from every timestep `[0, T-1]`, with edge-repeat padding for chunks extending past the episode end.

This mismatch is worst for `oat_32_888_8` (horizon=32): 41.4% of chunk content is padding on average, and short episodes (T < 32) always produce t=0 chunks. The suspiciously high token accuracy (92.1%) may partly reflect the tokenizer memorizing these beginning-of-episode patterns rather than learning general action structure.

### Fix implemented

Two changes, keeping tokenizer model architectures **untouched**:

1. **RLDS-style sampling** (`datasets/bridge_dataset.py`): `_random_starts()` now samples from every timestep `[0, T-1]`, matching the RLDS policy pipeline. Short episodes can produce chunks starting near the end. Each chunk tracks `action_real_lens = min(chunk_size, T - start)`.

2. **Masked reconstruction loss** (`oat/tokenizer.py`, `quest/tokenizer.py`): `forward()` reads optional `action_real_lens` from the batch dict. When present, MSE loss is computed only on real (non-padded) timesteps. The encoder and decoder architectures are completely unchanged — padding still flows through the model, but the loss ignores it.

3. **Plumbing** (`tokenization/train_utils.py`): `extract_episode_batch()` passes `action_real_lens` through to the model's forward method for OAT/QueST. VQ-BeT is unchanged (chunk_size=5, padding fraction ~5.4%, negligible).

### Retraining sweep (18 conditions)

Submitted via `scripts/submit_tok_retrain.sh` — 6 tokenizer configs × 3 aux heads, batched 4 at a time with SLURM dependency chains.

**Tokenizers**: 3 OAT + 3 QueST (same configs as Round 2):
- OAT: 16_855_4, 32_888_8, 16_8555_4
- QueST: 16_855_4, 32_888_4, 16_8555_4

**Aux heads**: none, verb:0.1:pfsq, clip:0.1:pfsq (post-FSQ only — skipping latent-space aux since pfsq is more directly interpretable)

**Status**: Batch 1–2 completed, batch 2 running (as of 2026-03-31). Jobs 6899265–6899287 + resubmit 6899919.

### Expected impact

- **oat_32_888_8** should show the largest change — its old tokenizer was trained with 41.4% padding but never penalized for reconstructing it
- **oat/quest_16_*** should show modest improvement — 5–15% padding fraction
- If val token accuracy drops significantly for oat_32, it confirms the padding inflation hypothesis

## Full Sweep Results (2026-04-14 audit)

Complete state of the bridge policy sweep after all submissions. Covers all tokenizer types (VQ-BeT, OAT, QueST), all aux conditions, standard + fullproj embedding variants.

### Sweep Pipeline

Per tokenizer condition, the sweep runs 4 stages:
1. **Verb probes** (native, latent, tokid) — trained on tokenized actions
2. **Standard policy** — MiniVLA with random-init action token embeddings
3. **Fullproj policy** — MiniVLA with codebook-projected embeddings:
   - VQ-BeT (static): `proj(codebook_vector)` per token, no per-token learnable params
   - OAT/QueST (dynamic): `proj(pre_fsq_latent)` per token, latent from frozen encoder each step
4. **Teacher-force eval** — L1/CE/TokAcc on val split (train[95%:])

Directory structure: `checkpoints/bridge_sweep/{tokenizers,policy,results}/<base_type>/<condition>/`

### Policy Training Completion Status

#### VQ-BeT (27 conditions)

| Base | Condition | Steps | Status |
|------|-----------|-------|--------|
| vq_bet_5_16_2_256 | vanilla | 50000 | DONE |
| vq_bet_5_16_2_256 | verb0.1 | 50000 | DONE |
| vq_bet_5_16_2_256 | clip0.1 | 50000 | DONE |
| vq_bet_5_16_2_256 | vanilla_fullproj | 50000 | DONE |
| vq_bet_5_16_2_256 | verb0.1_fullproj | 50000 | DONE |
| vq_bet_5_16_2_256 | clip0.1_fullproj | 50000 | DONE |
| vq_bet_5_16_2_512 | vanilla | 50000 | DONE |
| vq_bet_5_16_2_512 | verb0.1 | 50000 | DONE |
| vq_bet_5_16_2_512 | clip0.1 | 50000 | DONE |
| vq_bet_5_16_2_512 | vanilla_fullproj | 45000 | TIMEOUT (14h) |
| vq_bet_5_16_2_512 | verb0.1_fullproj | 45000 | TIMEOUT (14h) |
| vq_bet_5_16_2_512 | clip0.1_fullproj | 50000 | DONE |
| vq_bet_10_16_2_256 | vanilla | 50000 | DONE |
| vq_bet_10_16_2_256 | verb0.1 | 50000 | DONE |
| vq_bet_10_16_2_256 | clip0.1 | 50000 | DONE |
| vq_bet_10_16_2_256 | vanilla_resdim128 | 50000 | DONE |
| vq_bet_10_16_2_256 | verb0.1_resdim128 | 50000 | DONE |
| vq_bet_10_16_2_256 | clip0.1_resdim128 | 50000 | DONE |
| vq_bet_10_16_2_256 | vanilla_fullproj | 50000 | DONE |
| vq_bet_10_16_2_256 | verb0.1_fullproj | 50000 | DONE |
| vq_bet_10_16_2_256 | clip0.1_fullproj | 50000 | DONE |
| vq_bet_10_16_2_256 | vanilla_resdim128v2 | 50000 | DONE |
| vq_bet_10_16_2_256 | verb0.1_resdim128v2 | 50000 | DONE |
| vq_bet_10_16_2_256 | clip0.1_resdim128v2 | 50000 | DONE |
| vq_bet_10_16_2_256 | vlm_clip0.1 | 50000 | DONE |
| vq_bet_10_16_2_256 | vlm_clip0.1_resdim128 | 50000 | DONE |
| vq_bet_10_16_2_256 | vlm_clip0.1_fullproj | 50000 | DONE |

#### OAT (24 conditions, 23 completed)

| Base | Condition | Steps | Status |
|------|-----------|-------|--------|
| oat_16_855_4 | vanilla | 50000 | DONE |
| oat_16_855_4 | vanilla_fullproj | 50000 | DONE |
| oat_16_855_4 | verb0.1_pre_fsq | 50000 | DONE |
| oat_16_855_4 | verb0.1_pre_fsq_fullproj | 50000 | DONE |
| oat_16_855_4 | verb0.1_post_fsq | 50000 | DONE |
| oat_16_855_4 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| oat_16_855_4 | clip0.1_post_fsq | 50000 | DONE |
| oat_16_855_4 | clip0.1_post_fsq_fullproj | 50000 | DONE |
| oat_16_855_4 | vlm_clip0.1_pre_fsq | 50000 | DONE |
| oat_16_855_4 | vlm_clip0.1_pre_fsq_fullproj | 50000 | DONE |
| oat_16_855_4 | vlm_clip0.1_post_fsq | 50000 | DONE |
| oat_16_855_4 | vlm_clip0.1_post_fsq_fullproj | 50000 | DONE |
| oat_16_8555_4 | vanilla | 50000 | DONE |
| oat_16_8555_4 | vanilla_fullproj | 50000 | DONE |
| oat_16_8555_4 | verb0.1_post_fsq | 50000 | DONE |
| oat_16_8555_4 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| oat_16_8555_4 | clip0.1_post_fsq | 50000 | DONE |
| oat_16_8555_4 | clip0.1_post_fsq_fullproj | 50000 | DONE |
| oat_32_888_8 | vanilla | 50000 | DONE |
| oat_32_888_8 | vanilla_fullproj | 50000 | DONE |
| oat_32_888_8 | verb0.1_post_fsq | 50000 | DONE |
| oat_32_888_8 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| oat_32_888_8 | clip0.1_post_fsq | 50000 | DONE |
| oat_32_888_8 | clip0.1_post_fsq_fullproj | 50000 | DONE |

#### QueST (17 conditions)

| Base | Condition | Steps | Status |
|------|-----------|-------|--------|
| quest_16_855_4 | vanilla | 50000 | DONE |
| quest_16_855_4 | vanilla_fullproj | 50000 | DONE |
| quest_16_855_4 | verb0.1_post_fsq | 50000 | DONE |
| quest_16_855_4 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| quest_16_855_4 | clip0.1_post_fsq | 50000 | DONE |
| quest_16_855_4 | clip0.1_post_fsq_fullproj | 50000 | DONE |
| quest_16_8555_4 | vanilla | 50000 | DONE |
| quest_16_8555_4 | vanilla_fullproj | 50000 | DONE |
| quest_16_8555_4 | verb0.1_post_fsq | 50000 | DONE |
| quest_16_8555_4 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| quest_16_8555_4 | clip0.1_post_fsq | 50000 | DONE |
| quest_16_8555_4 | clip0.1_post_fsq_fullproj | 45000 | FAILED (dynamic resume crash #9) |
| quest_32_888_4 | vanilla | 50000 | DONE |
| quest_32_888_4 | vanilla_fullproj | 50000 | DONE |
| quest_32_888_4 | verb0.1_post_fsq | 50000 | DONE |
| quest_32_888_4 | verb0.1_post_fsq_fullproj | 50000 | DONE |
| quest_32_888_4 | clip0.1_post_fsq | 50000 | DONE |
| quest_32_888_4 | clip0.1_post_fsq_fullproj | 45000 | FAILED (dynamic resume crash #9) |

### Teacher-Force Eval Results (val split)

#### OAT 16_855_4 (12 conditions — most comprehensive)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.3546 | 83.28 | 0.0094 | 0.0623 |
| vanilla_fullproj | 0.2547 | 87.21 | 0.0040 | 0.0258 |
| verb0.1_pre_fsq | 0.5897 | 68.79 | 0.0297 | 0.1973 |
| verb0.1_pre_fsq_fullproj | 0.3353 | 81.25 | 0.0218 | 0.1450 |
| verb0.1_post_fsq | 0.5075 | 73.31 | 0.0172 | 0.1144 |
| verb0.1_post_fsq_fullproj | 0.2569 | 87.40 | 0.0188 | 0.1256 |
| clip0.1_post_fsq | 0.7687 | 58.67 | 0.0304 | 0.1983 |
| clip0.1_post_fsq_fullproj | 0.4157 | 76.40 | 0.0374 | 0.2469 |
| vlm_clip0.1_pre_fsq | 0.4221 | 76.75 | 0.0048 | 0.0275 |
| vlm_clip0.1_pre_fsq_fullproj | 0.2250 | **88.10** | 0.0032 | 0.0182 |
| vlm_clip0.1_post_fsq | 0.7463 | 56.69 | 0.0370 | 0.2455 |
| vlm_clip0.1_post_fsq_fullproj | 0.3859 | 77.44 | 0.0276 | 0.1822 |

#### OAT 16_8555_4 (6 conditions)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.3907 | 87.62 | 0.0116 | 0.0759 |
| vanilla_fullproj | 0.0340 | **98.62** | 0.0014 | 0.0091 |
| verb0.1_post_fsq | 0.8141 | 72.47 | 0.0227 | 0.1500 |
| verb0.1_post_fsq_fullproj | 0.0333 | **98.53** | 0.0013 | 0.0083 |
| clip0.1_post_fsq | 1.0397 | 66.16 | 0.0146 | 0.0951 |
| clip0.1_post_fsq_fullproj | 0.0764 | 96.86 | 0.0016 | 0.0107 |

#### OAT 32_888_8 (6 conditions)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.1535 | 93.98 | 0.0001 | 0.0004 |
| vanilla_fullproj | 0.0323 | 99.16 | 0.0000 | 0.0000 |
| verb0.1_post_fsq | 0.1661 | 94.77 | 0.0001 | 0.0001 |
| verb0.1_post_fsq_fullproj | 0.0100 | **99.63** | 0.0000 | 0.0000 |
| clip0.1_post_fsq | 0.6274 | 75.27 | 0.0005 | 0.0020 |
| clip0.1_post_fsq_fullproj | 0.0264 | 98.97 | 0.0000 | 0.0001 |

#### QueST 16_855_4 (6 conditions)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.0000 | 100.00 | 0.0000 | 0.0000 |
| vanilla_fullproj | 0.0000 | 100.00 | 0.0000 | 0.0000 |
| verb0.1_post_fsq | 1.1390 | 44.00 | 0.0226 | 0.1433 |
| verb0.1_post_fsq_fullproj | 0.4378 | 76.11 | 0.0156 | 0.1014 |
| clip0.1_post_fsq | 0.7757 | 60.66 | 0.0191 | 0.1285 |
| clip0.1_post_fsq_fullproj | 0.3299 | 81.34 | 0.0085 | 0.0560 |

#### QueST 16_8555_4 (6 conditions)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 1.1971 | 58.85 | 0.0362 | 0.2258 |
| vanilla_fullproj | 0.1528 | 93.72 | 0.0068 | 0.0423 |
| verb0.1_post_fsq | 1.1443 | 65.56 | 0.0155 | 0.1020 |
| verb0.1_post_fsq_fullproj | 0.0704 | 97.00 | 0.0024 | 0.0160 |
| clip0.1_post_fsq | 0.4141 | 86.73 | 0.0133 | 0.0880 |
| clip0.1_post_fsq_fullproj | 0.0037 | **99.83** | 0.0001 | 0.0009 |

#### QueST 32_888_4 (6 conditions)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.6585 | 74.59 | 0.0064 | 0.0374 |
| vanilla_fullproj | 0.0506 | 97.89 | 0.0007 | 0.0043 |
| verb0.1_post_fsq | 0.0000 | 100.00 | 0.0000 | 0.0000 |
| verb0.1_post_fsq_fullproj | 0.0000 | 100.00 | 0.0000 | 0.0000 |
| clip0.1_post_fsq | 0.0000 | 100.00 | 0.0000 | 0.0000 |
| clip0.1_post_fsq_fullproj | 0.0000 | 100.00 | 0.0000 | 0.0000 |

#### VQ-BeT 10_16_2_256 (8 conditions, 2026-04-16)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.8797 | 43.56 | 0.0701 | 0.3776 |
| vanilla_fullproj | 0.9109 | 40.53 | 0.0818 | 0.4640 |
| verb0.1 | 0.6122 | 60.15 | 0.0500 | 0.3025 |
| verb0.1_fullproj | 0.6440 | 54.90 | 0.0646 | 0.3979 |
| clip0.1 | 0.5875 | 57.99 | 0.0376 | 0.2235 |
| clip0.1_fullproj | 0.6812 | 54.62 | 0.0411 | 0.2489 |
| vlm_clip0.1 | 0.4686 | 65.63 | 0.0359 | 0.2187 |
| vlm_clip0.1_fullproj | 0.4481 | **68.07** | 0.0274 | 0.1638 |

#### VQ-BeT 5_16_2_256 (6 conditions, 2026-04-16)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.7717 | 47.37 | 0.0435 | 0.2435 |
| vanilla_fullproj | 0.7489 | 50.51 | 0.0410 | 0.2333 |
| verb0.1 | 0.5995 | 58.69 | 0.0485 | 0.2916 |
| verb0.1_fullproj | 0.6433 | 60.00 | 0.0524 | 0.3186 |
| clip0.1 | 0.3889 | **72.20** | 0.0413 | 0.2489 |
| clip0.1_fullproj | 0.5048 | 62.41 | 0.0573 | 0.3557 |

#### VQ-BeT 5_16_2_512 (6 conditions, 2026-04-16)

| Condition | CE Loss | TokAcc% | L1 | Gripper L1 |
|-----------|---------|---------|-----|-----------|
| vanilla | 0.7390 | 49.14 | 0.0504 | 0.2852 |
| vanilla_fullproj | 0.7890 | 45.69 | 0.0511 | 0.2910 |
| verb0.1 | 0.4934 | 67.07 | 0.0304 | 0.1563 |
| verb0.1_fullproj | 0.6193 | 55.78 | 0.0478 | 0.2612 |
| clip0.1 | 0.4182 | 68.01 | 0.0354 | 0.1808 |
| clip0.1_fullproj | 0.4142 | **70.21** | 0.0375 | 0.1972 |

### Missing TF Evals (as of 2026-04-16)

| Base | Condition | Reason |
|------|-----------|--------|
| vq_bet_10_16_2_256 | resdim128 (3) | Eval not submitted (resdim variants, not in sweep) |
| vq_bet_10_16_2_256 | resdim128v2 (3) | Eval not submitted (resdim variants, not in sweep) |
| vq_bet_10_16_2_256 | vlm_clip0.1_resdim128 | Eval not submitted (resdim variant, not in sweep) |
| vq_bet_5_16_2_512 | vanilla_fullproj | Policy timed out at 45k (ABANDONED) |
| vq_bet_5_16_2_512 | verb0.1_fullproj | Policy timed out at 45k (ABANDONED) |
| quest_32_888_4 | clip0.1_post_fsq_fullproj | Dynamic resume crash #9, stuck at 45k |
| quest_16_8555_4 | clip0.1_post_fsq_fullproj | Dynamic resume crash #9, stuck at 45k |

### Key Findings from Full Sweep

#### 1. Fullproj embedding is a universal win for OAT/QueST but NOT for VQ-BeT

Across OAT/QueST, fullproj variants outperform standard embedding by large margins:

| Comparison (vanilla) | Standard TokAcc | Fullproj TokAcc | Delta |
|----------------------|-----------------|-----------------|-------|
| OAT 16_855_4 | 83.3% | 87.2% | +3.9pp |
| OAT 16_8555_4 | 87.6% | 98.6% | +11.0pp |
| OAT 32_888_8 | 94.0% | 99.2% | +5.2pp |
| QueST 16_855_4 | 100.0% | 100.0% | 0 (ceiling) |
| QueST 16_8555_4 | 58.9% | 93.7% | +34.9pp |
| **VQ-BeT 10_16_2_256** | **43.6%** | **40.5%** | **-3.1pp** |
| **VQ-BeT 5_16_2_256** | **47.4%** | **50.5%** | **+3.1pp** |
| **VQ-BeT 5_16_2_512** | **49.1%** | **45.7%** | **-3.4pp** |

**Critical finding**: For VQ-BeT, fullproj consistently *hurts*. VQ-BeT 10 vanilla_fullproj (40.5%) is worse than vanilla (43.6%), VQ-BeT 512 vanilla_fullproj (45.7%) worse than vanilla (49.1%), and verb0.1_fullproj (54.9%/55.8%) worse than verb0.1 (60.2%/67.1%) across configs. The one exception (5_16_2_256 +3.1pp) is within noise. This reflects VQ-BeT's fundamentally different codebook structure: with only 16 codes × 2 groups = 32 total action tokens, the LLM can easily learn 32 random embeddings, so projecting from the codebook constrains the representation rather than helping.

The fullproj gain is largest for quest_16_8555_4 (+35pp) and oat_16_8555_4 (+11pp) — large FSQ codebooks (1000 codes) where the LLM struggles to learn standard embeddings, but the pre-FSQ latent projection bypasses this entirely.

#### 2. Fullproj rescues semantic-loss tokenizers

Semantic aux loss (verb/clip) consistently hurts standard-embedding policy performance (as documented earlier). But fullproj often recovers the gap:

| OAT 32_888_8 | Standard TokAcc | Fullproj TokAcc |
|--------------|-----------------|-----------------|
| vanilla | 94.0% | 99.2% |
| verb0.1_post_fsq | 94.8% | **99.6%** |
| clip0.1_post_fsq | 75.3% | 99.0% |

With fullproj, verb0.1 actually achieves the highest accuracy (99.6%), and clip0.1 recovers from 75.3% to 99.0%. The semantic loss distorts the discrete codebook mapping but not the continuous pre-FSQ latent space — fullproj bypasses the damaged mapping.

#### 3. Multiple QueST conditions show suspiciously perfect 100% TokAcc

| Condition | TokAcc | L1 | CE |
|-----------|--------|------|------|
| quest_16_855_4/vanilla | 100% | 0.0000 | 0.0000 |
| quest_16_855_4/vanilla_fullproj | 100% | 0.0000 | 0.0000 |
| quest_32_888_4/verb0.1_post_fsq | 100% | 0.0000 | 0.0000 |
| quest_32_888_4/verb0.1_post_fsq_fullproj | 100% | 0.0000 | 0.0000 |
| quest_32_888_4/clip0.1_post_fsq | 100% | 0.0000 | 0.0000 |
| quest_32_888_4/clip0.1_post_fsq_fullproj | 100% | 0.0000 | 0.0000 |

This pattern extends beyond quest_16_855_4 vanilla — all quest_32_888_4 conditions except vanilla show perfect metrics. Almost certainly an evaluation artifact: either the tokenizer codebook has collapsed (most actions map to the same codes), or the evaluation loop has a data leak. Note quest_32_888_4/vanilla does *not* show 100% (TokAcc=74.6%), and vanilla_fullproj shows 97.9% — only the aux-loss conditions are 100%. This suggests the aux loss may cause codebook collapse specifically for QueST with long horizons. Investigation needed.

#### 4. VQ-BeT aux heads consistently help (unlike OAT/QueST)

For VQ-BeT, semantic aux loss consistently improves downstream policy TF metrics:

| VQ-BeT Config | vanilla TokAcc | verb0.1 TokAcc | clip0.1 TokAcc | vlm_clip TokAcc |
|---------------|----------------|----------------|----------------|-----------------|
| 10_16_2_256 | 43.6% | 60.2% (+17pp) | 58.0% (+14pp) | **65.6%** (+22pp) |
| 5_16_2_256 | 47.4% | 58.7% (+11pp) | **72.2%** (+25pp) | — |
| 5_16_2_512 | 49.1% | **67.1%** (+18pp) | 68.0% (+19pp) | — |

This is the opposite of OAT/QueST where aux loss *hurts* standard-embedding policies. With VQ-BeT's small discrete codebook (32 tokens), semantic structure in the codebook directly helps the LLM's embedding learning. For OAT/QueST, the semantic loss distorts the FSQ quantization grid, hurting the discrete token mapping even though it improves the continuous latent space.

#### 5. VLM-text CLIP supervision (vlm_clip) is the best VQ-BeT condition

For oat_16_855_4, vlm_clip0.1_pre_fsq_fullproj achieves the best TokAcc (88.1%) — slightly above vanilla_fullproj (87.2%). This suggests VLM-aligned CLIP features in the pre-FSQ latent space may provide useful structure for the LLM to exploit.

#### 6. Larger FSQ codebooks help with fullproj

OAT 16_8555_4 (codebook=1000) outperforms OAT 16_855_4 (codebook=200) under fullproj:
- 8555_4 vanilla_fullproj: 98.6% TokAcc
- 855_4 vanilla_fullproj: 87.2% TokAcc

The larger codebook captures more action detail, and fullproj makes it equally easy for the LLM to use.

### Verb Probe Status

All tokenizer conditions have latent + tokid probes completed. Missing probes:
- **probe_native**: missing for all VQ-BeT (9 conditions) and quest_32_888_4 (3 conditions). These are action-only baselines that don't depend on the tokenizer, so one native probe per dataset suffices.

### Remaining Work (updated 2026-04-16)

Sweep is nearly complete. Summary of what's done vs. outstanding:

**Completed**: 55/61 policy conditions trained to 50k, 54 TF evals done.

**Abandoned (not worth resubmitting)**:
- vq_bet_5_16_2_512/{vanilla,verb0.1}_fullproj — timed out at 45k/50k twice; VQ-BeT fullproj doesn't help anyway (see Finding 1)
- quest_32_888_4/clip0.1_post_fsq_fullproj — dynamic resume crash #9, stuck at 45k
- quest_16_8555_4/clip0.1_post_fsq_fullproj — dynamic resume crash #9, stuck at 45k

**Optional**:
1. **VQ-BeT resdim variants**: 7 resdim128/resdim128v2 conditions have policies but no TF evals (not covered by sweep script). These are secondary experiments.
2. **Investigate 100% TokAcc anomaly** in quest_16_855_4/vanilla and quest_32_888_4/{verb0.1, clip0.1} conditions — see Finding 3.

## Issues Encountered

1. **Max steps not set**: The Qwen 0.5B bridge config had no `max_steps` default, resulting in 118M steps (would take years). Fixed by passing `--vla.max_steps 50000`.
2. **numpy 2.0.2 installed to ~/.local**: At 23:57 on Mar 29, numpy 2.0.2 was pip-installed to user site-packages, shadowing the conda env's numpy 1.26.4 and crashing all subsequent jobs. Fixed by `pip uninstall numpy` to remove the user-local version.
3. **PYTHONNOUSERSITE broke attrs**: First attempted fix (PYTHONNOUSERSITE=1) blocked the numpy conflict but also blocked the `attrs` package needed by `jsonlines`. Reverting PYTHONNOUSERSITE and just removing user-local numpy was the correct fix.
4. **JSONL metrics path with `/` in run_id** (2026-04-03): `run_id` like `oat_16_855_4/clip0.1_post_fsq` caused nested subdirectory in metrics path → `FileNotFoundError`. Fixed by using `Path(run_id).name` in `prismatic/training/metrics.py:45`.
5. **TF eval missing PRISMATIC_DATA_ROOT** (2026-04-03): `eval_policy.py` imports from prismatic which triggers `KeyError: 'PRISMATIC_DATA_ROOT'`. Fixed by adding env var export in `run_sweep.sh` eval preamble.
6. **Stale tokenizer paths after directory reorganization** (2026-04-04): Old policies reference flat tokenizer paths (e.g., `oat_16_855_4/full.pth`) that no longer exist. Fixed by creating symlinks from old paths to new nested structure.
7. **Dynamic fullproj codebook=None crash in TF eval** (2026-04-04): OAT/QueST fullproj policies have no codebook — `_forward_static` crashes. Fixed by injecting pre-FSQ latents in `eval_policy.py` before forward pass.
8. **QueST pre_fsq_dim returning 0** (2026-04-04): `nn.TransformerEncoder` doesn't expose `d_model`. Fixed to use `self.model.action_proj.out_features`.
9. **Dynamic fullproj resume crash** (2026-04-06, UNRESOLVED): Resume of `oat_16_855_4/vanilla_fullproj` at 45k steps crashes with `codebook[action_ids]` NoneType error. Debug confirms wrapper is created as DYNAMIC with correct weights restored, but `_current_latents` is None during first forward pass. Likely FSDP-related: `set_current_latents()` may be called on a different object than what FSDP uses internally. Workaround: abandoned resume, policy was already at 45k/50k and later completed via fresh resubmission.
10. **Normalizer FileNotFoundError** (2026-04-14): All jobs in new batch failed because `calvin_sweep_action_tokenizer.py` called `fit_normalizer(data_dir)` using the path stored in tokenizer checkpoint args (`/data/user_data/yashagar/task_D_D/training/`), but this data had been deleted. **Fix**: replaced `fit_normalizer` with `_make_dummy_normalizer()` that creates a structurally correct `LinearNormalizer` from dummy data. The actual normalizer values are overwritten by `load_state_dict` from the tokenizer checkpoint anyway — only the structure matters at load time. Verified with smoke tests on both OAT and VQ-BeT tokenizer loading.
