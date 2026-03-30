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

### OAT (running)

32 jobs submitted (15 OAT + 15 QueST + 2 remaining VQ-BeT). Expected completion: ~10h per job.

### QueST (running)

Same batch as OAT.

## Preliminary Analysis (VQ-BeT only)

**Aux heads dramatically accelerate and improve policy learning.** At step 50k:
- clip0.1 reduces loss by 46% (0.383 vs 0.716) and doubles token accuracy (72% vs 52%) over vanilla for 5_16_2_256
- verb0.1 is intermediate: 22% loss reduction, +13pp token accuracy
- The advantage is visible from step 500 — aux tokenizers give the policy a head start that persists throughout training

**clip0.1 > verb0.1 > none consistently.** This ordering holds across all configs and all steps. CLIP contrastive training produces the most policy-friendly tokenizations.

**L1 loss tells a more nuanced story.** While clip0.1 wins on loss and token accuracy, the L1 differences are smaller:
- 5_16_2_256: none=0.039, verb=0.044, clip=0.039 (tied with none!)
- 5_16_2_512: none=0.046, verb=**0.032**, clip=0.042 (verb wins!)

This suggests that better token prediction (higher accuracy) doesn't always translate to better continuous actions. Verb0.1 with latent_dim=512 achieves the lowest L1 despite lower token accuracy than clip — the tokens it gets wrong may be less consequential for the decoded action.

**chunk_size=10 is significantly worse.** vq_bet_10_16_2_256 has the highest loss (0.889) and L1 (0.073) — longer chunks are harder to predict. Only the vanilla condition completed; verb/clip results pending.

**latent_dim=512 ≈ latent_dim=256.** For vanilla: 512 slightly worse on loss (0.696 vs 0.716) but worse on L1 (0.046 vs 0.039). With aux heads, 512 catches up and verb0.1_512 achieves the best L1 overall.

**All curves still improving at 50k steps.** No sign of convergence — longer training may further separate conditions.

## Issues Encountered

1. **Max steps not set**: The Qwen 0.5B bridge config had no `max_steps` default, resulting in 118M steps (would take years). Fixed by passing `--vla.max_steps 50000`.
2. **numpy 2.0.2 installed to ~/.local**: At 23:57 on Mar 29, numpy 2.0.2 was pip-installed to user site-packages, shadowing the conda env's numpy 1.26.4 and crashing all subsequent jobs. Fixed by `pip uninstall numpy` to remove the user-local version.
3. **PYTHONNOUSERSITE broke attrs**: First attempted fix (PYTHONNOUSERSITE=1) blocked the numpy conflict but also blocked the `attrs` package needed by `jsonlines`. Reverting PYTHONNOUSERSITE and just removing user-local numpy was the correct fix.
