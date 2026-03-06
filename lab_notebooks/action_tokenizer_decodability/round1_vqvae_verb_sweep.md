# Round 1: Verb-Decodable VQ-VAE Lambda Sweep

**Date**: 2026-03-02
**Goal**: Train VQ-VAE tokenizers with joint verb classification loss at
varying lambda values. Measure verb decodability and codebook utilization.

## Setup

### VQ-VAE from scratch (chunk-based)
- Architecture: MLP encoder/decoder (not VQ-VLA)
- chunk_size=4, num_codes=512, latent_dim=64
- Verb classifier: 2-layer transformer (d=128, 4 heads) over quantized tokens
- Loss: `L = recon + vq + lambda * verb_CE` (weighted CE, 21 sparse classes)
- 200 epochs, lr=1e-3, cosine schedule, on CALVIN D-train
- **Note**: verb_acc reported is **train** accuracy (no val split for tokenizer)

### VQ-VLA fine-tuning
- Architecture: pretrained VQ-VLA (113M params, causal conv VAE + 4-group ResidualVQ, 256 codes each)
- Pretrained on Open X-Embodiment + LIBERO + RH20T + ManiSkill + RLBench
- Added VerbHead (MLP 128→128→21) on mean-pooled quantized latents
- Loss: `L = recon + 5*vq + lambda * verb_CE`
- 50 epochs, lr=1e-4, batch_size=32, on CALVIN D-train
- Script: `openvla_experiment/scripts/finetune_tokenizer.py`

---

## VQ-VAE Scratch Training Results (Final)

| Lambda | Job IDs | Status | Recon | VQ | Train Verb Acc | Codebook Usage |
|--------|---------|--------|-------|----|----------------|----------------|
| 0 (vanilla) | 6478401 | Done (ep200) | 0.0265 | 0.0648 | N/A | 15/512 **(2.9%)** |
| 0.01 | 6478403→6478899 | **Timed out ep190** | 0.0168 | 0.0546 | ~50.5%† | — |
| 0.05 | 6478405→6478900 | Done (ep200) | 0.0185 | 0.0743 | 41.7% | 36/512 (7.0%) |
| **0.1** | 6478407→6478901 | Done (ep200) | **0.0149** | 0.0616 | **51.0%** | 73/512 (14.3%) |
| 0.5 | 6478409 | Done (ep200) | 0.0158 | 0.0620 | 45.0% | **98/512 (19.1%)** |
| 1.0 | 6478411 | Done (ep200) | 0.0215 | 0.0578 | 37.1% | 48/512 (9.4%) |

†Lambda=0.01 timed out at ep 190/200 (3h limit, still improving). Trend at ep 190: recon=0.0168, verb_acc=50.5% and still climbing.

### Key findings (scratch training)

1. **Lambda=0.1 is the new best**: 51.0% train verb acc, lowest recon (0.0149), 14.3% codebook utilization.
   Previous "winner" lambda=0.5 has lower verb acc (45%) and worse recon despite higher codebook usage.

2. **Verb loss dramatically improves reconstruction**: vanilla recon=0.0265 → lambda=0.1 recon=0.0149 (44% improvement).
   Auxiliary verb objective acts as a regularizer that prevents codebook collapse.

3. **Codebook collapse is the central failure mode**: vanilla 2.9% → lambda=0.5 19.1%.
   Verb gradients force more diverse code usage, but high lambda (1.0) overshoots and collapses back to 9.4%.

4. **Non-monotone in lambda**: best recon at 0.1, best codebook at 0.5, verb_acc peak around 0.1.
   Lambda=1.0 degrades all three metrics — too much verb pressure crowds out reconstruction signal.

5. **Lambda=0.01 still improving at ep 190**: suggests very long convergence for small lambda.
   A 4h resubmit might be needed to get the final number; trend indicates ~52% final verb acc.

---

## VQ-VLA Fine-Tuning Results (Final)

All 50 epochs completed. Val verb accuracy tracked separately from train.

| Lambda | Job ID | Val Recon | Train Verb Acc | Best Val Verb Acc | Best at Epoch | Codebook Usage |
|--------|--------|-----------|----------------|-------------------|---------------|----------------|
| 0 (vanilla) | 6478918 | 0.00260 | N/A | N/A | ep 48 (recon) | **256/256 (100%)** |
| 0.1 | 6478919 | ~0.0076 | ~81.8% | **43.6%** | ep 42 | ~252/256 (98%) |
| **0.5** | 6478920 | ~0.0140 | ~84.6% | **44.9%** | ep 10 | ~247/256 (97%) |
| 1.0 | 6478921 | ~0.0176 | ~84.6% | 42.5% | ep 15 | ~251/256 (98%) |

### Key findings (VQ-VLA fine-tuning)

1. **Pretrained VQ-VLA has 100% codebook utilization from the start**: no collapse issue.
   The pretrained model is already well-conditioned; verb training cannot improve on this.

2. **Large train-val gap**: train verb acc 81-85%, val verb acc 42-45%.
   VerbHead overfits quickly — best val checkpoint at ep 10-42 (much earlier than end).

3. **Lambda=0.5 achieves best val verb acc (44.9%)** but at cost of 5× higher recon loss vs vanilla.
   Lambda=0.1 is second best (43.6%) with 2× lower recon loss — better trade-off.

4. **Val verb acc plateaus early**: lambda=0.5 plateaus by ep 10, lambda=1.0 by ep 15.
   The pretrained quantized representation has limited but quickly-exhausted verb information.

5. **VQ-VLA val verb acc (44.9%) is slightly below VQ-VAE train verb acc (51.0% for lambda=0.1)**.
   These are not directly comparable (val vs train), but both tokenizers achieve similar verb decodability.

---

## Verb Decodability Probe (2026-03-05)

These probes only depend on VQ-VLA tokenizer checkpoints, not OpenVLA fine-tuning.

### Original Probe (round-trip, superseded)

Ground-truth val trajectories encoded → tokenizer → decoded → pre-trained verb classifier
(trained on native continuous actions). 653 val samples, 21 sparse classes.

| Condition | Verb Acc | Macro F1 |
|-----------|----------|----------|
| Bin (256 bins) | **40.4%** | **37.0%** |
| VQ vanilla (λ=0) | 37.8% | 35.8% |
| VQ verb-decodable (λ=0.5) | 40.3% | 34.6% |

**Why this is flawed**: the classifier was trained on native continuous actions, biased toward
lossless tokenizers. The round-trip feeds reconstructed actions to a mismatched classifier distribution.

### Redesigned Probe: four levels along the pipeline

Each level tests verb decodability at a different point, from the tokenizer output to the final VLA behavior.

**Level 1 — z_q latent probe** *(this round)*: classify on sequences of quantized latent vectors (128-d per 5-step chunk, continuous). Measures whether verb CE moved the VQ latent space into verb-separable positions.

**Level 2 — token ID probe** *(this round)*: classify on sequences of discrete code indices (integer in [0,255] per group). Measures whether discrete token assignments are verb-separable — the exact quantity the LLM sees. Key insight: verb CE operates on z_q (differentiable via straight-through estimator), but the argmin z → token ID is not differentiable. Verb-separable z_q does not guarantee verb-separable token IDs. The gap `Level 1 − Level 2` directly measures quantization information loss.

**Level 3 — LLM action token embedding probe** *(round 2, after VLA fine-tuning)*: extract the fine-tuned LLM's embedding vectors for each action token ID seen in CALVIN, train a classifier on those embeddings. Measures whether the LLM learned verb-clustered representations for action tokens during fine-tuning. If verb-structured tokenization works end-to-end, action token embeddings should cluster by verb class.

**Level 4 — VLA behavior** *(round 2, after VLA fine-tuning)*: (a) continuous L1 loss under teacher-forcing on val split; (b) rollout task success rate (SR1–SR5, avg length) on CALVIN. The gold standard — does verb-structured tokenization actually improve policy performance?

Script (Levels 1 & 2): `openvla_experiment/scripts/train_verb_probe_vq.py`

For Levels 1 & 2, `latent_dim` and probe architecture differ by condition:
- **VQ conditions**: L1 probe inputs are z_q vectors (128-d per chunk); L2 inputs are code IDs [0,255], 4 groups, embedding lookup.
- **Bin condition**: L1 probe inputs are raw continuous actions (7-d per timestep); L2 inputs are per-dim 256-bin IDs, embedding lookup.

| Condition | Level 1 acc ↑ | Level 2 acc ↑ | Quant. loss (L1−L2) ↓ |
|-----------|--------------|---------------|----------------------|
| bin (job 6502775) | 42.84% | 37.81% | **5.03pp** |
| vq_vanilla (job 6502503) | 37.22% | 41.51% | −4.29pp |
| vq_verb λ=0.1 (job 6502611) | 43.72% | 42.54% | 1.18pp |
| vq_verb λ=0.5 (job 6502504) | **45.79%** | **43.72%** | 2.07pp |
| **Δ vq_verb λ=0.5 − vq_vanilla** | **+8.57pp** | **+2.21pp** | |

**Key findings:**

1. **Verb CE strongly improves z_q verb decodability** (+8.6pp at Level 1 vs vq_vanilla). The verb classification loss successfully clusters quantized latents by verb class.

2. **Quantization erodes the gain** (8.6pp at L1 → 2.2pp at L2). The argmin step (z → token ID) loses most of the verb structure that verb CE built into z_q. The quantization loss for vq_verb (L1−L2 = 2.07pp) shows verb information only partially survives into token IDs.

3. **Vanilla: L1−L2 is negative (−4.29pp)**. The token ID probe achieves *higher* accuracy than the z_q latent probe for vanilla. The token ID probe uses 4 separate embedding tables (one per RVQ group), which can learn "group k, code c → verb v" associations more directly than regressing on continuous 128-d z_q vectors. Vanilla's codebook has consistent group-code→verb associations even without explicit verb training.

4. **Bin has the largest quantization loss (5.03pp)**. Despite being a "lossless" quantizer (only rounding error), bin token IDs lose more verb info than VQ codes at L2. The L2 bin probe severely overfits (99% train, ~36% val) — the 256-bin ID space is too high-dimensional and sparse for the embedding probe to generalize. VQ codes benefit from structured learned codebooks that cluster semantically similar actions.

5. **Net effect at token level is small** (2.2pp for vq_verb). The LLM only sees token IDs, so this is the effective ceiling for language grounding via tokenizer choice alone. Bin token IDs actually carry *less* verb info than vq_verb at L2 (37.81% vs 43.72%), despite bin L1 being competitive (42.84% vs 45.79%).

Levels 3 & 4 results → see `round2_openvla_finetune.md`.

---

## Next Steps

- [x] Collect final results from resubmitted lambda=0.01/0.05/0.1 jobs
- [x] Register calvin_dataset in OpenVLA-mini and write fine-tuning scripts (Stage 2)
- [x] Fill in verb probe Level 1 & 2 results — all four conditions DONE (jobs 6502503, 6502504, 6502611, 6502775)
- [x] Bin verb probe (job 6502775): L1=42.84%, L2=37.81%, quant loss=5.03pp
- [ ] Interpret quantization gap: bin has largest loss (5.03pp); vq_verb lowest (2.07pp)
- [ ] Level 3 (LLM action token embedding probe) → round 2, after VLA fine-tuning
- [ ] Level 4 (L1 loss + rollout task success) → round 2, after VLA fine-tuning
- See `round2_openvla_finetune.md` for OpenVLA fine-tuning and Levels 3 & 4 results
