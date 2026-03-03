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

## CALVIN → RLDS Conversion (for OpenVLA-mini)

| Job ID | Status | Output |
|--------|--------|--------|
| 6478945 | **Done** | `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/` |

- Train: 5124 episodes, 308918 steps, 62 shards
- Val: 1011 episodes, 60575 steps, 13 shards
- Format: TFRecord (RLDS), fields: `observation/image`, `observation/wrist_image`, `observation/state`, `action`, `language_instruction`, `discount`, `reward`, `is_first`, `is_last`, `is_terminal`
- Next: register dataset in OpenVLA-mini configs and write fine-tuning script

---

## Next Steps

- [x] Collect final results from resubmitted lambda=0.01/0.05/0.1 jobs
- [ ] (Optional) Resubmit lambda=0.01 with 4h to get final ep200 number
- [ ] Plot lambda vs (recon, verb_acc, codebook_utilization) sweep curve for both VQ-VAE and VQ-VLA
- [ ] Register calvin_dataset in OpenVLA-mini's `configs.py`, `transforms.py`, `mixtures.py`
- [ ] Write OpenVLA-mini fine-tuning script (Stage 2): fine-tune with verb-decodable vs vanilla tokenizer
- [ ] Write teacher-forcing evaluation script (Stage 3): compare action prediction quality
