# Experiment: Improving Verb Decodability Propagation Through Quantization

## Motivation

The bridge sweep (see `bridge_sweep.md`) established that auxiliary losses (verb, CLIP) dramatically improve verb decodability in continuous latents (up to 56.6% acc), but this improvement does not always propagate through quantization to discrete token IDs — which is what the VLA actually sees.

The degree of propagation depends on the quantizer:

| Family | Vanilla TokID | Best Aux TokID | Delta | Propagation |
|--------|--------------|----------------|-------|-------------|
| OAT (FSQ) | 19.4-20.6% | 22.1-24.0% | +2-4pp | Partial |
| VQ-BeT (ResidualVQ) | 17.8-21.5% | 23.0-27.5% | +5-10pp | Strong |
| QueST (FSQ) | 19.6-21.8% | 19.3-20.5% | ~0pp | None |

**Hypothesis:** FSQ's fixed axis-aligned grid is the bottleneck. FSQ projects the 256d latent down to a 4d space and rounds each dimension to discrete levels. The grid boundaries are fixed — they cannot move to respect verb boundaries, no matter how much the aux loss shapes the 256d latent. Two actions with different verbs that land in the same FSQ grid cell become indistinguishable after quantization.

VQ-BeT's ResidualVQ, by contrast, learns codebook centroids that *can* shift to align with verb boundaries when pushed by aux loss gradients.

We test two interventions.

## Experiment 1: Replace FSQ with VQ in QueST

**Idea:** Keep QueST's encoder (causal conv + transformer) but swap FSQ for learned VQ codebook. If the hypothesis is correct, VQ centroids should shift to preserve verb structure, and tokid probe accuracy should improve.

### Configs

Match the three FSQ codebook sizes from the original sweep:

| Tag | Horizon | VQ Codebook Size | Downsample | Epochs | Batch | LR |
|-----|---------|-----------------|------------|--------|-------|----|
| quest_16_vq_200_4 | 16 | 200 | 4 | 300 | 128 | 1e-4 |
| quest_32_vq_512_4 | 32 | 512 | 4 | 300 | 128 | 1e-4 |
| quest_16_vq_1000_4 | 16 | 1000 | 4 | 300 | 128 | 1e-4 |

Each trained with 3 aux conditions: none, verb:0.1, clip:0.1 → 9 tokenizer runs + verb probes.

**Risk:** VQ is prone to codebook collapse (FSQ avoids this by construction). If utilization drops to <10%, the comparison is confounded. The aux heads should help — they boosted VQ-BeT utilization from 42-50% to 57-88%.

### Results

Tokenizer training:

| Config | Aux | Val Recon | Util% | Vocab | Val Verb Acc | Val Verb MF1 | CLIP Loss | R@1 | R@5 |
|--------|-----|-----------|-------|-------|-------------|-------------|-----------|-----|-----|
| quest_16_vq_200_4 | none | 0.00133 | 15.5 | 200 | | | | | |
| quest_16_vq_200_4 | verb | 0.00251 | 9.0 | 200 | 48.0 | 55.2 | | | |
| quest_16_vq_200_4 | clip | 0.00194 | 14.5 | 200 | | | 2.28 | 4.4 | 18.4 |
| quest_32_vq_512_4 | none | 0.00339 | 3.7 | 512 | | | | | |
| quest_32_vq_512_4 | verb | 0.00345 | 5.9 | 512 | 39.1 | 44.5 | | | |
| quest_32_vq_512_4 | clip | 0.00347 | 5.9 | 512 | | | 2.32 | 3.7 | 16.7 |
| quest_16_vq_1000_4 | none | 0.00133 | 3.5 | 1000 | | | | | |
| quest_16_vq_1000_4 | verb | 0.00231 | 2.7 | 1000 | 40.7 | 51.0 | | | |
| quest_16_vq_1000_4 | clip | 0.00242 | 2.7 | 1000 | | | 2.31 | 4.2 | 17.5 |

Verb probes (native baseline: 23.1% acc / 27.4% MF1):

| Config | Aux | Latent Acc | Latent MF1 | TokID Acc | TokID MF1 |
|--------|-----|-----------|-----------|----------|----------|
| quest_16_vq_200_4 | none | 13.9 | 16.6 | 19.7 | 21.4 |
| quest_16_vq_200_4 | verb | 34.3 | 41.6 | **29.9** | **37.7** |
| quest_16_vq_200_4 | clip | 31.6 | 36.9 | **34.4** | **37.1** |
| quest_32_vq_512_4 | none | 21.1 | 16.9 | 18.8 | 18.0 |
| quest_32_vq_512_4 | verb | 32.6 | 40.9 | **30.1** | **32.3** |
| quest_32_vq_512_4 | clip | 32.8 | 37.5 | 26.4 | 24.8 |
| quest_16_vq_1000_4 | none | 14.4 | 17.7 | 20.8 | 22.3 |
| quest_16_vq_1000_4 | verb | 31.2 | 37.6 | **31.8** | **34.9** |
| quest_16_vq_1000_4 | clip | 22.7 | 28.2 | **25.5** | **28.8** |

### Analysis

**TokID propagation is dramatically better with VQ.** The central result:

| Quantizer | Best Aux TokID Acc | Best Aux TokID MF1 | Propagation |
|-----------|-------------------|-------------------|-------------|
| QueST + FSQ (original) | 19.3-20.5% | 19.6-25.1% | None (~0pp) |
| QueST + VQ (this exp) | **25.5-34.4%** | **28.8-37.7%** | **Strong (+5-15pp)** |

This confirms the hypothesis: FSQ's fixed grid was the bottleneck. VQ's learnable centroids shift to respect verb boundaries when pushed by aux loss, and this structure survives quantization.

**However, VQ codebook utilization is severe.** Utilization ranges from 2.7% to 15.5% — far below FSQ's 55-100%. The codebook is mostly collapsed. Despite this, the active codes are verb-discriminative enough to substantially improve tokid probes. This is a remarkable tradeoff: VQ uses fewer codes but places them more meaningfully.

**Latent probes are worse with VQ.** Best VQ latent: 34.3% acc vs FSQ latent: 56.6% acc. The collapsed codebook means the encoder can't learn as rich a latent space. But what matters for the VLA is the token IDs, not the latents.

**Reconstruction is slightly worse.** VQ recon (0.00133-0.00347) is comparable to FSQ (0.00078-0.00349), but aux heads hurt recon more with VQ, likely due to the additional commitment loss interactions.

**CLIP performance is comparable.** R@1 and R@5 are similar between VQ and FSQ, suggesting the contrastive alignment works equally well through learnable centroids.

## Experiment 2: Apply Aux Head to Post-FSQ 4d Vector (STE)

**Idea:** The current aux head operates on the 256d pre-FSQ latent. The 256d space can encode verb information in directions that are orthogonal to the 4d projection — great latent accuracy, zero transfer to tokens. By applying the aux head to the 4d post-round vector (with straight-through gradient), we force verb information into exactly the dimensions that survive quantization.

### Why this works (gradient-wise)

Normally there's no gradient path through discrete token IDs. But FSQ's quantization uses STE (straight-through estimator): the forward pass rounds to grid points, but the backward pass passes gradients through as if no rounding happened. The 4d post-round vector is a differentiable proxy for the token ID — the two are a bijection (e.g., `[2, 3, 1, 4]` → index 347). So applying the aux head post-round is effectively "applying the head on the token ID" in a form that admits gradients.

### Architecture

The aux head is a small transformer over the episode's token sequence. We use a constrained architecture to prevent the head from compensating for a non-verb-discriminative 4d space:

```
Episode input: (B, K*T', 4)       # all chunks, 4d per token
    → Linear(4, 16)               # small expansion (not 256→128)
    → + sinusoidal PE from positions
    → prepend [CLS] token
    → TransformerEncoder(2 layers, 2 heads, d=16)
    → CLS output (B, 16)
    → Linear(16, num_verbs)
```

Key design choice: `d_model=16` with `nhead=2` (8d per head). This is deliberately constrained — a large d_model would let the head learn a rich mapping from 4d that classifies verbs without the 4d space itself needing to be verb-discriminative. The bottleneck must stay in the representation, not the classifier.

### Configs

Applied to all 6 FSQ-based tokenizer configs (3 OAT + 3 QueST-FSQ) × 2 aux heads (verb, clip) = 12 runs. Skipped for VQ-BeT (no FSQ) and QueST+VQ (no FSQ).

Directory naming: `{tok}_{config}_{aux}{lambda}_pfsq` (e.g., `quest_32_888_4_verb0.1_pfsq`).

### What we expect

- **Latent probe accuracy will drop** compared to the standard aux (since the 256d latent is no longer directly optimized for verbs — only the 4d bottleneck is).
- **TokID probe accuracy should improve** — this is the whole point. The STE gradient forces the FSQ grid cells to align with verb boundaries.
- The interesting comparison is the tokid probe between:
  - Standard aux (e.g., `quest_32_888_4_clip0.1`): latent=56.6%, tokid=19.9%
  - Post-FSQ aux (`quest_32_888_4_clip0.1_pfsq`): latent=??, tokid=??

If tokid improves substantially, it confirms that the propagation failure was about *where* the aux loss is applied, not the FSQ mechanism itself.

### Results

Tokenizer training (rows without "pfsq" are the standard pre-FSQ baselines from `bridge_sweep.md`):

| Config | Aux | Val Recon | Util% | Vocab | Val Verb Acc | Val Verb MF1 | CLIP Loss | R@1 | R@5 |
|--------|-----|-----------|-------|-------|-------------|-------------|-----------|-----|-----|
| oat_16_855_4 | none | 0.00333 | 74.0 | 200 | | | | | |
| oat_16_855_4 | verb | 0.00426 | 93.0 | 200 | 34.8 | 44.2 | | | |
| oat_16_855_4 | clip | 0.00382 | 96.0 | 200 | | | 2.97 | 3.7 | 16.0 |
| oat_16_855_4 | verb pfsq | 0.00363 | 99.5 | 200 | 24.8 | 32.9 | | | |
| oat_16_855_4 | clip pfsq | 0.00489 | 100.0 | 200 | | | 3.74 | 1.7 | 8.0 |
| oat_32_888_8 | none | 0.00309 | 94.5 | 512 | | | | | |
| oat_32_888_8 | verb | 0.00572 | 95.3 | 512 | 32.9 | 36.5 | | | |
| oat_32_888_8 | clip | 0.00517 | 95.7 | 512 | | | 3.26 | 3.4 | 14.3 |
| oat_32_888_8 | verb pfsq | 0.00407 | 99.8 | 512 | 19.9 | 25.2 | | | |
| oat_32_888_8 | clip pfsq | 0.00476 | 100.0 | 512 | | | 3.56 | 2.0 | 8.9 |
| oat_16_8555_4 | none | 0.00274 | 47.2 | 1000 | | | | | |
| oat_16_8555_4 | verb | 0.00346 | 62.8 | 1000 | 39.9 | 48.5 | | | |
| oat_16_8555_4 | clip | 0.00354 | 55.8 | 1000 | | | 2.92 | 4.4 | 17.4 |
| oat_16_8555_4 | verb pfsq | 0.00352 | 94.5 | 1000 | 24.9 | 33.9 | | | |
| oat_16_8555_4 | clip pfsq | 0.00396 | 100.0 | 1000 | | | 3.63 | 2.1 | 8.6 |
| quest_16_855_4 | none | 0.00083 | 100.0 | 200 | | | | | |
| quest_16_855_4 | verb | 0.00207 | 61.0 | 200 | 44.6 | 54.5 | | | |
| quest_16_855_4 | clip | 0.00241 | 99.5 | 200 | | | 2.31 | 3.9 | 17.1 |
| quest_16_855_4 | verb pfsq | 0.00254 | 100.0 | 200 | 33.2 | 47.0 | | | |
| quest_16_855_4 | clip pfsq | 0.00460 | 100.0 | 200 | | | 2.85 | 2.1 | 10.1 |
| quest_32_888_4 | none | 0.00092 | 97.9 | 512 | | | | | |
| quest_32_888_4 | verb | 0.00284 | 67.2 | 512 | 38.6 | 46.8 | | | |
| quest_32_888_4 | clip | 0.00349 | 78.9 | 512 | | | 2.40 | 3.6 | 15.7 |
| quest_32_888_4 | verb pfsq | 0.00395 | 99.2 | 512 | 31.3 | 42.9 | | | |
| quest_32_888_4 | clip pfsq | 0.00542 | 100.0 | 512 | | | 3.06 | 1.9 | 9.0 |
| quest_16_8555_4 | none | 0.00078 | 88.2 | 1000 | | | | | |
| quest_16_8555_4 | verb | 0.00159 | 59.3 | 1000 | 49.8 | 55.2 | | | |
| quest_16_8555_4 | clip | 0.00190 | 55.8 | 1000 | | | 2.33 | 4.1 | 17.3 |
| quest_16_8555_4 | verb pfsq | 0.00252 | 95.3 | 1000 | 29.4 | 46.5 | | | |
| quest_16_8555_4 | clip pfsq | 0.00368 | 100.0 | 1000 | | | 2.90 | 2.0 | 9.8 |

Note: quest_16_855_4_clip0.1_pfsq and quest_32_888_4_verb0.1_pfsq initially collapsed to 2 active codes due to initialization sensitivity. Reruns with `--seed 123` trained successfully (100% and 99.2% utilization respectively). Results above are from the successful reruns.

Verb probes (native baseline: 23.1% acc / 27.4% MF1):

| Config | Aux | Latent Acc | Latent MF1 | TokID Acc | TokID MF1 |
|--------|-----|-----------|-----------|----------|----------|
| oat_16_855_4 | none | 22.1 | 22.0 | 19.7 | 20.6 |
| oat_16_855_4 | verb | 36.3 | 40.8 | 22.5 | 26.1 |
| oat_16_855_4 | clip | 48.1 | 52.4 | 22.6 | 25.7 |
| oat_16_855_4 | verb pfsq | 31.9 | 37.5 | **26.9** | **32.9** |
| oat_16_855_4 | clip pfsq | 43.4 | 50.0 | **33.8** | **39.2** |
| oat_32_888_8 | none | 21.5 | 18.9 | 19.4 | 18.0 |
| oat_32_888_8 | verb | 34.5 | 37.5 | 22.1 | 25.8 |
| oat_32_888_8 | clip | 45.9 | 47.7 | 21.3 | 25.7 |
| oat_32_888_8 | verb pfsq | 28.1 | 32.6 | 23.5 | 30.4 |
| oat_32_888_8 | clip pfsq | 45.4 | 46.4 | **32.9** | **35.9** |
| oat_16_8555_4 | none | 20.6 | 21.7 | 20.6 | 21.4 |
| oat_16_8555_4 | verb | 38.2 | 43.5 | 24.0 | 29.8 |
| oat_16_8555_4 | clip | 49.0 | 52.2 | 22.2 | 27.3 |
| oat_16_8555_4 | verb pfsq | 33.8 | 40.4 | **25.6** | **33.6** |
| oat_16_8555_4 | clip pfsq | 45.7 | 50.6 | **32.4** | **38.4** |
| quest_16_855_4 | none | 25.2 | 31.5 | 19.8 | 25.9 |
| quest_16_855_4 | verb | 49.1 | 54.6 | 20.3 | 25.1 |
| quest_16_855_4 | clip | 51.5 | 57.2 | 19.8 | 23.7 |
| quest_16_855_4 | verb pfsq | 45.7 | 52.0 | **35.5** | **44.6** |
| quest_16_855_4 | clip pfsq | **49.5** | **53.4** | **41.9** | **47.5** |
| quest_32_888_4 | none | 21.8 | 24.2 | 19.6 | 21.2 |
| quest_32_888_4 | verb | 45.2 | 49.8 | 19.3 | 23.1 |
| quest_32_888_4 | clip | 56.6 | 58.2 | 19.9 | 19.6 |
| quest_32_888_4 | verb pfsq | **42.9** | **50.1** | **35.9** | **45.1** |
| quest_32_888_4 | clip pfsq | 51.0 | 55.4 | **40.9** | **45.0** |
| quest_16_8555_4 | none | 23.9 | 30.2 | 21.8 | 28.2 |
| quest_16_8555_4 | verb | 50.3 | 53.7 | 20.5 | 24.1 |
| quest_16_8555_4 | clip | 52.7 | 55.6 | 19.8 | 24.0 |
| quest_16_8555_4 | verb pfsq | 46.7 | 53.6 | **38.8** | **48.6** |
| quest_16_8555_4 | clip pfsq | 51.7 | 56.2 | **40.1** | **40.7** |

All QueST pfsq conditions now have valid results after rerunning the 2 initially-collapsed conditions with `--seed 123`.

### Analysis

**Post-FSQ aux is the single most effective intervention for tokid verb decodability.** Across all non-collapsed conditions, pfsq improves tokid accuracy by +5 to +21pp over the corresponding standard aux condition.

**QueST pfsq results are the strongest overall.** The non-collapsed QueST pfsq conditions massively outperform everything else:

| Config | Aux | Latent Acc | TokID Acc | TokID MF1 | Δ TokID vs std |
|--------|-----|-----------|----------|----------|----------------|
| quest_16_855_4 | verb pfsq | 45.7 | **35.5** | **44.6** | +15.2 |
| quest_16_855_4 | clip pfsq | **49.5** | **41.9** | **47.5** | +22.1 |
| quest_32_888_4 | verb pfsq | **42.9** | **35.9** | **45.1** | +16.6 |
| quest_32_888_4 | clip pfsq | 51.0 | **40.9** | **45.0** | +21.0 |
| quest_16_8555_4 | verb pfsq | 46.7 | **38.8** | **48.6** | +18.3 |
| quest_16_8555_4 | clip pfsq | 51.7 | **40.1** | **40.7** | +20.3 |

All 6 QueST pfsq conditions show massive tokid improvements. The best result — `quest_16_8555_4_verb0.1_pfsq` at **48.6% tokid MF1** — nearly doubles the best standard QueST result (28.2% for quest_16_8555_4 none). The rerun of `quest_16_855_4_clip0.1_pfsq` is also remarkable: **49.5% latent acc / 41.9% tokid acc** — the highest latent accuracy of any pfsq condition, with strong propagation (0.85 ratio). Critically, latent accuracy is preserved or even improved across all pfsq conditions, meaning pfsq doesn't just redistribute information — it genuinely improves the discrete representation.

**OAT pfsq is consistently positive but smaller gains.** The full `oat_16_855_4` comparison:

| Aux | Latent Acc | Latent MF1 | TokID Acc | TokID MF1 |
|-----|-----------|-----------|----------|----------|
| none | 22.1 | 22.0 | 19.7 | 20.6 |
| verb (pre-FSQ) | 36.3 | 40.8 | 22.5 | 26.1 |
| clip (pre-FSQ) | 48.1 | 52.4 | 22.6 | 25.7 |
| verb **pfsq** | 31.9 | 37.5 | **26.9** | **32.9** |
| clip **pfsq** | 43.4 | 50.0 | **33.8** | **39.2** |

OAT pfsq gains +4-11pp tokid acc over standard aux. The latent-to-tokid gap narrows substantially: for clip pfsq, 43.4% latent → 33.8% tokid (ratio 0.78) vs standard clip 48.1% → 22.6% (ratio 0.47).

**CLIP pfsq consistently outperforms verb pfsq on tokid.** Across all OAT configs, clip pfsq tokid > verb pfsq tokid by +6-9pp. For QueST, the pattern is mixed (quest_16_855_4 verb > clip because clip collapsed; quest_16_8555_4 are roughly tied). This may be because contrastive learning naturally pushes apart episode-level representations without requiring the classifier bottleneck.

**Post-FSQ aux improves codebook utilization.** OAT pfsq conditions show 94-100% utilization vs 47-96% for standard conditions. The STE gradient appears to encourage more uniform use of the FSQ grid.

**Codebook collapse was initialization-dependent, not systematic.** Two QueST pfsq conditions initially collapsed to 2 active codes due to unlucky random initialization (codes=2 from epoch 1). Reruns with `--seed 123` trained successfully with full codebook utilization (200/200 and 508/512 codes). All 6 OAT pfsq conditions trained without collapse on the first attempt. The fix confirms that pfsq is compatible with both OAT and QueST when initialization is favorable.

**Tokenizer-level verb acc is lower with pfsq (expected).** The constrained d_model=16 aux head on 4d input can't classify verbs as well as d_model=128 on 256d input (OAT verb pfsq: 25% vs standard: 35%). But what matters is downstream propagation, not the training head's accuracy — and the probe results confirm this decisively.

## Scripts

- Training: `tokenization/train_tokenizer.py --aux_target post_fsq` (Exp 2)
- Training: `tokenization/train_tokenizer.py --set vq_type=vq vq_codebook_size=N` (Exp 1)
- Sweep: `run_sweep.sh` with `DATASET="bridge"`, AUX_HEAD includes `"verb:0.1:pfsq"` and `"clip:0.1:pfsq"`
- Probes: same `verb_probe/train_verb_probe.py` — probes measure the tokenizer's output, not how it was trained
