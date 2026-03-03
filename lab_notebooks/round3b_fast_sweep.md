# Round 3b: Full FAST Scale × Vocab Sweep

**Date**: 2025-02-26
**Motivation**: Round 1 swept vocab size at fixed scale=10. Round 2 swept scale with matched vocab. This round completes a full 5×3 grid (scales 1/5/10/20/50 × vocabs 256/512/1024) to understand the joint effect of scale and vocab on classification performance.

---

## Background: FAST Tokenization

The FAST pipeline has two lossy/lossless stages:
1. **DCT + quantization** (controlled by `scale`): Lossy. Higher scale → finer quantization → larger alphabet but longer sequences. Lower scale → coarser quantization → smaller alphabet, shorter sequences.
2. **BPE** (controlled by `vocab_size`): Lossless compression. Merges frequent bigrams. Larger vocab → more merge rules → shorter sequences but larger embedding table.

**Key constraint**: `vocab_size` must be ≥ alphabet size (number of unique DCT quantized values). Three grid cells are infeasible:
- (scale=20, vocab=256): alphabet≈306 > 256 — **impossible**
- (scale=50, vocab=256): alphabet≈766 > 256 — **impossible**
- (scale=50, vocab=512): alphabet≈766 > 512 — **impossible**

---

## Full Results Grid

MSE is essentially identical across all configurations (~0.331–0.333) — DCT quantization is nearly lossless for 7D actions at all tested scales. The meaningful variable is **tokens per trajectory** (sequence length fed to the transformer). At scale=1 (very coarse quantization), BPE has little left to compress — vocab=256/512/1024 all produce ~24–27 tokens/traj. At scale=50 (fine quantization), sequences are already very long (255–303 tokens) even with the largest vocabs.

| Scale | Vocab      | MSE    | Tok/traj | Compress | Acc    | MacF1  | Active | BestAcc | BestMF1 | Round  |
|-------|------------|--------|----------|----------|--------|--------|--------|---------|---------|--------|
| **1** | **256**    | 0.3331 | **26.7** | 16.0x    | **28.7%** | **13.7%** | **14/27** | **30.9%** | **9.9%** | R2 |
| 1     | 512        | 0.3331 | 25.0     | 17.1x    | 24.9%  | 10.4%  | 14/27  | 28.7%   | 8.5%    | R3b    |
| 1     | 1024       | 0.3331 | 23.9     | 17.9x    | 23.4%  | 10.3%  | 13/27  | 28.7%   | 8.3%    | R3b    |
| 5     | 256        | 0.3327 | 94.8     |  4.5x    | 24.3%  | 8.0%   | 8/27   | 25.5%   | 8.6%    | R2     |
| 5     | 512        | 0.3327 | 79.4     |  5.4x    | 17.0%  | 7.3%   | 12/27  | 24.0%   | 6.4%    | R3b    |
| 5     | 1024       | 0.3327 | 70.4     |  6.1x    | 18.6%  | 7.9%   | 12/27  | 23.4%   | 6.5%    | R3b    |
| 10    | 256        | —      | —        | —        | 18.9%  | 4.6%   | 5/27   | —       | —       | R1     |
| 10    | 512        | 0.3318 | 137.7    |  3.1x    | 23.3%  | 7.5%   | 9/27   | 24.9%   | 7.9%    | R1/R2  |
| 10    | 1024       | —      | —        | —        | 17.9%  | 7.6%   | 12/27  | —       | —       | R1     |
| 10    | pretrained | 0.3318 | 127.7    |  3.3x    | 21.2%  | 8.5%   | 11/27  | 23.4%   | 5.2%    | R2     |
| 20    | 512        | 0.3315 | 263.5    |  1.6x    | 20.2%  | 6.5%   | 9/27   | 21.2%   | 7.1%    | R3b    |
| 20    | 768        | 0.3315 | 240.1    |  1.8x    | 21.1%  | 5.6%   | 6/27   | 22.5%   | 5.9%    | R2     |
| 20    | 1024       | 0.3315 | 230.1    |  1.9x    | 22.3%  | 6.3%   | 8/27   | 23.4%   | 6.4%    | R3b    |
| 50    | 1024       | 0.3312 | 302.7    |  1.4x    | 25.3%  | 6.9%   | 7/27   | 26.9%   | 6.8%    | R3b    |
| 50    | 1536       | 0.3312 | 254.9    |  1.7x    | 26.8%  | 7.7%   | 8/27   | 28.0%   | 7.0%    | R2     |
| **Native** | **—** | —      | **~64**  | —        | **40.0%** | **32.0%** | **20/27** | **43.0%** | **32.5%** | R2 |

---

## Analysis

![FAST Sweep Combined](../figures/ao_fast_sweep_combined.png)

### 1. Scale=1 wins across the board

Scale=1 with any vocab size (256/512/1024) consistently outperforms all other configurations. The pattern:

| Scale | Best BestAcc across vocabs | Tokens |
|-------|---------------------------|--------|
| 1     | 30.9%                     | ~24–27  |
| 5     | 25.5%                     | ~70–95  |
| 50    | 28.0%                     | ~255–303 |
| 10    | 24.9%                     | ~128–138 |
| 20    | 23.4%                     | ~230–264 |

Scale=1 produces 24–27 tokens/trajectory — the shortest sequences by far (16–18× compression). With only 3,400 training samples, shorter sequences are dramatically easier to learn from. The model can focus on the few tokens that matter rather than attending over 240–300 mostly-redundant tokens.

### 2. Within scale=1: larger vocab doesn't help

| Scale | Vocab | Tok/traj | BestAcc | BestMF1 |
|-------|-------|----------|---------|---------|
| 1     | 256   | 26.7     | **30.9%** | **9.9%** |
| 1     | 512   | 25.0     | 28.7%   | 8.5%    |
| 1     | 1024  | 23.9     | 28.7%   | 8.3%    |

Larger vocab gives slightly shorter sequences (more BPE merges) but worse accuracy. The embedding table for 512 or 1024 tokens is harder to learn from 3,400 samples. v256 wins because the smaller embedding space is easier to generalize.

### 3. Within other scales: vocab has little consistent effect

For scale=5, 10, 20: accuracy varies by 1–4pp across vocabs with no consistent trend. Sequence length varies significantly but the differences are second-order compared to the scale effect.

### 4. Scale=50 is second-best despite longest sequences

Surprisingly, scale=50 (302 tokens, 1.4× compression) beats scale=20 (230 tokens, 1.9×). At scale=50, the fine quantization produces a large alphabet with rich token variety — each token is more semantically distinctive. But scale=1's advantage (very short sequences) still wins overall.

### 5. FAST native vs pretrained: minimal difference

Custom-fitted FAST+ (scale=10, v512, best=24.9%) ≈ pretrained FAST+ (best=23.4%). Despite the pretrained model being trained on 1M real-robot trajectories, our custom model fitted on 3,400 CALVIN trajectories performs comparably. 

### 6. All FAST configurations fall far short of native actions

Best FAST: 30.9% (s1/v256) vs native: 43.0%. The DCT tokenization pipeline, even at its best, loses information that native continuous actions preserve. 

---

## Complete Grid Heatmap (Best Val Accuracy %)

```
Vocab →    256     512    1024   [1536]  [2048]
Scale ↓
  1        30.9   28.7    28.7     —       —
  5        25.5   24.0    23.4     —       —
 10        —      24.9     —      —      23.4*
 20        N/A    21.2    23.4   22.5     —
 50        N/A    N/A     26.9  28.0*    —

N/A = infeasible (alphabet > vocab)
* = non-standard vocab from original sweep
Native = 43.0%
```

---

## Key Findings

1. **Scale=1, vocab=256 is the best FAST configuration** (30.9% best accuracy, 14 active classes, 24–27 tokens/trajectory). The coarsest quantization wins because it produces the shortest sequences, which are easiest to learn from 3,400 samples.

2. **Larger vocab within the same scale doesn't help** — the embedding table is harder to generalize, and BPE's marginal compression gain doesn't compensate.

3. **All FAST variants fall 12+ pp below native actions** (30.9% vs 43.0%). FAST tokenization is fundamentally lossy for classification: the DCT discards continuous variation needed to discriminate between verbs.

4. **Reconstruction MSE is nearly identical across all configurations** (~0.331–0.333). MSE is a poor predictor of classification performance — sequence length is the better proxy.

5. **For future work**: If FAST-style tokenization is desired (e.g., for joint action generation + classification), scale=1 with vocab=256 is the recommended starting point.

---

## Experiment Index

| Job ID  | Experiment         | Scale | Vocab | Status    |
|---------|--------------------|-------|-------|-----------|
| 6445600 | action_only_fast_v256  | 10  | 256   | completed (R1) |
| 6445602 | action_only_fast_v512  | 10  | 512   | completed (R1) |
| 6445604 | action_only_fast_v1024 | 10  | 1024  | completed (R1) |
| 6445606 | action_only_fast_v2048 | 10  | 2048  | completed (R1) |
| 6452036 | ao_fast_pretrained     | 10  | ptd   | completed (R2) |
| 6457871 | ao_fast_s1_v256        | 1   | 256   | completed (R2) |
| 6457872 | ao_fast_s5_v256        | 5   | 256   | completed (R2) |
| 6457873 | ao_fast_s10_v512       | 10  | 512   | completed (R2) |
| 6457874 | ao_fast_s20_v768       | 20  | 768   | completed (R2) |
| 6457875 | ao_fast_s50_v1536      | 50  | 1536  | completed (R2) |
| 6457876 | ao_fast_pretrained_v2  | 10  | ptd   | completed (R2) |
| 6458053 | ao_fast_s1_v512        | 1   | 512   | completed (R3b) |
| 6458054 | ao_fast_s1_v1024       | 1   | 1024  | completed (R3b) |
| 6458055 | ao_fast_s5_v512        | 5   | 512   | completed (R3b) |
| 6458056 | ao_fast_s5_v1024       | 5   | 1024  | completed (R3b) |
| 6458057 | ao_fast_s20_v512       | 20  | 512   | completed (R3b) |
| 6458058 | ao_fast_s20_v1024      | 20  | 1024  | completed (R3b) |
| 6458059 | ao_fast_s50_v1024      | 50  | 1024  | completed (R3b) |
| —       | s20_v256              | 20  | 256   | infeasible (alphabet > vocab) |
| —       | s50_v256              | 50  | 256   | infeasible (alphabet > vocab) |
| —       | s50_v512              | 50  | 512   | infeasible (alphabet > vocab) |
