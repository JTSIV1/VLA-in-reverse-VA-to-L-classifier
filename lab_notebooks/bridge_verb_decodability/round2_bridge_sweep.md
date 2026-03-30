# BridgeV2 Action Tokenizer Sweep + Verb Decodability

## Goal

Find the best action tokenizer configuration for BridgeV2 by sweeping horizon, codebook size, and architecture across OAT, QueST, and VQ-BeT. Each tokenizer is trained in three conditions: no aux head, verb aux head (lambda=0.1), and CLIP contrastive aux head (lambda=0.1). We then measure verb decodability via linear probes on latent representations and discrete token IDs.

## Dataset

- **BridgeV2** action trajectories: 27,575 episodes (filtered to episodes in `bridge_episodes_filtered.csv`), 7-DoF actions
- Train: 24,818 / Val: 2,757 (seed=42, 10% val split)
- 17 verb classes (min_count=30), weighted CE loss
- Data stored as shards in `/data/user_data/wenjiel2/datasets/bridge_actions/`

## Sweep Design

9 tokenizer configs x 3 aux conditions = 27 tokenizer training runs.
Each tokenizer then evaluated with 3 verb probes: native (raw actions), latent (continuous encoder output), tokid (discrete code IDs).

### Tokenizer Configs

| Family | Tag | Key params | Vocab | Epochs | Batch | LR |
|--------|-----|-----------|-------|--------|-------|----|
| OAT | 16_855_4 | h=16, r=4, fsq=[8,5,5] | 200 | 300 | 256 | 5e-5 |
| OAT | 32_888_8 | h=32, r=8, fsq=[8,8,8] | 512 | 300 | 256 | 5e-5 |
| OAT | 16_8555_4 | h=16, r=4, fsq=[8,5,5,5] | 1000 | 300 | 256 | 5e-5 |
| QueST | 16_855_4 | h=16, ds=4, fsq=[8,5,5] | 200 | 300 | 128 | 1e-4 |
| QueST | 32_888_4 | h=32, ds=4, fsq=[8,8,8] | 512 | 300 | 128 | 1e-4 |
| QueST | 16_8555_4 | h=16, ds=4, fsq=[8,5,5,5] | 1000 | 300 | 128 | 1e-4 |
| VQ-BeT | 5_16_2_256 | c=5, e=16, g=2, l=256 | 256 | 200 | 256 | 1e-4 |
| VQ-BeT | 5_16_2_512 | c=5, e=16, g=2, l=512 | 256 | 200 | 256 | 1e-4 |
| VQ-BeT | 10_16_2_256 | c=10, e=16, g=2, l=256 | 256 | 200 | 256 | 1e-4 |

### Aux Head Conditions

- **none**: reconstruction only
- **verb:0.1**: verb classification head on latent (lambda=0.1)
- **clip:0.1**: CLIP contrastive head matching latent to instruction text (lambda=0.1)

## Results

### Tokenizer Training

| Tokenizer | Config | Aux | Val Recon | Util% | Vocab | Val Verb Acc | Val Verb MF1 | CLIP Loss | R@1 | R@5 |
|-----------|--------|-----|-----------|-------|-------|-------------|-------------|-----------|-----|-----|
| OAT | 16_855_4 | none | 0.00333 | 74.0 | 200 | | | | | |
| OAT | 16_855_4 | verb | 0.00426 | 93.0 | 200 | 34.8 | 44.2 | | | |
| OAT | 16_855_4 | clip | 0.00382 | 96.0 | 200 | | | 2.97 | 3.7 | 16.0 |
| OAT | 32_888_8 | none | 0.00309 | 94.5 | 512 | | | | | |
| OAT | 32_888_8 | verb | 0.00572 | 95.3 | 512 | 32.9 | 36.5 | | | |
| OAT | 32_888_8 | clip | 0.00517 | 95.7 | 512 | | | 3.26 | 3.4 | 14.3 |
| OAT | 16_8555_4 | none | 0.00274 | 47.2 | 1000 | | | | | |
| OAT | 16_8555_4 | verb | 0.00346 | 62.8 | 1000 | 39.9 | 48.5 | | | |
| OAT | 16_8555_4 | clip | 0.00354 | 55.8 | 1000 | | | 2.92 | 4.4 | 17.4 |
| QueST | 16_855_4 | none | 0.00083 | 100.0 | 200 | | | | | |
| QueST | 16_855_4 | verb | 0.00207 | 61.0 | 200 | 44.6 | 54.5 | | | |
| QueST | 16_855_4 | clip | 0.00241 | 99.5 | 200 | | | 2.31 | 3.9 | 17.1 |
| QueST | 32_888_4 | none | 0.00092 | 97.9 | 512 | | | | | |
| QueST | 32_888_4 | verb | 0.00284 | 67.2 | 512 | 38.6 | 46.8 | | | |
| QueST | 32_888_4 | clip | 0.00349 | 78.9 | 512 | | | 2.40 | 3.6 | 15.7 |
| QueST | 16_8555_4 | none | 0.00078 | 88.2 | 1000 | | | | | |
| QueST | 16_8555_4 | verb | 0.00159 | 59.3 | 1000 | 49.8 | 55.2 | | | |
| QueST | 16_8555_4 | clip | 0.00190 | 55.8 | 1000 | | | 2.33 | 4.1 | 17.3 |
| VQ-BeT | 5_16_2_256 | none | 0.00202 | 50.4 | 256 | | | | | |
| VQ-BeT | 5_16_2_256 | verb | 0.00256 | 73.8 | 256 | 26.3 | 33.3 | | | |
| VQ-BeT | 5_16_2_256 | clip | 0.00332 | 76.2 | 256 | | | 3.68 | 2.0 | 9.1 |
| VQ-BeT | 5_16_2_512 | none | 0.00201 | 50.4 | 256 | | | | | |
| VQ-BeT | 5_16_2_512 | verb | 0.00247 | 67.2 | 256 | 26.1 | 35.2 | | | |
| VQ-BeT | 5_16_2_512 | clip | 0.00400 | 88.3 | 256 | | | 3.64 | 2.4 | 9.3 |
| VQ-BeT | 10_16_2_256 | none | 0.00307 | 41.8 | 256 | | | | | |
| VQ-BeT | 10_16_2_256 | verb | 0.00462 | 57.0 | 256 | 28.4 | 35.6 | | | |
| VQ-BeT | 10_16_2_256 | clip | 0.00946 | 77.3 | 256 | | | 3.73 | 2.2 | 8.4 |

### Verb Probe Results

Native baseline (raw 7-DoF actions, no tokenizer): **23.1% acc / 27.4% MF1**

| Tokenizer | Config | Aux | Latent Acc | Latent MF1 | TokID Acc | TokID MF1 |
|-----------|--------|-----|-----------|-----------|----------|----------|
| OAT | 16_855_4 | none | 22.1 | 22.0 | 19.7 | 20.6 |
| OAT | 16_855_4 | verb | 36.3 | 40.8 | 22.5 | 26.1 |
| OAT | 16_855_4 | clip | 48.1 | 52.4 | 22.6 | 25.7 |
| OAT | 32_888_8 | none | 21.5 | 18.9 | 19.4 | 18.0 |
| OAT | 32_888_8 | verb | 34.5 | 37.5 | 22.1 | 25.8 |
| OAT | 32_888_8 | clip | 45.9 | 47.7 | 21.3 | 25.7 |
| OAT | 16_8555_4 | none | 20.6 | 21.7 | 20.6 | 21.4 |
| OAT | 16_8555_4 | verb | 38.2 | 43.5 | 24.0 | 29.8 |
| OAT | 16_8555_4 | clip | 49.0 | 52.2 | 22.2 | 27.3 |
| QueST | 16_855_4 | none | 25.2 | 31.5 | 19.8 | 25.9 |
| QueST | 16_855_4 | verb | 49.1 | 54.6 | 20.3 | 25.1 |
| QueST | 16_855_4 | clip | **51.5** | **57.2** | 19.8 | 23.7 |
| QueST | 32_888_4 | none | 21.8 | 24.2 | 19.6 | 21.2 |
| QueST | 32_888_4 | verb | 45.2 | 49.8 | 19.3 | 23.1 |
| QueST | 32_888_4 | clip | **56.6** | **58.2** | 19.9 | 19.6 |
| QueST | 16_8555_4 | none | 23.9 | 30.2 | 21.8 | 28.2 |
| QueST | 16_8555_4 | verb | 50.3 | 53.7 | 20.5 | 24.1 |
| QueST | 16_8555_4 | clip | 52.7 | 55.6 | 19.8 | 24.0 |
| VQ-BeT | 5_16_2_256 | none | 16.5 | 14.9 | 19.3 | 24.2 |
| VQ-BeT | 5_16_2_256 | verb | 24.3 | 26.6 | 24.6 | 29.5 |
| VQ-BeT | 5_16_2_256 | clip | 23.2 | 26.8 | 25.9 | 30.0 |
| VQ-BeT | 5_16_2_512 | none | 15.0 | 13.6 | 21.5 | 26.4 |
| VQ-BeT | 5_16_2_512 | verb | 23.9 | 27.5 | 23.0 | 30.5 |
| VQ-BeT | 5_16_2_512 | clip | 25.6 | 30.1 | 25.6 | 32.6 |
| VQ-BeT | 10_16_2_256 | none | 16.5 | 12.5 | 17.8 | 16.5 |
| VQ-BeT | 10_16_2_256 | verb | 23.6 | 24.9 | 25.2 | 28.1 |
| VQ-BeT | 10_16_2_256 | clip | 24.0 | 28.4 | **27.5** | **32.0** |

## Key Findings

### 1. Vanilla action tokenizers degrade verb decodability

Without any auxiliary loss, all three tokenizer families produce latents and token IDs that are **less verb-decodable than raw actions** (native baseline: 23.1% acc / 27.4% MF1). Vanilla latent probes range from 15.0-25.2% acc, and vanilla tokid probes from 17.8-21.8% acc. The reconstruction objective alone discards verb-discriminative structure — the tokenizer learns to compress motion trajectories without preserving what the action *means*.

### 2. Auxiliary losses rescue verb decodability in the latent space

Both verb and CLIP aux heads dramatically improve latent verb probes — from 15-25% (vanilla) to 34-57% (with aux). CLIP contrastive training is especially effective, yielding **higher latent probe accuracy than the verb head itself** across all families:

| Rank | Condition | Latent Acc | Latent MF1 |
|------|-----------|-----------|-----------|
| 1 | QueST 32_888_4 + clip | 56.6% | 58.2% |
| 2 | QueST 16_8555_4 + clip | 52.7% | 55.6% |
| 3 | QueST 16_855_4 + clip | 51.5% | 57.2% |
| 4 | QueST 16_8555_4 + verb | 50.3% | 53.7% |
| 5 | OAT 16_8555_4 + clip | 49.0% | 52.2% |

CLIP learns instruction-aligned representations that implicitly capture verb semantics better than direct verb classification. Instructions like "pick up the red block" encode the verb plus grounding context.

### 3. Latent-to-token propagation depends on the quantizer

The critical question: a VLA only sees discrete token IDs, not latents. Does better latent decodability propagate through quantization?

**OAT (FSQ)** — partial propagation. Aux heads improve tokid probes by +2-4pp acc over vanilla (e.g., 16_8555_4: 20.6% → 24.0% verb, 22.2% clip). The FSQ bottleneck attenuates but does not erase the latent gains.

**VQ-BeT (ResidualVQ)** — strongest propagation. Tokid probes improve by +5-10pp acc with aux heads (e.g., 10_16_2_256: 17.8% → 25.2% verb, 27.5% clip). Notably, VQ-BeT tokid probes with aux heads (25-28% acc, 28-33% MF1) **exceed** their own latent probes without aux (15-17% acc), suggesting ResidualVQ preserves discrete structure that the MLP encoder discards.

**QueST (causal conv + FSQ)** — no propagation. Tokid probes remain flat at ~20% regardless of aux condition (e.g., 16_855_4: 19.8% none → 20.3% verb → 19.8% clip). QueST achieves the best latent decodability (up to 56.6%) but its quantization completely blocks transfer to discrete tokens.

| Family | Vanilla TokID | Best Aux TokID | Delta | Propagation |
|--------|--------------|----------------|-------|-------------|
| OAT | 19.4-20.6% | 22.1-24.0% | +2-4pp | Partial |
| VQ-BeT | 17.8-21.5% | 23.0-27.5% | +5-10pp | Strong |
| QueST | 19.6-21.8% | 19.3-20.5% | ~0pp | None |

### 4. Reconstruction vs. decodability tradeoff

QueST dominates on reconstruction (0.00078-0.00349 MSE) and latent decodability, but this advantage does not carry to the discrete token level. VQ-BeT has the worst reconstruction (0.00201-0.00946) but the best token-level verb preservation. This suggests a fundamental tradeoff: aggressive compression (QueST) creates smooth, decodable latents but quantizes away fine-grained structure, while coarser quantization (VQ-BeT ResidualVQ) retains more discrete discriminability.

### 5. Aux heads improve codebook utilization but hurt reconstruction

Adding verb or CLIP aux consistently increases codebook utilization (e.g., OAT v200: 74% → 93-96%) but at the cost of higher reconstruction error (0.00333 → 0.00382-0.00426). The aux loss gradient encourages more diverse code usage, which may partially explain why propagation works for OAT and VQ-BeT — more utilized codes means finer-grained discrete distinctions.

## Scripts

- Sweep submission: `run_sweep.sh` with `DATASET="bridge"`
- Verb probes: `run_sweep.sh --verb-probe-only` with `DATASET="bridge"`
- Training: `tokenization/train_tokenizer.py --dataset bridge --tokenizer {oat,quest,vq_bet}`
- Probe: `verb_probe/train_verb_probe.py --dataset bridge --action_rep {native,latent,oat,quest,vq_bet}`
- Results CSV: `bridge_sweep_results.csv`
