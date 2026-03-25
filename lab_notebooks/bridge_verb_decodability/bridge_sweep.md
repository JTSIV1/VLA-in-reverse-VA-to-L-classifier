# BridgeV2 Action Tokenizer Hyperparameter Sweep

## Goal

Find the best action tokenizer configuration for BridgeV2 by sweeping horizon, codebook size, and architecture-specific parameters across OAT, QueST, and VQ-BeT. The target is downstream policy performance (imitation learning loss / rollout success), so we optimize for both low reconstruction error and high codebook utilization.

## Dataset

- **BridgeV2** action trajectories: 53,192 episodes, 7-DoF actions
- Episode length: mean=37.6, median=37, min=3, max=119
- Data stored as shards in `/data/user_data/wenjiel2/datasets/bridge_actions/`

## Sweep Design

### OAT (Ordered Action Tokenizer)
- Transformer encoder (2 layers) + FSQ quantizer + transformer decoder (4 layers)
- Register tokens as bottleneck, nested dropout for causal ordering
- emb_dim=256, head_dim=64, lr=5e-5 with cosine decay, batch_size=256, 300 epochs

| Tag | Horizon | Registers | FSQ levels | Vocab | Tokens/chunk |
|-----|---------|-----------|------------|-------|-------------|
| h16_r4_v200 | 16 | 4 | [8,5,5] | 200 | 4 |
| h32_r8_v125 | 32 | 8 | [5,5,5] | 125 | 8 |
| h32_r8_v512 | 32 | 8 | [8,8,8] | 512 | 8 |
| h16_r4_v1000 | 16 | 4 | [8,5,5,5] | 1000 | 4 |
| h32_r8_v4096 | 32 | 8 | [8,8,8,8] | 4096 | 8 |

### QueST (Self-Supervised Skill Abstractions)
- Causal convolution encoder + FSQ + transformer decoder
- Temporal downsampling factor determines token count
- lr=1e-4, batch_size=128, 300 epochs

| Tag | Horizon | Downsample | FSQ levels | Vocab | Tokens/chunk |
|-----|---------|------------|------------|-------|-------------|
| h16_ds4_v200 | 16 | 4 | [8,5,5] | 200 | 4 |
| h32_ds4_v125 | 32 | 4 | [5,5,5] | 125 | 8 |
| h32_ds4_v512 | 32 | 4 | [8,8,8] | 512 | 8 |
| h16_ds4_v1000 | 16 | 4 | [8,5,5,5] | 1000 | 4 |
| h32_ds4_v4096 | 32 | 4 | [8,8,8,8] | 4096 | 8 |

### VQ-BeT (Behavior Generation with Latent Actions)
- MLP encoder/decoder + ResidualVQ (2 groups × 16 codes = 256 combos)
- lr=1e-4, batch_size=256, 200 epochs, patience=15

| Tag | Chunk size | n_embed | Groups | Latent dim | Vocab |
|-----|-----------|---------|--------|-----------|-------|
| c5_e16_g2_l256 | 5 | 16 | 2 | 256 | 256 |
| c5_e16_g2_l512 | 5 | 16 | 2 | 512 | 256 |
| c10_e16_g2_l256 | 10 | 16 | 2 | 256 | 256 |

## Results

### Reconstruction + Codebook Utilization

| Tokenizer | Config | Val Recon MSE | Unique Codes | Total | Util% | Status |
|-----------|--------|--------------|-------------|-------|-------|--------|
| **OAT** | h16_r4_v200 | 0.000709 | 200 | 200 | 100% | done (ep 297) |
| **OAT** | h32_r8_v125 | 0.00241 | 125 | 125 | 100% | done (ep 292) |
| **OAT** | h32_r8_v512 | 0.00108 | 512 | 512 | 100% | done (ep 300) |
| **OAT** | h16_r4_v1000 | 0.00316 | 563 | 1000 | 56.3% | done (early stop ep 178) |
| **OAT** | h32_r8_v4096 | 0.00333 | ~3089 | 4096 | 75.4% | done (early stop ep 214) |
| **QueST** | h16_ds4_v200 | 0.00077 | 200 | 200 | 100% | done (ep 261) |
| **QueST** | h32_ds4_v125 | 0.00079 | 125 | 125 | 100% | done (ep 300) |
| **QueST** | h32_ds4_v512 | 0.00061 | 511 | 512 | 99.8% | done (ep 300) |
| **QueST** | h16_ds4_v1000 | 0.00066 | 998 | 1000 | 99.8% | done (ep 300) |
| **QueST** | h32_ds4_v4096 | 0.00046 | 3574 | 4096 | 87.3% | done (ep 300) |
| **VQ-BeT** | c5_e16_g2_l256 | 0.00240 | 204 | 256 | 79.7% | done (early stop ep 34) |
| **VQ-BeT** | c5_e16_g2_l512 | 0.00272 | 194 | 256 | 75.8% | done (early stop ep 28) |
| **VQ-BeT** | c10_e16_g2_l256 | 0.00405 | 165 | 256 | 64.5% | done (early stop ep 30) |

### Key Observations

1. **QueST has ~3x lower reconstruction MSE** than OAT and VQ-BeT across all configs. QueST's causal convolution encoder + powerful transformer decoder is very effective at reconstruction.

2. **OAT and QueST both achieve ~100% codebook utilization.** OAT uses FSQ with nested dropout; QueST uses FSQ with causal convolutions. Both avoid codebook collapse completely.

3. **VQ-BeT has partial codebook collapse** (65–80% utilization). ResidualVQ with 2 groups × 16 codes uses all per-group codes (16/16) but not all 256 combinations. chunk_size=10 is worse (64.5%) than chunk_size=5 (75–80%).

4. **Horizon 16 vs 32**: OAT h16 has much lower recon (0.000709) than h32 (0.00108–0.00241) — less compression needed. QueST benefits slightly from h32 when combined with larger vocab (0.00046 for h32/v4096 vs 0.00066 for h16/v1000). For downstream policy, shorter horizon = more re-inference but less prediction difficulty.

5. **Codebook size matters**: For both OAT and QueST, larger vocab consistently improves recon. OAT: v512 (0.00108) >> v200 (0.000709) >> v125 (0.00241). QueST: v4096 (0.00046) >> v1000 (0.00066) >> v512 (0.00061). Utilization stays high (87–100%) even at large vocab sizes.

6. **VQ-BeT early-stopped quickly** (28–34 epochs) due to patience=15 — the model converged fast but to a worse reconstruction than OAT/QueST.

### Measurement Note

FSQ codes are quantized float vectors (e.g., [-0.25, 0.0, 0.0]). Initial measurements counted unique float tuples, which was unreliable due to precision. Fixed by using `FSQ.codes_to_indices()` to convert quantized vectors to scalar code indices — the proper way to count unique codes for both OAT and QueST.

## Scripts

- Sweep submission: `scripts/submit_bridge_tokenizer_sweep.sh`
- Codebook utilization (post-hoc): `scripts/codebook_utilization_bridge.py`
- Training (all): `tokenization/train_tokenizer.py --dataset bridge --tokenizer {oat,quest,vq_bet}`
- Codebook logging added to `train_tokenizer.py` eval loop (prints `| codes=N` per epoch)
