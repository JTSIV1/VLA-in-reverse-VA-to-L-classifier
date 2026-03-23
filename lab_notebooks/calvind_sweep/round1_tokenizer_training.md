# Round 1: Tokenizer Training on CALVIN D→D

Date: 2026-03-20 → 2026-03-21

## Overview

Trained 3 gradient-based tokenizers (VQ-BeT, OAT, QueST) on CALVIN D→D with:
- Vanilla (recon only)
- Verb classification head (sweep verb_cls_lambda)
- CLIP contrastive head (sweep clip_lambda)

All jobs use `tokenization/train_tokenizer.py` with 200 max epochs, batch_size=64,
early stopping (patience=15).

## Vanilla Results (completed)

| Tokenizer | Params | Best Epoch | Val Recon | Early Stop |
|-----------|--------|-----------|-----------|------------|
| VQ-BeT    | 57K    | ~185      | 0.0059    | ep 200 (no) |
| QueST     | 6.5M   | ~182      | 0.0157    | ep 200 (no) |
| OAT       | 5.8M   | ~56       | 0.0460    | ep 71       |

VQ-BeT achieves best reconstruction despite being 100x smaller.
QueST trains longest before converging. OAT early-stops soonest with worst recon.

## VQ-BeT Verb Sweep (completed)

| verb_cls_lambda | Best Epoch | Val Recon (last) | Val Verb Loss | Best Val Verb Acc |
|----------------|-----------|-----------------|---------------|------------------|
| 0 (vanilla)    | 199       | 0.0059          | —             | —                |
| 0.01           | 37        | 0.0076          | 1.445         | 36.5%            |
| 0.1            | 29        | 0.0094          | 1.357         | 36.8%            |
| 0.5            | 85        | 0.0110          | 1.211         | **41.1%**        |
| 1.0            | 62        | 0.0121          | 1.279         | 39.5%            |

Trade-off: higher lambda → worse recon but better verb acc.
**Lambda=0.5 gives best verb acc (41.1%)** with moderate recon degradation (~2x vanilla).
Gradient flows through continuous latents before VQ quantization — joint training works.

## VQ-BeT CLIP Sweep (completed)

| clip_lambda | Best CLIP Ep | Val Recon (last) | Best Val CLIP |
|-------------|-------------|-----------------|---------------|
| 0.1         | 54          | 0.0081          | 1.978         |
| 0.5         | 45          | 0.0101          | 1.966         |
| 1.0         | 52          | 0.0100          | 1.946         |
| 2.0         | 44          | 0.0100          | **1.936**     |

CLIP loss converges to ~2.0 regardless of lambda. Higher lambda → marginally better CLIP alignment.
Recon impact similar to verb sweep (~1.7x vanilla).

## OAT Verb Sweep (completed)

| verb_cls_lambda | Best Epoch | Val Verb Loss | Best Val Verb Acc |
|----------------|-----------|---------------|------------------|
| 0 (vanilla)    | 56        | —             | —                |
| 0.01           | 32        | 2.595         | **21.6%**        |
| 0.1            | 10        | 2.654         | 21.2%            |
| 0.5            | 11        | —             | 18.2%            |
| 1.0            | 13        | —             | 21.0%            |

OAT verb accuracy much lower than VQ-BeT (~21% vs ~41%).
Inconsistent pattern across lambda — higher lambda does NOT monotonically improve acc.
Note: OAT verb head cannot backprop through FSQ discrete codes — verb head learns as
a fixed linear probe on 4-dim discrete tokens, not joint training.

## OAT CLIP Sweep (completed)

| clip_lambda | Best CLIP Ep | Val Recon (last) | Best Val CLIP |
|-------------|-------------|-----------------|---------------|
| 0.1         | 38          | 0.0503          | 3.664         |
| 0.5         | 88          | 0.0364          | **3.602**     |
| 1.0         | 57          | 0.0429          | 3.649         |
| 2.0         | 45          | 0.0458          | 3.644         |

OAT CLIP loss much higher than VQ-BeT (~3.6 vs ~2.0). Interesting: clip_lambda=0.5
gives best OAT recon (0.036), *better* than vanilla (0.043) — CLIP regularizes OAT encoder.
Note: OAT's CLIP head also cannot backprop through FSQ — only affects encoder pre-quantization.

## QueST Verb Sweep (completed)

| verb_cls_lambda | Best Epoch | Val Recon (last) | Val Verb Loss | Best Val Verb Acc |
|----------------|-----------|-----------------|---------------|------------------|
| 0 (vanilla)    | 185       | 0.0157          | —             | —                |
| 0.01           | 5         | 0.0314          | 2.418         | 13.8%            |
| 0.1            | 15        | 0.0306          | 2.506         | 13.8%            |
| 0.5            | 11        | 0.0275          | 2.463         | **15.2%**        |
| 1.0            | 17        | 0.0346          | 2.453         | 15.2%            |

QueST verb accuracy also low (~15%), similar to OAT. VQ-BeT dominates.
Verb head causes much earlier early-stopping (ep 5-17 vs 185 vanilla).
Same FSQ gradient issue as OAT — verb head is a fixed probe on 4-dim codes.

## QueST CLIP Sweep (completed)

| clip_lambda | Best CLIP Ep | Val Recon (last) | Best Val CLIP |
|-------------|-------------|-----------------|---------------|
| 0.1         | 91          | 0.0190          | **3.051**     |
| 0.5         | 86          | 0.0193          | 3.135         |
| 1.0         | 124         | 0.0177          | 3.106         |
| 2.0         | 113         | 0.0189          | 3.080         |

QueST CLIP loss (~3.1) between VQ-BeT (~2.0) and OAT (~3.6).
CLIP head barely affects QueST recon — stays near vanilla. clip_lambda=0.1 gives best CLIP loss.

## Key Findings (Round 1 — now superseded by Round 1b resubmission)

1. **VQ-BeT dominates** on both recon and verb decodability despite 100x fewer params
2. **Verb head trade-off clear**: lambda=0.5 best for VQ-BeT (**41.1%** acc, ~2x recon cost)
3. **OAT/QueST verb acc low** (~15-21%) — but Round 1 had a critical gradient bug (see below)
4. **CLIP loss plateau**: all tokenizers converge to a floor (~2.0 VQ-BeT, ~3.1 QueST, ~3.6 OAT)
5. **CLIP regularization**: OAT+CLIP(0.5) gets *better* recon than vanilla OAT (0.036 vs 0.043)

## Auxiliary Head Architecture (verb & CLIP)

All three tokenizers apply the verb/CLIP head to **post-quantization latents**
(z_q). This measures whether the discrete codes actually retain verb/language
information. Gradients flow back through the quantizer's straight-through
estimator (STE) to the encoder for joint training.

The classification model is a small Transformer with a learnable [CLS] token:
```
z_q tokens (B, n_chunks, latent_dim) → [CLS] prepended → + pos_emb → Transformer → CLS output
  → VerbHead: LayerNorm → ReLU → Dropout → Linear(d_model, num_verbs)
  → ContrastiveHead: Linear(d_model, proj_dim) → L2-normalize
```
Same architecture for both verb and CLIP heads (1-layer for verb, 2-layer for CLIP).
Same model used for joint training (`--verb_cls_lambda`) and frozen probing
(`--freeze_tokenizer --resume ...`).

### VQ-BeT
```
chunk (4×7=28,) → MLP encoder → z (64,)         [continuous]
               → ResidualVQ(groups=2, codebook=512)
               → z_q (64,)                        [STE: forward=codebook, backward=z]
               → MLP decoder → recon (28,)

Per trajectory (~15 chunks):
  z_q (15, 64) → [CLS] + z_q tokens + pos_emb → Transformer(1-layer) → CLS out (128,)
               → classifier → verb logits
```

### OAT
```
actions (20, 7) → normalize → RegisterEncoder(emb=256→4)
               → (4 registers, 4)                  [continuous]
               → custom FSQ([8,5,5,5])  round_ste
               → (4, 4)                             [post-FSQ, STE grad]
               → SinglePassDecoder → recon

  z_q (4, 4) → [CLS] + tokens + pos_emb → Transformer → CLS out (128,)
             → classifier → verb logits
```

### QueST
```
actions (20, 7) → normalize → action_proj(7→256) → causal_conv(↓4x) → TransformerEncoder
               → (5, 256)                          [continuous]
               → FSQ(dim=256, levels=[8,5,5,5])
                   project_in(256→4) → round_ste → project_out(4→256)
               → (5, 256)                          [post-FSQ, STE grad]
               → TransformerDecoder → recon

  z_q (5, 256) → [CLS] + tokens + pos_emb → Transformer → CLS out (128,)
               → classifier → verb logits
```

### Gradient flow summary

| Tokenizer | Head on | Quantizer | STE? | head_latent_dim | Joint training? |
|-----------|---------|-----------|------|-----------------|-----------------|
| VQ-BeT    | z_q (post-ResidualVQ) | ResidualVQ | yes | 64 | yes |
| OAT       | post-FSQ codes | custom FSQ | yes | 4 | yes |
| QueST     | post-FSQ (project_out) | vq_pytorch FSQ | yes | 256 | yes |

### Joint training vs frozen probe
- **Joint training**: `--verb_cls_lambda 0.5` — all params (tokenizer + head) trained
- **Frozen probe**: `--freeze_tokenizer --resume ckpt.pth --verb_cls_lambda 1.0` — tokenizer frozen, only head trained

## Vanilla Recon Summary

| Tokenizer | Params | Best Val Recon |
|-----------|--------|---------------|
| VQ-BeT    | 57K    | 0.00587       |
| QueST     | 6.5M   | 0.01510       |
| OAT       | 5.8M   | 0.04315       |

## Bug Fixes During This Round

1. **4D→3D shape** (2026-03-20): `CalvinTokenizerDataset` returns `(B, max_windows, ws, D)`
   but OAT/QueST expect `(B, T, D)`. Fixed squeeze in `extract_latents_oat_quest()`.
2. **head_latent_dim** (2026-03-21): OAT verb/CLIP heads were incorrectly set to 256
   (encoder dim) instead of 4 (FSQ code dim). Fixed OAT to 4. QueST stays at 256
   because `vector_quantize_pytorch.FSQ(dim=256)` has `project_out(4→256)`.
3. **torch.no_grad() on OAT/QueST encode** (2026-03-21): `extract_latents_oat_quest()`
   wrapped `model.encode()` in `torch.no_grad()`, blocking all gradient flow from
   aux heads to the encoder — making the verb/CLIP head a frozen linear probe.
   Removed `no_grad` so FSQ's STE can propagate gradients. All Round 1 OAT/QueST
   verb+CLIP results are **invalid** (no joint training happened). Resubmitting.
4. **Early stopping criterion** (2026-03-21): Changed from micro accuracy to weighted
   val verb loss for verb runs, and from val_total to R@1 retrieval for CLIP runs.
5. **CLIP retrieval eval** (2026-03-21): Added top-k retrieval (R@1/R@5/R@10) to
   training loop for CLIP conditions, logged in metrics CSV.
6. **CLS token pooling** (2026-03-21): Replaced mean-pooling in VerbHead with
   a 1-layer ActionTransformer + CLS token (same pattern as ContrastiveHead).
   Mean-pooling discards temporal ordering; CLS token preserves it via attention.

## Plots

- `vanilla_training.png` — train/val recon curves for 3 tokenizers
- `verb_sweep.png` — recon + verb acc per tokenizer across lambda
- `clip_sweep.png` — recon + CLIP loss per tokenizer across lambda
- `recon_vs_verb_tradeoff.png` — scatter of best recon vs best verb acc

## Job IDs (final)

### Completed
| Run | Job ID | Status |
|-----|--------|--------|
| vq_bet_vanilla | 6684790 | completed |
| vq_bet_verb0.01 | 6684793 | completed |
| vq_bet_verb0.1 | 6684794 | completed |
| vq_bet_verb0.5 | 6684795 | completed |
| vq_bet_verb1.0 | 6684796 | completed |
| vq_bet_clip0.1 | 6684805 | completed |
| vq_bet_clip0.5 | 6684806 | completed |
| vq_bet_clip1.0 | 6684807 | completed |
| vq_bet_clip2.0 | 6684808 | completed |
| oat_vanilla | 6684791 | completed |
| oat_verb0.01 | 6684820 | completed |
| oat_verb0.1 | 6684821 | completed |
| oat_clip0.1 | 6684809 | completed |
| oat_clip0.5 | 6684810 | completed |
| oat_clip1.0 | 6684811 | completed |
| oat_clip2.0 | 6684812 | completed |
| quest_vanilla | 6684792 | completed |
| quest_verb0.01 | 6685144 | completed |
| quest_verb0.1 | 6685145 | completed |
| quest_verb0.5 | 6685146 | completed |
| quest_verb1.0 | 6685147 | completed |
| quest_clip0.1 | 6685148 | completed |
| quest_clip0.5 | 6685149 | completed |
| quest_clip1.0 | 6685150 | completed |
| quest_clip2.0 | 6685151 | completed |

| oat_verb0.5 | 6687869 | completed |
| oat_verb0.1 | 6684821 | completed |
| oat_verb1.0 | 6687870 | completed |
