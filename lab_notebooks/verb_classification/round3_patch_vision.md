# Round 3: Patch-Based Vision Encoders (DINOv2 + VC-1)

**Date**: 2025-02-26
**Goal**: Replace R3M's single-token-per-frame with ViT patch tokens, giving the vision model ~50x more spatial information per frame. Also introduce **delta-patch selection** — only attend to patches that change between frames.

## Motivation

Round 2 showed that R3M vision-only models collapse to 2–3 classes (best: 23.1% acc, 4/27 active). Root cause: R3M is a ResNet-50 with global average pooling, producing **1 token per frame** — a massive information bottleneck. Meanwhile, action-only models (64 tokens of 7D actions) reach 39.4%.

**Key insight**: Most image patches are static background. The signal for verb classification is in the *few patches that change* between frames. Rather than feeding all 49 pooled patches (mostly redundant), we select the top-K patches with the largest feature change between consecutive frames.

## Encoders

| Encoder | Type | embed_dim | Native patches (224px) | Params | Source |
|---------|------|-----------|------------------------|--------|--------|
| DINOv2-S | ViT-S/14 | 384 | 16×16 = 256 | 21.6M | `timm` pretrained |
| VC-1 Base | ViT-B/16 | 768 | 14×14 = 196 | 86M | HuggingFace `facebook/vc1-base` |

Both encoders are **frozen** — only the linear projection (embed_dim → 128) and transformer classifier are trained.

### Spatial pooling
Native patches per frame are too many (196–256). We use adaptive avg pool to **7×7 = 49 patches/frame**.

### Delta-patch selection (new)
For each consecutive frame pair, compute per-patch feature difference, select top-K positions by L2 magnitude. Tokens are the **difference vectors** + gathered spatial position embeddings + temporal position. With K=16 and 2 frames: only 17 tokens total (CLS + 16 delta patches).

## Experiments

### All-patches experiments (frozen, 2 frames)

| Tag | Encoder | Classes | Acc | MacF1 | WtF1 | Active | BestAcc | BestMF1 | BestWF1 |
|-----|---------|---------|-----|-------|------|--------|---------|---------|---------|
| vision_dinov2s j6457975 | DINOv2-S | 27 | 27.1% | 9.6% | 19.0% | 9/27 | 27.5% | 10.1% | 19.7% |
| vision_dinov2s_sparse j6457976 | DINOv2-S | 21 | 13.2% | 1.1% | 3.1% | 1/21 | 17.1% | 2.3% | 7.1% |
| vision_vc1 j6457977 | VC-1 | 27 | 17.3% | 1.9% | 7.4% | 2/27 | 17.3% | 1.9% | 7.4% |
| vision_vc1_sparse j6457978 | VC-1 | 21 | 31.4% | 11.6% | 19.6% | 6/21 | 31.7% | 11.6% | 19.8% |

**Observation**: All-patches models are only marginally better than R3M (round 2 best: 23.1%). DINOv2-S 27cls reaches 27.5% with 10 active classes (vs R3M's 4), but still far from action-only. VC-1 is inconsistent — sparse variant does much better (31.7%) while full-class variant collapses (17.3%). The 49 tokens/frame are mostly redundant static background.

### Delta-patch experiments (K=16)

| Tag | Encoder | Frames | Classes | Acc | MacF1 | WtF1 | Active | BestAcc | BestMF1 | BestWF1 |
|-----|---------|--------|---------|-----|-------|------|--------|---------|---------|---------|
| vision_dinov2s_delta16 j6457979 | DINOv2-S | 2 | 27 | 32.8% | 23.3% | 32.0% | 19/27 | **37.6%** | 21.5% | 31.7% |
| vision_dinov2s_delta16_8f j6457980 | DINOv2-S | 8 | 27 | 31.6% | 22.2% | 30.7% | 19/27 | **37.9%** | 22.2% | 33.8% |
| vision_vc1_delta16 j6458015 | VC-1 | 2 | 27 | 37.5% | 23.8% | 35.2% | 19/27 | **40.4%** | 23.0% | 34.3% |

**Delta patches are a game-changer.** VC-1 delta16 reaches **40.4% best accuracy with 15/27 active classes** at epoch 9 — surpassing action-only native (39.4%). This is the first vision-only model to beat action-only.

## Key Findings

![Vision Analysis](../figures/vision_analysis.png)

### 1. Delta patches >> all patches
Focusing on the K=16 most-changed patches eliminates static background noise and forces the model to attend to the actual motion signal. The improvement is dramatic:
- DINOv2-S all patches: 27.5% acc, 10/27 active
- DINOv2-S delta K=16: 37.6% acc, 15/27 active (+10.1 pp)

### 2. Eight frames ≈ two frames for delta mode
DINOv2-S delta16 with 8 frames (37.9%) barely beats 2 frames (37.6%). The endpoint signal is what matters most for verb classification.

### 3. VC-1 > DINOv2-S for delta patches
VC-1 (pretrained on Ego4D egocentric video + robot data via MAE) reaches 40.4% vs DINOv2-S's 37.6%. VC-1's pretraining on egocentric manipulation data likely gives it better patch features for robot actions.

### 4. Per-class coverage dramatically improved
Best R3M model predicted only 2–4 classes. Delta models predict 15–19 classes. VC-1 delta best-val per-class recall:
- `open` (100%), `rotate` (89.5%), `pick up` (74%), `turn off` (70.6%), `push` (67%)
- `turn` (63.6%), `move` (50%), `stack` (46.2%), `place` (35.5%), `left` (30%)
- `put` (28%), `remove` (25%), `grasp` (18.3%), `store` (16.7%), `slide` (5.1%)
- Still fails on: `sweep`, `lift`, `lift up`, `collapse`, `go`, `move up`, `slide down/up`, `take off`, `close`, `pull`, `unstack`

### 5. Significant overfitting
DINOv2-S delta: train accuracy 90.7% at epoch 30, val peaks at 37.6% (epoch 9). The gap suggests:
- 3,400 training samples insufficient for 27-class classification
- Model memorizes training set rapidly with frozen backbone
- Early stopping (epoch ~9) is critical

## Training Curves (Delta Experiments)

### DINOv2-S delta16 2f (best val at epoch 9)
```
Ep  1: train=14.0% val=20.8%
Ep  3: train=29.1% val=31.2%
Ep  5: train=36.0% val=32.2%
Ep  7: train=40.5% val=36.7%
Ep  9: train=44.6% val=37.6% ← best
Ep 15: train=56.9% val=33.7%
Ep 20: train=74.7% val=32.4%
Ep 30: train=90.7% val=32.8%
```

### DINOv2-S delta16 8f (best val at epoch 14)
```
Ep  1: train=15.3% val=16.1%
Ep  5: train=36.3% val=36.2%
Ep  9: train=44.6% val=37.3%
Ep 14: train=59.2% val=37.9% ← best
Ep 20: train=85.2% val=32.4%
Ep 30: train=96.5% val=31.6%
```

### VC-1 delta16 2f (best val at epoch 9)
```
Ep  1: train=14.4% val=24.0%
Ep  3: train=34.6% val=36.7%
Ep  6: train=43.1% val=39.1%
Ep  9: train=43.9% val=40.4% ← best
Ep 15: train=54.8% val=37.9%
Ep 20: train=67.6% val=37.8%
Ep 30: train=84.1% val=37.5%
```

## Comparison to Previous Rounds

| Model | Acc | MacF1 | Active | Notes |
|-------|-----|-------|--------|-------|
| Action-only native (R1) | 39.4% | — | 21/27 | Baseline |
| R3M finetune (R2 best) | 23.1% | 3.5% | 4/27 | 1 token/frame |
| DINOv2-S all patches (R3) | 27.5% | 10.1% | 10/27 | 49 tokens/frame |
| DINOv2-S delta K=16 (R3) | 37.6% | 21.5% | 15/27 | 16 diff tokens |
| **VC-1 delta K=16 (R3)** | **40.4%** | **23.0%** | **15/27** | **16 diff tokens** |

Vision-only is now competitive with action-only for the first time.

## Implementation Notes

### ViTEncoder class
- Uses `timm` for both DINOv2 and VC-1
- DINOv2: `timm.create_model('vit_small_patch14_dinov2', pretrained=True, img_size=224)`
- VC-1: `timm.create_model('vit_base_patch16_224')` + downloaded HF weights with key remapping
- Spatial pooling: `AdaptiveAvgPool2d(7)` → 49 patches/frame
- Linear projection: embed_dim → d_model (128)
- All backbone params frozen

### Delta-patch forward pass
```python
# For each consecutive frame pair:
diff = all_patches[i+1] - all_patches[i]     # (B, 49, d_model)
mag = diff.norm(dim=-1)                        # (B, 49)
topk_idx = mag.topk(K, dim=-1).indices         # (B, K)
selected = torch.gather(diff, 1, idx_exp)      # (B, K, d_model)
pos = torch.gather(patch_pos, 1, idx_exp)      # gather spatial pos for selected patches
tokens = selected + pos + frame_pos + type_img  # add all embeddings
```

## Open Questions / Next Steps

1. **Try K=8 and K=32**: Is K=16 optimal, or would fewer/more delta patches help?
2. **Weighted delta + sparse classes**: Combine delta patches with weighted CE and sparse class filtering
3. **Multimodal fusion**: Now that vision is competitive (40.4%), combine with action-only (39.4%) — the two modalities likely capture complementary information
4. **Regularization**: Severe overfitting (train 90% vs val 37%) — try dropout, label smoothing, or data augmentation
