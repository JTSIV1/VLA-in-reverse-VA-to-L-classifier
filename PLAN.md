# Round 3 Plan: DINOv2 + VC-1 Patch-Based Vision Encoders

## Goal
Replace R3M's single-token-per-frame with ViT patch tokens, giving the vision model ~50x more spatial information per frame. Quick experiment before moving to multimodal.

## Models

| Encoder | Type | embed_dim | Native patches (224x224) | Params | Source |
|---------|------|-----------|--------------------------|--------|--------|
| DINOv2-S | ViT-S/14 | 384 | 16×16 = 256 | 21.6M | `timm` pretrained (works on cluster) |
| VC-1 Base | ViT-B/16 | 768 | 14×14 = 196 | 86M | HuggingFace `facebook/vc1-base` (downloaded, load into timm ViT-B) |

## Token count with spatial pooling

Full patches per frame are too many (196-256). Use adaptive avg pool to 7×7 = 49 patches/frame:

| Setup | Tokens |
|-------|--------|
| 2 frames, 49 patches/frame | CLS + 2×49 = 99 |
| 8 frames, 49 patches/frame | CLS + 8×49 = 393 |
| Action-only (native) | CLS + ~64 |

99 tokens for 2 frames is comparable to action-only — a fair comparison.

## Implementation Changes

### 1. New `ViTEncoder` class in `train_transformer.py`

```python
class ViTEncoder(nn.Module):
    """Frozen ViT patch encoder (DINOv2 or VC-1)."""
    def __init__(self, d_model, variant="dinov2_s", pool_size=7):
        # Load backbone, freeze all params
        # DINOv2: timm.create_model('vit_small_patch14_dinov2', img_size=224)
        # VC-1: timm.create_model('vit_base_patch16_224') + load HF weights
        # Adaptive avg pool: native grid -> pool_size×pool_size
        # Linear proj: embed_dim -> d_model
        self.num_patches = pool_size * pool_size  # 49

    def forward(self, x):
        # x: (B, 3, 224, 224)
        # -> extract patch tokens (skip ViT CLS)
        # -> reshape to spatial grid, adaptive avg pool to 7×7
        # -> project to d_model
        # returns: (B, 49, d_model)
```

### 2. Model constructor (`ActionToVerbTransformer.__init__`)
Add branch for new encoders:
```python
elif vision_encoder in ("dinov2_s", "dinov2_b", "vc1"):
    self.vision_enc = ViTEncoder(d_model, variant=vision_encoder)
    self.num_patches = self.vision_enc.num_patches  # 49
```

Everything downstream (patch_pos, frame_pos, forward pass) is already parameterized by `num_patches` — no other changes needed.

### 3. Image transforms
DINOv2 and VC-1 both use 224×224 + ImageNet normalization — same as R3M. Just extend the img_size logic:
```python
img_size = 224 if args.vision_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1") else IMAGE_SIZE[0]
```

### 4. Checkpoint saving/loading
Already saves `vision_encoder` string — no changes needed.

### 5. `test_transformer.py`
Just needs to import/handle the new encoder types — same pattern as R3M.

## Experiments

| Tag | Encoder | Freeze | Frames | #Cls | Notes |
|-----|---------|--------|--------|------|-------|
| vision_dinov2s_frozen | DINOv2-S | yes | 2 | 27 | baseline, all classes |
| vision_dinov2s_frozen_sparse | DINOv2-S | yes | 2 | 21 | sparse filtering |
| vision_vc1_frozen | VC-1 | yes | 2 | 27 | baseline, all classes |
| vision_vc1_frozen_sparse | VC-1 | yes | 2 | 21 | sparse filtering |

All frozen, no weighted CE. 30 epochs, batch_size=16, lr=5e-4, d_model=128, 4 layers.

Start frozen only — with 3,400 samples and 22-86M param backbones, finetuning is risky.

## Files to modify
1. `train_transformer.py` — add ViTEncoder class + model constructor branch + img_size logic
2. `test_transformer.py` — handle new vision_encoder values in checkpoint loading
3. `submit_round3.sh` — new SLURM submission script (4 jobs)
