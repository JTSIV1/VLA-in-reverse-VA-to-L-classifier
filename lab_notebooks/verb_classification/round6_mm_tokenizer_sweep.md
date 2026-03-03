# Round 6: Multimodal Action Tokenizer Sweep

**Date**: 2026-02-27
**Motivation**: R4 established the best multimodal architecture (VC-1 delta16, late2, d128, sp+wt: 42.4% acc / 40.7% MacF1). R5 ablations did not improve on this. This round fixes the best architecture and sweeps the **action representation** axis across all available tokenizers, with both VC-1 and DINOv2-S as vision encoders.

**Core question**: Does fusion with vision rescue lossy action tokenizers (FAST, VQ-VLA) that underperformed native in action-only mode?

---

## Setup

### Fixed architecture (from R4 best)
- Modality: `full` (multimodal: vision + action)
- Vision: delta patches, K=16 (top-16 patches by L2 change between consecutive frames)
- Cross-attention: `late2` (last 2 of 4 transformer layers use cross-modal attention)
- d_model=128, nhead=8, num_layers=4
- 30 epochs, batch_size=16, lr=5e-4, OneCycleLR
- Sparse filtering: `--min_class_count 30` (21 classes)
- Weighted CE loss

### Experiment grid: 2 × 14 = 28 jobs

| Vision encoder | Action reps |
|----------------|-------------|
| VC-1 (facebook/vc1-base, 768-d, frozen) | native, vqvla, 12 FAST variants |
| DINOv2-S (vit_small_patch14_dinov2, 384-d, frozen) | native, vqvla, 12 FAST variants |

### Action representations

| Action rep | Type | Tokens/traj | Vocab | Notes |
|------------|------|-------------|-------|-------|
| native | continuous | ~61 | — | Linear(7→d_model) per timestep |
| vqvla | discrete (pretrained) | ~48 | 256 | VQ-VLA PE model, 5-step windows × 4 RVQ codes |
| FAST s1/v256 | discrete (fitted) | ~27 | 256 | Best AO FAST (30.9%) |
| FAST s1/v512 | discrete (fitted) | ~25 | 512 | |
| FAST s1/v1024 | discrete (fitted) | ~24 | 1024 | |
| FAST s5/v256 | discrete (fitted) | ~95 | 256 | |
| FAST s5/v512 | discrete (fitted) | ~79 | 512 | |
| FAST s5/v1024 | discrete (fitted) | ~70 | 1024 | |
| FAST s10/v512 | discrete (fitted) | ~138 | 512 | Original default |
| FAST s20/v512 | discrete (fitted) | ~264 | 512 | |
| FAST s20/v768 | discrete (fitted) | ~240 | 768 | |
| FAST s20/v1024 | discrete (fitted) | ~230 | 1024 | |
| FAST s50/v1024 | discrete (fitted) | ~303 | 1024 | |
| FAST s50/v1536 | discrete (fitted) | ~255 | 1536 | |

---

## Prior AO-only results (for reference)

| Action rep | AO BestAcc | AO BestMF1 | Active |
|------------|-----------|-----------|--------|
| native sp+wt | 39.5% | 38.7% | 21/21 |
| vqvla sp+wt | 35.6% | 31.5% | 19/21 |
| FAST s1/v256 (27 cls) | 30.9% | 9.9% | 14/27 |

---

## Results

_Results will be filled in as jobs complete._

### VC-1 vision encoder

| Action rep | BestAcc | BestMF1 | Active | Job ID |
|------------|---------|---------|--------|--------|
| native | | | | |
| vqvla | | | | |
| FAST s1/v256 | | | | |
| FAST s1/v512 | | | | |
| FAST s1/v1024 | | | | |
| FAST s5/v256 | | | | |
| FAST s5/v512 | | | | |
| FAST s5/v1024 | | | | |
| FAST s10/v512 | | | | |
| FAST s20/v512 | | | | |
| FAST s20/v768 | | | | |
| FAST s20/v1024 | | | | |
| FAST s50/v1024 | | | | |
| FAST s50/v1536 | | | | |

### DINOv2-S vision encoder

| Action rep | BestAcc | BestMF1 | Active | Job ID |
|------------|---------|---------|--------|--------|
| native | | | | |
| vqvla | | | | |
| FAST s1/v256 | | | | |
| FAST s1/v512 | | | | |
| FAST s1/v1024 | | | | |
| FAST s5/v256 | | | | |
| FAST s5/v512 | | | | |
| FAST s5/v1024 | | | | |
| FAST s10/v512 | | | | |
| FAST s20/v512 | | | | |
| FAST s20/v768 | | | | |
| FAST s20/v1024 | | | | |
| FAST s50/v1024 | | | | |
| FAST s50/v1536 | | | | |

---

## Analysis

_To be completed after all jobs finish._

### Key questions to answer:
1. Does multimodal fusion close the gap between native and discrete tokenizers?
2. Does the ranking of FAST variants change in multimodal vs action-only?
3. Is VQ-VLA competitive with FAST when augmented with vision?
4. Does VC-1 vs DINOv2-S interact with the action representation choice?

---

## Experiment Index

| Job ID | Name | Vision | Action | Status |
|--------|------|--------|--------|--------|
| | r6_vc1_native | VC-1 | native | pending |
| | r6_vc1_vqvla | VC-1 | vqvla | pending |
| | r6_vc1_fs1v256 | VC-1 | FAST s1/v256 | pending |
| ... | ... | ... | ... | ... |
| | r6_dv2_native | DINOv2-S | native | pending |
| | r6_dv2_vqvla | DINOv2-S | vqvla | pending |
| | r6_dv2_fs1v256 | DINOv2-S | FAST s1/v256 | pending |
| ... | ... | ... | ... | ... |
