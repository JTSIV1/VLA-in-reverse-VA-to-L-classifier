# Tokenizer Verb Decodability Comparison

**Date**: 2026-03-17
**Goal**: Generate a clean comparison table (paper Table 2) showing how well verbs
can be decoded from action trajectories under different tokenization schemes.

## Experimental Setup

### Shared conditions
- **Dataset**: CALVIN D→D, ~3,380 train / 666 val episodes (after sparse filtering)
- **Classes**: 20 verb classes (`--min_class_count 30`, drops `collapse` and `unstack`)
- **Loss**: Weighted cross-entropy (inverse class frequency)
- **Classifier**: 4-layer bidirectional transformer encoder, d_model=128, 8 heads, dropout=0.1
- **Training**: 30 epochs, batch_size=16, lr=5e-4 (OneCycleLR), max_seq_len=64 (192 for FAST+)
- **Evaluation**: Best-val-accuracy checkpoint, per-class precision/recall/F1 via sklearn

### Why 20 classes
`collapse` (18 train, 5 val) and `unstack` (24 train, 3 val) have too few samples for reliable
evaluation. Consistent with all previous experiments (R1–R7) and the paper's convention.

## Tokenizers Compared

| # | Tokenizer | Source | Type | Tok/traj | Vocab | Description |
|---|-----------|--------|------|----------|-------|-------------|
| 1 | **Native continuous** | — | continuous | ~61 | — | Linear(7→d_model) per timestep; no information loss |
| 2 | **VQ-VLA pretrained** | Li et al. 2024 | VQ-VAE | ~48 | 256 | 4-group residual VQ, trained on Open X-Embodiment; pretrained weights from paper |
| 3 | **VQ-VLA ft vanilla** | ours | VQ-VAE | ~48 | 256 | Same architecture, finetuned on CALVIN with λ=0 (reconstruction only, no verb head) |
| 4 | **FAST+ pretrained** | Pertsch et al. 2025 | DCT+BPE | ~128 | 2048 | Pretrained on DROID (1M real robot trajectories); scale=10, vocab=2048 |
| 5 | **FAST s1/v256** | ours (fitted) | DCT+BPE | ~27 | 256 | BPE fitted on CALVIN; scale=1 (coarsest quantization), vocab=256; best FAST config from R3b sweep |

### Tokenizers planned (later)
| # | Tokenizer | Source | Type | Status |
|---|-----------|--------|------|--------|
| 6 | VQ-BeT | ours | VQ-VAE | fit on CALVIN from scratch, TODO |
| 7 | OAT | ours | VQ-VAE | fit on CALVIN from scratch, TODO |
| 8 | QueST | ours | VQ-VAE | fit on CALVIN from scratch, TODO |

## Jobs

| Job ID | Name | Tokenizer | Status |
|--------|------|-----------|--------|
| 6622197 | report_ao_22cls | Native continuous | COMPLETED (training), test crashed (torchvision) |
| 6623467 | tok_vqvla_pre | VQ-VLA pretrained | FAILED (diffusers/transformers version conflict) |
| 6623468 | tok_vqvla_ft0 | VQ-VLA ft vanilla (λ=0) | FAILED (same diffusers issue) |
| 6623469 | tok_fastp | FAST+ pretrained (DROID) | **COMPLETED** |
| 6623470 | tok_fast_s1v256 | FAST s1/v256 (CALVIN) | **COMPLETED** |

Script: `scripts/submit_report_tables_v2.sh`

### Env issues
- **VQ-VLA**: `diffusers` 0.30.0 in conda mmml requires `transformers >=4.43` (for `EncoderDecoderCache`)
  but mmml has transformers 4.40.1. Also `torchvision` 0.17.0 is incompatible with torch 2.8.0.
  Fix: clone mmml → mmml_tok, install compatible torchvision + transformers. **Do not modify mmml.**
- **FAST s1/v256**: Old tokenizer files (fitted with `tokenizers<0.19`) incompatible with current
  `tokenizers` 0.22.2. Fixed by refitting: `checkpoints/fast_tokenizer_s1_v256_v2/`.
- **Native AO**: Training completed (30 epochs) but test phase crashed on `transforms.Compose()`.
  Checkpoint saved at `checkpoints/report_ao_22cls_j6622197_best.pth`. Eval pending.

## Results

### Attempt 1 (22 classes) — superseded

Initial runs used all 22 classes. Results below are for reference only; see 20-class results below.

| Tokenizer | Tok/traj | Val Acc (%) | Val MacF1 (%) | Best Ep | Job |
|-----------|----------|-------------|---------------|---------|-----|
| FAST+ pretrained | ~128 | 17.7 | 14.0 | 27 | 6623469 |
| FAST s1/v256 | ~27 | 24.2 | 20.8 | 30 | 6623470 |

### Attempt 2 (20 classes, `--min_class_count 30 --weighted_loss`) — current

| Tokenizer | Tok/traj | Val Acc (%) | Val MacF1 (%) | Best Ep | Active | Job |
|-----------|----------|-------------|---------------|---------|--------|-----|
| Native continuous | ~61 | TBD | TBD | TBD | — | pending |
| VQ-VLA pretrained | ~48 | TBD | TBD | — | — | 6625420 (running) |
| VQ-VLA ft vanilla | ~48 | TBD | TBD | — | — | 6625421 (running) |
| **FAST+ pretrained** | ~128 | **17.3** | **13.6** | 26 | 10/20 | 6624751 |
| **FAST s1/v256** | ~27 | **27.0** | **24.1** | 28 | 18/20 | 6624752 |

**Env notes**: VQ-VLA jobs use `mmml_tok` env (cloned mmml + upgraded torchvision/transformers).
FAST jobs use `mmml` env. Both set `PYTHONNOUSERSITE=1` and `HF_HUB_OFFLINE=1`.

**Eval note**: test_transformer.py needs `--max_seq_len 192` for FAST+ and
`--fast_tokenizer_path ./checkpoints/fast_tokenizer_s1_v256_v2` for refitted FAST s1/v256.

### Training convergence (20-class runs)

**FAST+ pretrained** (s10/v2048, DROID): Significant overfitting. Train acc climbs to 42.6% by ep 30
but val acc peaks at 17.3% (ep 26) and val loss increases after ep 5. The pretrained BPE vocabulary
(trained on DROID) does not transfer well to CALVIN — different action spaces mean different DCT
coefficient distributions, so the BPE merge rules are mismatched.

| Ep | Train Acc | Val Acc | Train Loss | Val Loss |
|----|-----------|---------|------------|----------|
| 1  | 7.3%  | 15.0% | 3.115 | 3.058 |
| 5  | 12.5% | 8.2%  | 2.723 | 2.738 |
| 10 | 16.2% | 11.3% | 2.371 | 2.702 |
| 15 | 21.0% | 15.7% | 2.001 | 2.872 |
| 20 | 28.7% | 16.8% | 1.670 | 2.925 |
| 25 | 36.0% | 15.4% | 1.321 | 3.129 |
| 30 | 38.7% | 17.1% | 1.244 | 3.128 |

**FAST s1/v256** (fitted on CALVIN): Also overfitting but better generalization than FAST+.
Train acc reaches 55.1% by ep 30, val acc peaks at 24.2% (ep 30 — still improving).
BPE vocab fitted on CALVIN data helps vs DROID-pretrained (+6.5pp acc).

| Ep | Train Acc | Val Acc | Train Loss | Val Loss |
|----|-----------|---------|------------|----------|
| 1  | 6.9%  | 12.9% | 3.110 | 3.029 |
| 5  | 17.0% | 21.4% | 2.498 | 2.479 |
| 10 | 23.6% | 21.1% | 2.055 | 2.285 |
| 15 | 30.7% | 17.7% | 1.590 | 2.322 |
| 20 | 42.7% | 22.0% | 1.110 | 2.476 |
| 25 | 51.6% | 23.9% | 0.808 | 2.626 |
| 30 | 55.1% | 24.2% | 0.724 | 2.686 |

### Per-class F1 breakdown (20 classes, 666 val)

**FAST+ pretrained** (17.3% acc / 13.6% MacF1):

| Verb | Support | Prec | Recall | F1 |
|------|---------|------|--------|-----|
| rotate | 57 | 42.4 | 43.9 | 43.1 |
| place | 35 | 39.0 | 45.7 | 42.1 |
| grasp | 61 | 22.1 | 49.2 | 30.5 |
| open | 9 | 27.3 | 33.3 | 30.0 |
| move | 19 | 15.0 | 31.6 | 20.3 |
| sweep | 26 | 14.3 | 19.2 | 16.4 |
| lift | 49 | 13.0 | 14.3 | 13.6 |
| take off | 13 | 11.1 | 15.4 | 12.9 |
| turn | 16 | 9.5 | 12.5 | 10.8 |
| slide | 81 | 14.6 | 7.4 | 9.8 |
| remove | 8 | 6.7 | 12.5 | 8.7 |
| turn off | 17 | 5.2 | 17.6 | 8.0 |
| pick up | 85 | 25.0 | 4.7 | 7.9 |
| put | 25 | 6.9 | 8.0 | 7.4 |
| turn on | 24 | 5.1 | 8.3 | 6.3 |
| stack | 13 | 3.6 | 7.7 | 4.9 |
| close | 9 | 0.0 | 0.0 | 0.0 |
| pull | 4 | 0.0 | 0.0 | 0.0 |
| push | 109 | 0.0 | 0.0 | 0.0 |
| store | 6 | 0.0 | 0.0 | 0.0 |

Dead classes (0% recall): close, pull, push, store (4/20).

**FAST s1/v256** (27.0% acc / 24.1% MacF1):

| Verb | Support | Prec | Recall | F1 |
|------|---------|------|--------|-----|
| place | 35 | 50.0 | 48.6 | 49.3 |
| rotate | 57 | 58.5 | 42.1 | 49.0 |
| grasp | 61 | 31.1 | 52.5 | 39.0 |
| put | 25 | 34.6 | 36.0 | 35.3 |
| close | 9 | 66.7 | 22.2 | 33.3 |
| turn off | 17 | 31.2 | 29.4 | 30.3 |
| move | 19 | 23.5 | 42.1 | 30.2 |
| turn | 16 | 20.5 | 50.0 | 29.1 |
| take off | 13 | 26.7 | 30.8 | 28.6 |
| sweep | 26 | 14.9 | 38.5 | 21.5 |
| stack | 13 | 20.0 | 23.1 | 21.4 |
| slide | 81 | 24.6 | 18.5 | 21.1 |
| lift | 49 | 15.3 | 26.5 | 19.4 |
| pick up | 85 | 26.1 | 14.1 | 18.3 |
| push | 109 | 37.1 | 11.9 | 18.1 |
| turn on | 24 | 18.8 | 12.5 | 15.0 |
| open | 9 | 12.5 | 11.1 | 11.8 |
| store | 6 | 9.1 | 16.7 | 11.8 |
| pull | 4 | 0.0 | 0.0 | 0.0 |
| remove | 8 | 0.0 | 0.0 | 0.0 |

Dead classes (0% recall): pull, remove (2/20). Much better class coverage than FAST+.

### Confusion matrices (20 classes)

**FAST+ pretrained** (`figures/tok_fastp_20cls_j6624751_best_cm.png`):
- Massive confusion across the board. `grasp` absorbs many other verbs (pick up → grasp: 30).
- `rotate` (25/57) and `place` (16/35) are the only moderately separated classes.
- `push` (109 samples) gets 0% recall — scattered across grasp, lift, place, slide, turn off/on.
- The DROID-pretrained BPE vocabulary doesn't capture CALVIN-specific motion distinctions.

**FAST s1/v256** (`figures/tok_fast_s1v256_20cls_j6624752_best_cm.png`):
- Better structure than FAST+. `rotate` (24/57), `place` (17/35), `grasp` (32/61) show clear diagonal.
- `push` partially recovered (13/109) but heavily confused with slide (22 misclassified as slide)
  and place — DCT quantization destroys fine-grained magnitude differences.
- `pick up` → `grasp` confusion persists (34/85 misclassified as grasp or lift).
- `turn` shows up well (8/16) due to distinctive rotational DCT coefficients.

**Key observation**: Both FAST variants struggle most with verbs that differ in **magnitude** rather
than **direction** of motion (push/slide, pick up/grasp). DCT quantization at any scale destroys
the continuous magnitude information needed to separate these pairs.

### Results files

20-class (current):
- `results/tok_fastp_20cls_j6624751_best_metrics.json`, `results/tok_fastp_20cls_j6624751_best_preds.json`
- `results/tok_fast_s1v256_20cls_j6624752_best_metrics.json`, `results/tok_fast_s1v256_20cls_j6624752_best_preds.json`
- Confusion matrices: `figures/tok_fastp_20cls_j6624751_best_cm.png`, `figures/tok_fast_s1v256_20cls_j6624752_best_cm.png`
- Checkpoints: `checkpoints/tok_fastp_20cls_j6624751_best.pth`, `checkpoints/tok_fast_s1v256_20cls_j6624752_best.pth`

22-class (superseded):
- `results/tok_fastp_j6623469_best_metrics.json`, `results/tok_fast_s1v256_j6623470_best_metrics.json`
- `figures/tok_fastp_j6623469_best_cm.png`, `figures/tok_fast_s1v256_j6623470_best_cm.png`

## Table 1: Per-Class F1 — AO vs Scene (Paper Section 4)

Compares the action-only transformer (best motion dynamics classifier) with the
scene-obs sklearn MLP (best action outcome classifier) on the same 22 classes.

| Verb | Support | AO F1 | Scene F1 |
|------|---------|-------|----------|
| ... | | | |
| **Accuracy** | | | |
| **Macro F1** | | | |

AO data from `results/report_ao_22cls_j{JID}_best_metrics.json`.
Scene data from `results/scene_obs_mlp_metrics.json` (generated by `analysis/sklearn_scene_obs_preds.py`).

## Table 2: Tokenizer Comparison (Paper Section 4.1)

Shows accuracy and macro-F1 for each tokenizer (best per type).

| Tokenizer | Description | Acc (%) | MacF1 (%) |
|-----------|-------------|---------|-----------|
| Native (continuous) | Linear(7→d), ~61 tokens | | |
| VQ-VLA pretrained | 4-group RVQ, Open X-Emb | | |
| VQ-VLA ft vanilla | finetuned on CALVIN, λ=0 | | |
| FAST+ pretrained | DCT+BPE, DROID, s10/v2048 | | |
| FAST s1/v256 | DCT+BPE, CALVIN, s1/v256 | | |

## Notes

- **Bug fix**: `torchvision` import in `train_transformer.py` and `test_transformer.py`
  was causing `RuntimeError: operator torchvision::nms does not exist` on some cluster
  nodes due to torch/torchvision version mismatch. Fixed by making the import lazy
  (try/except) and only building image transforms for vision modalities. Action-only
  and oracle modalities now run without torchvision.
- Previous experiments (R1–R5) used 20 or 21 sparse classes; these 22-class results
  are not directly comparable to those older numbers.
- FAST+ pretrained uses max_seq_len=192 because its sequences average ~128 tokens.
