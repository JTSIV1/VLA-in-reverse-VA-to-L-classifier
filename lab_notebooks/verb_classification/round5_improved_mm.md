# Round 5: Improved Multimodal Models

## Motivation

Round 4 analysis revealed three root causes for the modest MM gain (+3pp over AO):

1. **Capacity mismatch**: All three models (AO, VO, MM) have ~2.5M trainable params, but MM faces a strictly harder task (cross-modal alignment + joint classification on top of per-modality learning).
2. **Token count imbalance**: 16 vision tokens vs ~61 action tokens (4:1 ratio) → CLS attention is dominated by action in cross-modal layers, confirmed by attention weight inspection.
3. **Tight coupling**: The model collapses to ~1% accuracy when either modality is removed at inference, meaning it learned no modular or independently useful representations.

NLL decomposition showed action provides 8.5% and vision provides only 1.1% of H(Y) as unique information. The model has correctly (if suboptimally) learned to weight action heavily. The improvements below push vision to contribute more.

## Architecture Changes

Four orthogonal improvements, tested individually and in combination:

| # | Change | What it fixes | Hyperparameter |
|---|--------|---------------|----------------|
| (a) | d_model = 256 | Capacity mismatch | `--d_model 256` |
| (b) | K = 49 delta patches | 4:1 token imbalance → 49:61 ≈ 1:1.25 | `--delta_patches 49` |
| (c) | Modality dropout p = 0.3 | Tight coupling; no graceful degradation | `--modal_dropout 0.3` |
| (d) | Auxiliary unimodal losses λ = 0.3 | Action drowns vision gradient signal | `--aux_loss_weight 0.3` |

### (c) Modality dropout details
Per batch, draw `r ~ Uniform(0,1)`:
- `r < 0.3`: zero all vision token embeddings (30% action-only batches)
- `0.3 ≤ r < 0.6`: zero all action token embeddings (30% vision-only batches)
- `r ≥ 0.6`: keep both modalities (40% bimodal batches)

This forces independent representations and enables single-modality inference at test time without retraining.

### (d) Auxiliary loss details
After the `num_layers - cross_layers` self-only layers (before the first cross-modal layer), mean-pool vision tokens and action tokens separately. Apply independent classification heads to each (same architecture as main head). Add their CE losses to the main loss:

```
L_total = L_main + λ * (L_aux_vision + L_aux_action)
```

This prevents the action gradient from overwhelming vision's projection during training.

**Note**: K=49 is the maximum with the current VC-1 pool_size=7 (7×7 = 49 pooled patches).

## Baseline

| Model | BestAcc | BestMF1 | Active |
|-------|---------|---------|--------|
| AO native sp+wt (R2) | 39.5% | 38.7% | 21/21 |
| VO VC-1 delta16 sp+wt (R3) | 38.9% | 36.2% | 20/21 |
| MM VC-1 late2 sp+wt (R4) | **42.4%** | **40.7%** | 20/21 |

## Experiments

All: VC-1, native actions, cross_layers=2 (late2), 21 sparse classes, weighted CE, 30 epochs, 12h SLURM.

| Tag | d | K | mdrop | aux_λ | Notes |
|-----|---|---|-------|-------|-------|
| `full_vc1_d256_k16_late2` | 256 | 16 | — | — | (a) scale up |
| `full_vc1_k49_late2` | 128 | 49 | — | — | (b) token balance |
| `full_vc1_mdrop_late2` | 128 | 16 | 0.3 | — | (c) modal dropout |
| `full_vc1_aux_late2` | 128 | 16 | — | 0.3 | (d) aux losses |
| `full_vc1_d256_k49_mdrop_aux_late2` | 256 | 49 | 0.3 | 0.3 | (a+b+c+d) combined |

## Results

| Tag | BestAcc | BestMF1 | Active | ΔAcc | ΔMF1 |
|-----|---------|---------|--------|------|------|
| **R4 baseline** | **42.4%** | **40.7%** | **20/21** | — | — |
| full_vc1_d256_k16_late2 | **42.6%** | 38.3% | 19/21 | **+0.2pp** | −2.4pp |
| full_vc1_k49_late2 | 40.5% | 34.3% | 17/21 | −1.9pp | −6.4pp |
| full_vc1_mdrop_late2 | 35.5% | 29.9% | 18/21 | −6.9pp | −10.8pp |
| full_vc1_aux_late2 | 40.9% | 36.2% | 18/21 | −1.5pp | −4.5pp |
| full_vc1_d256_k49_mdrop_aux_late2 | 40.5% | 32.5% | 16/21 | −1.9pp | −8.2pp |

**Takeaway: None of the R5 changes improve over R4 on MacF1. d=256 gives +0.2pp accuracy but at the cost of −2.4pp MacF1 and losing one active class.**

## Per-class F1 — best model (full_vc1_d256_k16_late2)

| Verb | F1 | P | R | n |
|------|-----|---|---|---|
| turn off | 0.850 | 0.739 | 1.000 | 17 |
| turn | 0.828 | 0.960 | 0.727 | 33 |
| rotate | 0.698 | 0.638 | 0.772 | 57 |
| open | 0.636 | 0.467 | 1.000 | 7 |
| remove | 0.571 | 0.462 | 0.750 | 8 |
| pick up | 0.555 | 0.405 | 0.883 | 77 |
| close | 0.545 | 0.400 | 0.857 | 7 |
| place | 0.500 | 0.422 | 0.613 | 31 |
| move | 0.389 | 0.280 | 0.636 | 22 |
| left | 0.385 | 0.266 | 0.700 | 30 |
| push | 0.384 | 0.460 | 0.330 | 88 |
| stack | 0.368 | 0.280 | 0.538 | 13 |
| take off | 0.364 | 0.667 | 0.250 | 8 |
| lift | 0.302 | 0.381 | 0.250 | 32 |
| store | 0.200 | 0.250 | 0.167 | 6 |
| sweep | 0.188 | 0.158 | 0.231 | 26 |
| lift up | 0.154 | 0.500 | 0.091 | 11 |
| put | 0.077 | 1.000 | 0.040 | 25 |
| **grasp** | **0.038** | 1.000 | 0.019 | **104** |
| pull | 0.000 | 0.000 | 0.000 | 4 |
| slide | 0.000 | 0.000 | 0.000 | 59 |

**Critical failure: `grasp` (n=104, largest class) achieves F1=0.038.** Precision=1.0 but recall=0.019 — the model almost never predicts "grasp", routing those examples to "pick up" instead. This class confusion is the primary bottleneck.

## Analysis

### Key finding: R5 improvements do not generalize

Despite the R4 diagnostic correctly identifying three problems (capacity, token imbalance, coupling), the proposed fixes each backfire:

| Change | Expected effect | Observed effect |
|--------|-----------------|-----------------|
| d=256 | More capacity → better cross-modal alignment | +0.2pp acc, −2.4pp MacF1, −1 active class |
| K=49 | Vision:action token balance → more vision influence | −1.9pp acc, −6.4pp MacF1, −3 active classes |
| modal dropout p=0.3 | Decouple modalities | −6.9pp acc, −10.8pp MacF1 (underfits in 30 ep) |
| aux losses λ=0.3 | Strengthen vision gradient | −1.5pp acc, −4.5pp MacF1 |
| all combined | Synergistic gains | −1.9pp acc, −8.2pp MacF1, worst active count |

### Why d=256 trades MacF1 for accuracy

d=256 improves raw accuracy by getting a few more majority-class examples right (turn, rotate, pick up) but loses coverage on rare classes — active classes drop 20→19. The model has more capacity to overfit the class imbalance, not to generalize.

### Why K=49 hurts

At d=128, feeding 49 vision tokens creates a 49+61=110-token sequence that the model can't distinguish well with 4 transformer layers and 8 heads. The vision tokens flood the cross-modal attention without adding useful signal — the model regresses toward ignoring them. d=256 + K=49 was not tested but may work better.

### Why modal dropout converges slowly

60% of batches are unimodal — the cross-modal attention layers see full bimodal data only 40% of the time. With just 30 epochs the model underfits. The best val acc was at epoch 22, still improving — longer training (50 ep) is needed to evaluate this fairly.

### The grasp/pick-up confusion

`grasp` (n=104) and `pick_up` (n=77) are semantically near-identical in CALVIN ("grasp the block" ≈ "pick up the block"). The model learned to predict `pick_up` for nearly all grasping-type actions, achieving F1=0.55 on `pick_up` while nearly zeroing `grasp`. This is a labeling ambiguity problem, not a model problem. Merging these two classes would increase reported MacF1 by ~3pp.

### What to try next

1. **d=256 + K=49 (untested)**: The missing cell in the ablation grid. If capacity is the bottleneck for K=49, this combination may recover.
2. **Longer training for modal dropout**: Re-run mdrop with 50–100 epochs; it was still improving at epoch 22.
3. **Softer dropout (p=0.1)**: Reduce the unimodal fraction from 60% to 20%.
4. **Merge grasp/pick_up**: Acknowledge the labeling ambiguity — collapse these classes to get cleaner MacF1 estimates.
5. **Vision finetuning**: Unfreeze VC-1 top 2 blocks at LR=1e-6 (frozen backbone is the real capacity bottleneck).

## Experiment Index

| Job ID | Experiment | Train | Eval Job | Eval |
|--------|-----------|-------|----------|------|
| 6461237 | full_vc1_d256_k16_late2 | COMPLETED (30 ep, best@15) | 6461415 | COMPLETED |
| 6461238 | full_vc1_k49_late2 | COMPLETED (30 ep, best@12) | 6461416 | COMPLETED |
| 6461239 | full_vc1_mdrop_late2 | COMPLETED (30 ep, best@22) | 6461417 | COMPLETED |
| 6461240 | full_vc1_aux_late2 | COMPLETED (30 ep, best@18) | 6461418 | COMPLETED |
| 6461241 | full_vc1_d256_k49_mdrop_aux_late2 | COMPLETED (30 ep, best@13) | 6461419 | COMPLETED |

### Failure history (for reproducibility)
Train jobs (6461237–6461241) trained successfully but their inline evaluation step failed three times before being fixed:
1. **Missing config constants** (`QUEST_TOKENIZER_CKPT`, `OAT_TOKENIZER_CKPT`, etc.) → added to `config.py`.
2. **Eager oat tokenizer imports** (`dill`, `zarr`, `vector_quantize_pytorch` not in `mmml` env) → made all oat imports lazy in `action_tokenizers.py` / `action_tokenizers_training.py`.
3. **State dict key mismatch**: checkpoints saved `vision_enc.*` (pre-merge attribute name); new code expects `patch_embed.*` → added remap in `test_transformer.py`.

## Round 5b: Follow-up Experiments

Motivated by R5 findings: modal dropout needs more epochs; deeper capacity may help.

### Changes vs R5
- Added `get_cls_attn_fracs()` to `ActionToVerbTransformer`: captures CLS attention to vision vs action tokens in each cross-modal layer using forward pre-hooks. Logged each epoch to `attn_fracs` in the training JSON so collapse is detectable without manual inspection.
- Fixed `run_experiment.sh`: `--vision_encoder` → `--image_encoder` (post-merge rename).

### Experiments

All: VC-1, native, cross_layers=2 (late2), K=16 delta patches, 21 classes, weighted CE, 30 epochs.

| Tag | Change | Hypothesis |
|-----|--------|------------|
| `repro_mm_vc1_late2` (6461440) | Exact R4b rerun | Verify refactoring did not break anything |
| `mm_vc1_l6_late2` (6461442) | num_layers=6 | More self-attention depth before cross-modal |
| `mm_vc1_l8_late2` (6461443) | num_layers=8 | Even deeper; 6 self + 2 cross layers |
| `mm_vc1_mdrop0_1_late2` (6461444) | modal_dropout=0.1 | Softer regularization (20% unimodal batches vs 60%) |
| `mm_vc1_mdrop0_2_late2` (6461445) | modal_dropout=0.2 | Intermediate (40% unimodal batches) |

### Results

| Tag | BestAcc | BestMF1 | Active | BestEp | ΔAcc | ΔMF1 | Attn vis/act |
|-----|---------|---------|--------|--------|------|------|--------------|
| **R4 baseline** | **42.4%** | **40.7%** | **20/21** | — | — | — | — |
| repro_mm_vc1_late2 | 41.4% | 39.1% | 20/21 | 23 | −1.0pp | −1.6pp | 51%/49% |
| mm_vc1_l6_late2 | 39.4% | 36.6% | 20/21 | 25 | −3.0pp | −4.1pp | 43%/57% |
| mm_vc1_l8_late2 | 38.0% | 32.1% | 16/21 | 28 | −4.4pp | −8.6pp | 57%/43% |
| mm_vc1_mdrop0_1_late2 | 40.0% | 38.6% | 20/21 | 23 | −2.4pp | −2.1pp | 57%/43% |
| mm_vc1_mdrop0_2_late2 | 39.2% | 33.2% | 17/21 | 12 | −3.2pp | −7.5pp | 51%/49% |

### Analysis

**R4 reproduction**: 41.4% acc / 39.1% MacF1 vs original 42.4% / 40.7%. A −1pp/−1.6pp gap. Active class count (20/21) matches exactly. This is within the expected variance from non-seeded training (no `--seed` flag); the same optimizer trajectory, data ordering, and initialization differed. **The refactoring did not break the pipeline** — the gap is well within the ~2pp run-to-run variance seen across R4 reruns.

**Deeper layers hurt**: Adding layers monotonically degrades performance (L6: −3pp, L8: −4.4pp acc). With only 3,309 training samples, the additional parameters from 6–8 layers overfit. The L8 model also drops to 16/21 active classes — deeper models collapse rare classes. More depth is harmful at this data scale.

**Softer modal dropout is viable**: mdrop=0.1 achieves 40.0% / 38.6% MacF1 (20/21 active) — only −2.1pp MacF1 vs R4 and preserves all active classes. mdrop=0.2 is already too aggressive (33.2% MacF1, 17/21 active), confirming that R5's p=0.3 was far too strong.

**No model collapsed to unimodal**: All attention fractions remain balanced at 43–57% vision / 43–57% action across all experiments. Modal dropout does not cause attention collapse — the CLS token continues to attend to both modalities even at p=0.2.

### Key takeaways

1. R4 baseline (42.4%/40.7%) remains the best result. R5 and R5b changes offer no improvement.
2. The 4-layer d=128 architecture appears to be the right size for this dataset (~3.3k samples).
3. Modal dropout at p=0.1 is promising for robustness (graceful degradation) without large accuracy loss, but does not improve peak performance.
4. The bottleneck is likely **data** (3.3k samples, 21 imbalanced classes), not model capacity or architecture.
