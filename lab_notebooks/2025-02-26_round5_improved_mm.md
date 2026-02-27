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

*(To be filled in once jobs complete)*

| Tag | BestAcc | BestMF1 | Active | Δ vs R4 |
|-----|---------|---------|--------|---------|
| full_vc1_d256_k16_late2 | — | — | — | — |
| full_vc1_k49_late2 | — | — | — | — |
| full_vc1_mdrop_late2 | — | — | — | — |
| full_vc1_aux_late2 | — | — | — | — |
| full_vc1_d256_k49_mdrop_aux_late2 | — | — | — | — |

## Analysis

*(To be filled in once jobs complete)*

**Key questions**:
- Does d=256 or K=49 shift CLS attention toward vision? → re-run `analyze_multimodal.py` on best R5 checkpoint.
- Does modality dropout enable graceful degradation? → re-run `test_modality_ablation.py` on dropout checkpoint.
- Which improvement contributes most? → per-improvement accuracy delta table.

## Experiment Index

| Job ID | Experiment | Status |
|--------|-----------|--------|
| TBD | full_vc1_d256_k16_late2 | submitted |
| TBD | full_vc1_k49_late2 | submitted |
| TBD | full_vc1_mdrop_late2 | submitted |
| TBD | full_vc1_aux_late2 | submitted |
| TBD | full_vc1_d256_k49_mdrop_aux_late2 | submitted |
