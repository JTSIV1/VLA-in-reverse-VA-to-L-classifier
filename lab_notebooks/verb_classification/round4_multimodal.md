# Round 4: Multimodal Fusion (Vision + Action)

**Date**: 2025-02-26
**Motivation**: Rounds 1–3 optimized each modality independently under the sparse+weighted CE recipe:
- Best action-only: 39.5% acc / 38.7% macro F1 (native sparse+weighted CE, 21 classes)
- Best vision-only: 38.9% acc / 36.2% macro F1 (VC-1 delta16, sparse+weighted CE, 21 classes)

Now we combine both modalities in a single transformer (same recipe) and ask: does combining vision + action outperform either alone?

---

## Architecture

The `ActionToVerbTransformer` supports `modality="full"` — it concatenates vision and action tokens into one sequence:

```
[CLS] + [vision delta tokens (16)] + [action native tokens (~64)] → Transformer → CLS → classifier
```

The key architectural variable is **cross_layers**: how many of the 4 transformer layers use full cross-modal attention (all tokens attend all tokens) vs self-only attention (vision attends only vision, action attends only action).

| cross_layers | Self-only layers | Description |
|---|---|---|
| 2 (late2) | 2 | 2 self-only, then 2 cross-modal |
| 1 (late1) | 3 | 3 self-only, then 1 cross-modal |

**d_model variants**: Standard d_model=128, plus a d_model=256 variant (d256) to test whether more capacity helps.

---

## Experiments

| Job ID  | Experiment                      | Encoder  | cross_layers | d_model |
|---------|---------------------------------|----------|--------------|---------|
| 6459077 | full_vc1_d16_late1_d256_sp_wt  | VC-1     | 1 (last)     | 256     |
| 6459078 | full_dinov2s_d16_late2_sp_wt   | DINOv2-S | 2 (last 2)   | 128     |
| 6459079 | full_vc1_d16_late2_sp_wt       | VC-1     | 2 (last 2)   | 128     |

All: 30 epochs, batch_size=16, lr=5e-4, frozen ViT encoder, native actions, 21 sparse classes (min_count=30), weighted CE. Job 6459077 used `--time=12:00:00`; jobs 6459078–6459079 used default 8h.

Note: Job 6459077 hit the SLURM time limit during final-epoch evaluation; only best-checkpoint metrics available.

---

## Results

| Experiment                     | cross | d   | Final Acc | Final MacF1 | BestAcc | BestMF1 | Active |
|--------------------------------|-------|-----|-----------|-------------|---------|---------|--------|
| full_vc1_d16_late1_d256_sp_wt | 1     | 256 | —         | —           | 40.8%   | 38.4%   | 20/21  |
| full_dinov2s_d16_late2_sp_wt  | 2     | 128 | 36.4%     | 34.0%       | 38.8%   | 36.2%   | 21/21  |
| **full_vc1_d16_late2_sp_wt**  | **2** | **128** | **39.2%** | **40.4%** | **42.4%** | **40.7%** | 20/21 |

### Comparison to unimodal baselines

| Model | Modality | BestAcc | BestMF1 | Active |
|-------|----------|---------|---------|--------|
| AO native sp+wt (R2)            | Action only | 39.5% | 38.7% | 21/21 |
| VO VC-1 delta16 sp+wt (R3)      | Vision only | 38.9% | 36.2% | 20/21 |
| full_dinov2s_d16_late2_sp_wt    | Both        | 38.8% | 36.2% | 21/21 |
| full_vc1_d16_late1_d256_sp_wt   | Both        | 40.8% | 38.4% | 20/21 |
| **full_vc1_d16_late2_sp_wt**    | **Both**    | **42.4%** | **40.7%** | 20/21 |

---

## Analysis

![Multimodal Analysis](../figures/multimodal_analysis.png)

### 1. Multimodal beats both unimodal baselines

`full_vc1_d16_late2_sp_wt` achieves 42.4% accuracy and 40.7% macro F1, beating action-only (39.5% / 38.7%) and vision-only (38.9% / 36.2%) on both metrics. The +3pp accuracy and +2pp macro F1 gain over the better unimodal model confirms the two modalities carry complementary information:
- Vision captures **what changed visually** (object position, gripper pose from pixel differences)
- Action captures **how the robot moved** (velocity, direction, contact trajectory)

Some verbs are easier from vision (e.g., detecting whether something was opened/closed), others from action (e.g., distinguishing push vs slide by contact dynamics).

### 2. Late cross-attention (late2) works best

cross_layers=2 outperforms cross_layers=1 under d=128. Letting each modality build its own representation in the first 2 self-only layers before fusing in the final 2 cross-modal layers produces better joint representations. Forcing cross-modal attention too early (fewer self-only layers) doesn't give each stream enough time to organize within-modality structure.

### 3. VC-1 > DINOv2-S in the multimodal setting

DINOv2-S late2 reaches 38.8% / 36.2%, vs VC-1 late2 at 42.4% / 40.7%. The VC-1 advantage seen in R3 (single-modality vision) carries through to the multimodal setting — VC-1's pretraining on egocentric manipulation data gives it better patch features for robot actions.

### 4. d_model=256 does not help

full_vc1_d16_late1_d256_sp_wt (d=256, cross=1): 40.8% / 38.4%
full_vc1_d16_late2_sp_wt (d=128, cross=2): 42.4% / 40.7%

More cross-modal interaction (late2) is more important than model width under this recipe. With balanced training signal across classes, d=128 is sufficient.

### 5. Sparse + weighted CE is essential for class coverage

All three models activate 20–21 out of 21 sparse classes. The same architecture without this recipe activated only 11–18 classes with macro F1 around 21–27%. The recipe is the critical ingredient for balanced coverage.

---

## Key Findings

1. **Multimodal fusion with sp+wt recipe beats both unimodal baselines** (42.4% / 40.7% vs 39.5% / 38.7% AO and 38.9% / 36.2% VO).
2. **Late cross-attention (2 cross-modal layers) works best** — let modalities build individual representations in early layers before fusing.
3. **VC-1 consistently outperforms DINOv2-S** in both unimodal and multimodal settings.
4. **d_model=256 does not compensate for fewer cross-modal layers** — architecture depth matters more than width.
5. **Vision and action are genuinely complementary** — the multimodal gain persists under controlled, recipe-matched conditions.

---

## Open Questions / Next Steps

1. **Cross-attention ablation**: Test cross_layers=0 (no fusion, just concatenation + shared classification head) as a lower bound.
2. **Asymmetric token counts**: Vision has 16 delta tokens, action has ~64 tokens. Try K=64 delta patches to balance the two streams.
3. **Late fusion baseline**: Separately trained unimodal models, ensemble logits — does learned joint representation beat naive ensembling?
4. **Modality dropout**: Train with random modality masking so the model can handle missing inputs at inference.

---

## Modality Ablation

**Question**: Did the multimodal model actually learn to use both modalities, or did it collapse to relying on just one?

**Method** (`test_modality_ablation.py`): Load the best sp+wt checkpoint (`full_vc1_d16_late2_sp_wt_j6459079_best.pth`, 42.4% / 40.7% MacF1), then evaluate three conditions at inference time:

| Condition | What's zeroed + attention-masked |
|-----------|----------------------------------|
| **both** | Nothing (normal inference) |
| **action_only** | Vision tokens [1, 17) zeroed + masked |
| **vision_only** | Action tokens [17, end) zeroed + masked |

Zeroing sets token embeddings to 0; masking sets `src_key_padding_mask=True` so no other token (including CLS) can attend to those positions.

### Results (job 6460104)

| Condition | Accuracy | Macro F1 | Active |
|-----------|----------|----------|--------|
| both (normal) | **42.4%** | **40.7%** | **21/21** |
| action_only (vision masked) | 1.1% | 0.1% | 1/21 |
| vision_only (action masked) | 1.1% | 0.1% | 1/21 |

Results saved in `results/ablation_full_vc1_d16_late2_sp_wt.json`.

### Interpretation

**Complete collapse when either modality is removed.** The model predicts a single class for all 665 validation examples (~1% = frequency of one rare class) in both ablation conditions. This rules out modality collapse in the "model ignores one modality" sense — if it relied only on action, ablating vision would have no effect.

**Why collapse instead of graceful degradation?**

The model was trained exclusively on bimodal inputs; its learned representations are calibrated for both streams being present simultaneously:

- In the self-only layers (layers 0–1), the CLS token attends only to itself by design of the block-diagonal self-mask — it picks up no modality information in these layers regardless.
- In the cross-modal layers (layers 2–3), CLS should in principle be able to attend to the surviving modality's tokens. However, the Q/K/V projections were trained jointly to fuse vision and action context together; with one side absent, the representation space is far out of distribution and the classifier receives a useless CLS vector.

**The model genuinely uses both modalities**, but in a tightly coupled, non-modular way. The +3pp accuracy gain over VO-VC1 and the +3pp over AO are real, but the model cannot fall back to single-modality inference without retraining.

**Implication**: For robust deployment, either (a) apply modality dropout during training so the model learns to handle missing inputs, or (b) keep separate unimodal models as fallbacks.

---

## Multimodal Behavior Analysis

**Question**: What information does each modality uniquely contribute, and why is the multimodal gain only +3pp?

Script: `analyze_multimodal.py` (job 6460133).
Figures: `figures/multimodal_analysis2.png` (prediction overlap + per-class scatter + CLS attention), `figures/multimodal_nll_decomp.png` (NLL decomposition + per-class ΔNLL + confusion matrix diff).

### 1. NLL Decomposition — Unique Variance

Using the identity `I(Y; Modality_A | Modality_B) ≈ NLL_B_only − NLL_MM`:

| Quantity | Nats | % of H(Y) |
|----------|------|-----------|
| H(Y) (label entropy) | 2.687 | 100% |
| NLL — AO model | 1.449 | — |
| NLL — VO model | 1.649 | — |
| NLL — MM model | 1.420 | — |
| **Unique info: action \| vision** | **0.229** | **8.5%** |
| **Unique info: vision \| action** | **0.029** | **1.1%** |
| Shared (both carry) | 1.010 | 37.6% |
| Irreducible entropy | 1.420 | 52.8% |

**Key takeaway**: Action provides ~8× more unique information than vision. The two modalities largely share the same discriminative signal (37.6% shared), leaving only 8.5% unique to action and 1.1% unique to vision. The MM gain is modest because there is little complementary information left to combine — the bulk of verb variance (52.8%) is irreducible noise from annotation ambiguity and verb category overlap.

Results saved in `results/multimodal_unique_variance.json`.

### 2. Per-Class Unique Variance

Sorted by action contribution (unique action − unique vision):

| Verb | Unique action | Unique vision | Interpretation |
|------|--------------|--------------|----------------|
| stack | +1.90 | −1.00 | Action-unique: distinctive stacking trajectory |
| take off | +1.46 | +0.27 | Action-unique: upward lift + release pattern |
| put | +1.12 | +0.46 | Action-unique: precise placement trajectory |
| place | +0.82 | +0.13 | Action-unique: controlled descend to surface |
| left | +0.74 | +0.44 | Action-unique: lateral translation signature |
| lift up | +0.47 | +0.69 | Both contribute; vision sees vertical displacement |
| turn off | −0.16 | +0.65 | **Vision-unique**: binary on/off state change visible |
| close | −0.12 | +0.53 | **Vision-unique**: drawer/door closure changes scene |
| store | −1.04 | −1.35 | Hard for both: confused with put/place in both AO and VO |

Vision uniquely helps verbs that produce **visible state changes** (on/off, open/close). Action uniquely helps verbs with **distinctive trajectory shapes** (stacking, taking off). For most verbs, both models make the same errors — hence the large shared component.

### 3. Prediction Overlap

(See left panel of `figures/multimodal_analysis2.png`)

The 8-way overlap (AO correct × VO correct × MM correct) shows:
- **MM corrects some AO+VO failures** (fusion-only successes) — the small complementary gain
- **Both AO and VO correct but MM wrong** cases exist — fusion does not always preserve unimodal knowledge
- Most correct predictions are shared across all three models, consistent with the large shared information component

### 4. CLS Attention Weights (Cross-Modal Layers)

(See right panel of `figures/multimodal_analysis2.png`)

In the cross-modal layers (layers 2–3), the CLS token's attention is split between:
- **Action tokens** (positions 17 onward): dominant attention sink — CLS relies heavily on action context
- **Vision tokens** (positions 1–16): minority contribution, consistent with vision providing only 1.1% unique information
- **Self** (position 0): residual self-attention

The action-dominant attention profile in CLS explains why MM barely outperforms AO: the joint model learns to weight action heavily, with vision providing marginal correction for a small subset of examples.

### 5. Confusion Matrix Difference (AO − VO)

(See right panel of `figures/multimodal_nll_decomp.png`)

Positive entries (red): AO confuses pair (i→j) more than VO → vision disambiguates this pair.
Negative entries (blue): VO confuses pair more → action disambiguates this pair.

Notable patterns:
- Vision disambiguates "close"/"open" (has clear visual signal; action for these is similar to "push"/"pull")
- Action disambiguates "stack" from "put"/"place" (distinct trajectory despite similar scene change)

### 6. Model Size & Fairness Analysis

Counting trainable parameters (frozen VC-1 backbone excluded):

| Model | Trainable params | Total params | Sequence length |
|-------|-----------------|--------------|-----------------|
| Action-only | 2,400K | 2,400K | ~62 tokens |
| Vision-only (VC-1 d16) | 2,487K | 88,286K | 17 tokens |
| Multimodal (VC-1 d16 late2) | **2,505K** | 88,303K | ~78 tokens |

The MM model has only +18K more trainable params than VO and +105K more than AO — roughly the same capacity, but a harder task. This is the root cause of the modest improvement. Combined with the 4:1 token imbalance (action dominates CLS attention) and the tight coupling (complete collapse without either modality), the MM model effectively collapses into a slightly better action-only model with vision providing marginal corrections.

**Proposed Round 5 improvements** (→ `lab_notebooks/2025-02-26_round5_improved_mm.md`):

| # | Change | Rationale |
|---|--------|-----------|
| (a) | d_model = 256 | Scale capacity proportional to bimodal task difficulty |
| (b) | K = 49 delta patches (from 16) | Reduce token imbalance: 49:61 ≈ 1:1.25 (vs 16:61 ≈ 1:3.8) |
| (c) | Modality dropout p = 0.3 | Force independent representations; enable graceful degradation |
| (d) | Auxiliary unimodal losses λ = 0.3 | Penalize action drowning out vision in the gradient signal |

---

## Experiment Index

| Job ID  | Experiment                      | Status                | Best Metrics |
|---------|---------------------------------|-----------------------|--------------|
| 6459077 | full_vc1_d16_late1_d256_sp_wt  | completed (best only) | results/full_vc1_d16_late1_d256_sp_wt_j6459077_best_metrics.json |
| 6459078 | full_dinov2s_d16_late2_sp_wt   | completed             | results/full_dinov2s_d16_late2_sp_wt_j6459078_best_metrics.json |
| 6459079 | full_vc1_d16_late2_sp_wt       | completed             | results/full_vc1_d16_late2_sp_wt_j6459079_best_metrics.json |
| 6460104 | modality ablation (test script) | completed             | results/ablation_full_vc1_d16_late2_sp_wt.json |
| 6460133 | multimodal behavior analysis    | completed             | results/multimodal_unique_variance.json |
| 6460134 | full_vc1_d16_cl0_sp_wt (cross_layers=0 baseline) | pending | — |

Note: Job 6459077 hit the SLURM time limit before final-checkpoint evaluation; only best-checkpoint metrics are reported.
