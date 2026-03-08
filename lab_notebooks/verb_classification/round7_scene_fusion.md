# Round 7: Scene Obs + Action Fusion

**Date**: 2026-02-27
**Motivation**: R3c showed scene_obs (24-d privileged simulator state) carries strong verb signal
(RF: 48.4% acc), but transformers are a poor architecture for low-dim tabular data (12% acc).
R7 tests fusion architectures that combine scene_obs with action trajectories using
architecture-appropriate modules (MLPs) for the scene signal, rather than forcing it through
the transformer as a sequence.

**Comparison goal**: action+scene_obs multimodal vs action+vision multimodal — does cheap
privileged state match or beat expensive vision encoders?

## Design

**Scene representation**: delta_start (48-d) = `[scene_obs[start], scene_obs[end] - scene_obs[start]]`
- Captures both initial state and state change from the episode endpoints.

**Fusion strategies**:
| Strategy | Modality | How it works |
|----------|----------|-------------|
| Token injection | `scene_token` | scene_obs → MLP → 1 token prepended to action sequence in transformer |
| Concat branches | `scene_concat` | Separate MLP for scene + transformer for actions → concat → classifier |
| FiLM conditioning | `scene_film` | scene_obs → MLP → (gamma, beta) that modulate action embeddings before transformer |

**Action representations**: native (continuous, ~61 tokens), VQ-VLA (discrete, 48 tokens), FAST s1/v256 (discrete, ~25 tokens)

**Standing convention**: --min_class_count 30 --weighted_loss (21 classes)

**Label set note**: R7 uses cleaned labels (commit 45c1002): directional words removed,
two-verb sentences excluded, "turn on"/"turn off" kept separate. This yields a different
21-class set than R1–R5 (replaces "left" with "turn on") and 601 val samples vs 665.
20 of 21 classes overlap; aggregate metrics are approximately comparable.

## Experiment Grid (9 jobs)

| # | Fusion | Action Rep | Name | BestAcc | BestMF1 | Active |
|---|--------|------------|------|---------|---------|--------|
| 1 | token | native | r7_token_native | **43.1%** | **41.0%** | **21/21** |
| 2 | token | vqvla | r7_token_vqvla | 39.3% | 32.6% | 19/21 |
| 3 | token | fast | r7_token_fast | 22.8% | 19.7% | 18/21 |
| 4 | concat | native | r7_concat_native | **43.1%** | 37.9% | 20/21 |
| 5 | concat | vqvla | r7_concat_vqvla | **43.1%** | 36.3% | 19/21 |
| 6 | concat | fast | r7_concat_fast | 32.1% | 29.4% | 20/21 |
| 7 | film | native | r7_film_native | 42.8% | 37.8% | 19/21 |
| 8 | film | vqvla | r7_film_vqvla | 39.6% | 31.5% | 18/21 |
| 9 | film | fast | r7_film_fast | 33.1% | 33.1% | 20/21 |

## Baselines for Comparison

Baselines from earlier rounds (R1–R5) use old label set (665 val, "left" class); approximate.

| Model | BestAcc | BestMF1 | Active | Notes |
|-------|---------|---------|--------|-------|
| AO native sp+wt | 39.5% | 38.7% | 21/21 | Action-only transformer (R2, old labels) |
| VO VC-1 delta16 sp+wt | 38.9% | 36.2% | 20/21 | Vision-only (R3, old labels) |
| VO DINOv2-S delta16 sp+wt | 38.0% | 34.0% | 19/21 | Vision-only (R3, old labels) |
| MM VC-1 late2 sp+wt | 42.4% | 40.7% | 20/21 | Best vision MM (R4b, old labels) |
| scene_obs RF | 48.4% | — | — | Oracle upper bound (R3c, sklearn) |

**Same-label-set unimodal baselines** (new labels, 601 val, for fair modality decomposition):

| Model | BestAcc | BestMF1 | Active | Notes |
|-------|---------|---------|--------|-------|
| AO native (new labels) | **44.8%** | 39.1% | 20/21 | Action-only transformer |
| Scene MLP (new labels) | 23.1% | 24.2% | 20/21 | scene_obs delta_start → MLP → classifier |

## Results

### Fusion strategy comparison (native actions only)

| Fusion | Acc | MacF1 | Active | Dead classes |
|--------|-----|-------|--------|-------------|
| Token injection | **43.1%** | **41.0%** | **21/21** | (none) |
| Concat branches | 43.1% | 37.9% | 20/21 | open |
| FiLM conditioning | 42.8% | 37.8% | 19/21 | open, remove |

Token injection is the clear winner: same accuracy as concat but +3.1pp MacF1 and all 21
classes active. Concat and FiLM achieve similar accuracy but worse class coverage — the
scene MLP branch may overfit to majority classes when it bypasses the transformer entirely
(concat) or dominates the action signal (FiLM).

### Action representation comparison

| Action Rep | Token Acc/MF1 | Concat Acc/MF1 | Film Acc/MF1 |
|------------|---------------|----------------|--------------|
| Native | **43.1 / 41.0** | 43.1 / 37.9 | 42.8 / 37.8 |
| VQ-VLA | 39.3 / 32.6 | **43.1 / 36.3** | 39.6 / 31.5 |
| FAST s1/v256 | 22.8 / 19.7 | 32.1 / 29.4 | 33.1 / 33.1 |

Native actions dominate on MacF1 across all fusion strategies. Notably, **concat + VQ-VLA
matches native accuracy** (43.1%) — the concat architecture compensates for VQ-VLA's lossy
tokenization by giving the scene MLP branch an independent path to the classifier. However,
MacF1 remains lower (36.3% vs 41.0%) indicating worse rare-class coverage.

FAST loses ~10–20pp on accuracy, consistent with prior rounds where DCT quantization
destroys discriminative signal. VQ-VLA is intermediate (39–43% acc) but always trails
native on MacF1 (−5 to −8pp).

### Scene-obs MM vs Vision MM vs unimodal

| Model | Type | Observation | Acc | MacF1 | Active |
|-------|------|-------------|-----|-------|--------|
| AO native sp+wt | Unimodal | actions only | 39.5% | 38.7% | 21/21 |
| VO VC-1 delta16 sp+wt | Unimodal | vision only | 38.9% | 36.2% | 20/21 |
| VO DINOv2-S delta16 sp+wt | Unimodal | vision only | 38.0% | 34.0% | 19/21 |
| MM VC-1 late2 sp+wt | Multimodal | action + vision | 42.4% | 40.7% | 20/21 |
| **R7 token + native** | **Multimodal** | **action + scene_obs** | **43.1%** | **41.0%** | **21/21** |
| scene_obs RF | Oracle | scene_obs only | 48.4% | — | — |

**Key finding**: Scene-obs token fusion (43.1% / 41.0%) slightly outperforms the best
vision MM model (42.4% / 40.7%) while being far simpler — no vision encoder, no image
loading, no patch embedding. A 48-d MLP replaces a frozen ViT + 196 patch tokens.

Gap to oracle RF: 43.1% vs 48.4% (−5.3pp). The transformer still underperforms RF on
scene_obs, but fusion with actions closes ~60% of the gap vs scene_obs-only transformer (12%).

### Per-class F1 comparison (top model from each paradigm)

Showing R7 Scene Token (best scene MM), MM VC-1 (best vision MM), and AO Native (action-only).
Classes sorted by R7 Scene Token F1. 20 overlapping classes shown.

| Class | Scene Token | MM VC-1 | AO Native | Winner |
|-------|-------------|---------|-----------|--------|
| rotate | 82.6 | 75.7 | 75.6 | Scene |
| turn off | 66.7 | 81.1 | 53.7 | Vision |
| grasp | **61.3** | 8.4 | 22.7 | **Scene** |
| place | 58.0 | 54.5 | 50.9 | Scene |
| close | 57.1 | 58.3 | 52.2 | Vision |
| pull | 50.0 | 50.0 | 47.1 | Tie |
| take off | 47.1 | 52.6 | 47.6 | Vision |
| remove | 46.2 | 33.3 | 50.0 | AO |
| move | 44.0 | 40.6 | 33.3 | Scene |
| open | 40.0 | 42.9 | 44.4 | AO |
| sweep | 31.7 | 29.0 | 25.8 | Scene |
| stack | 30.8 | 41.4 | 33.3 | Vision |
| lift | 27.6 | 0.0 | 27.6 | Scene/AO |
| slide | 23.0 | 26.0 | 28.6 | AO |
| pick up | 22.0 | **53.9** | **53.1** | **Vision** |
| lift up | 20.7 | 17.1 | 8.7 | Scene |
| put | 19.0 | 6.7 | 17.1 | Scene |
| push | 11.9 | 30.8 | 17.5 | Vision |
| turn | 8.0 | 75.9 | 56.1 | Vision |

(R7 also has "turn on" at 72.7% F1, not in baseline class set.)

### Per-class patterns

**Scene-obs excels at**: grasp (61.3 vs 8.4), rotate (82.6 vs 75.7), move (44.0 vs 40.6),
lift up (20.7 vs 17.1), put (19.0 vs 6.7). These are verbs where the object state change
(drawer position, block location) is more informative than the visual appearance.

**Vision MM excels at**: pick up (53.9 vs 22.0), turn (75.9 vs 8.0), turn off (81.1 vs 66.7),
push (30.8 vs 11.9), stack (41.4 vs 30.8). These verbs benefit from visual context
(which object? which direction?) that scene_obs doesn't fully capture.

**grasp vs pick up split**: Scene Token gets 61.3% on grasp but only 22.0% on pick up.
Vision MM shows the opposite (8.4% grasp, 53.9% pick up). These verbs are semantically
similar (both involve hand closing on object); scene_obs captures the grip state change
while vision captures the broader context of the manipulation.

## Modality Contribution Analysis

To understand how much information comes from action trajectories vs scene_obs, and whether
the fusion model captures complementary signal, we ran sample-level prediction agreement
analysis across three models trained on the same label set (601 val, 21 classes, weighted CE):

- **Action-only** (r7_ao_native): 44.8% acc
- **Scene MLP-only** (r7_scene_mlp): 23.1% acc
- **Fusion** (r7_token_native): 43.1% acc

### Method

For each of the 601 validation samples, we check whether each unimodal model predicts
correctly, yielding four quadrants. We then measure how often the fusion model is correct
in each quadrant to reveal how it uses each modality's signal.

Script: `analyze_modality_contribution.py`

### Sample-level agreement

| Quadrant | Count | % of total | Fusion correct | Fusion rate |
|----------|------:|----------:|--------------:|------------:|
| Both unimodal correct | 91 | 15.1% | 76 | 83.5% |
| Only action correct | 178 | 29.6% | 90 | 50.6% |
| Only scene correct | 48 | 8.0% | 25 | 52.1% |
| Neither correct | 284 | 47.3% | 68 | 23.9% |

- **Union unimodal accuracy**: 52.7% (317/601 where at least one unimodal is correct)
- **Fusion complementary gains**: 68/284 samples (11.3% of total) where neither unimodal
  model is correct but fusion is — genuine complementary signal from combining modalities
- **Fusion regressions**: 15/91 samples where both unimodals are correct but fusion is wrong

### Per-class breakdown

| Class | N | A% | S% | F% | Both | OnlyA | OnlyS | Neither | F\|Neither |
|-------|--:|---:|---:|---:|-----:|------:|------:|--------:|-----------:|
| rotate | 64 | 100 | 60.9 | 89.1 | 39 | 25 | 0 | 0 | 0 |
| grasp | 54 | 48.1 | 27.8 | **90.7** | 7 | 19 | 8 | 20 | **17** |
| place | 31 | 9.7 | 3.2 | **64.5** | 0 | 3 | 1 | 27 | **17** |
| sweep | 26 | 15.4 | 26.9 | **61.5** | 2 | 2 | 5 | 17 | **11** |
| turn on | 24 | 50.0 | 95.8 | 83.3 | 11 | 1 | 12 | 0 | 0 |
| move | 19 | 73.7 | 5.3 | 57.9 | 1 | 13 | 0 | 5 | 1 |
| turn off | 17 | 100 | 82.4 | 82.4 | 14 | 3 | 0 | 0 | 0 |
| push | 88 | 21.6 | 8.0 | 6.8 | 0 | 19 | 7 | 62 | 4 |
| pick up | 77 | 42.9 | 3.9 | 14.3 | 0 | 33 | 3 | 41 | 1 |
| slide | 64 | 35.9 | 1.6 | 20.3 | 0 | 23 | 1 | 40 | 8 |
| lift | 32 | 37.5 | 12.5 | 25.0 | 2 | 10 | 2 | 18 | 0 |
| stack | 13 | 30.8 | 7.7 | 30.8 | 0 | 4 | 1 | 8 | 1 |
| lift up | 11 | 27.3 | 18.2 | 27.3 | 1 | 2 | 1 | 7 | 0 |
| remove | 8 | 12.5 | 0.0 | 37.5 | 0 | 1 | 0 | 7 | 3 |
| take off | 8 | 87.5 | 25.0 | 50.0 | 2 | 5 | 0 | 1 | 0 |
| close | 7 | 100 | 57.1 | 85.7 | 4 | 3 | 0 | 0 | 0 |
| open | 7 | 28.6 | 71.4 | 28.6 | 1 | 1 | 4 | 1 | 0 |
| store | 6 | 100 | 33.3 | 66.7 | 2 | 4 | 0 | 0 | 0 |
| put | 25 | 44.0 | 20.0 | 16.0 | 4 | 7 | 1 | 13 | 3 |
| pull | 4 | 25.0 | 50.0 | 75.0 | 1 | 0 | 1 | 2 | 1 |
| turn | 16 | 0.0 | 6.2 | 6.2 | 0 | 0 | 1 | 15 | 1 |

### Interpretation

**Action trajectory dominates overall**: 29.6% of samples are correctly classified by
action-only but not scene-only (vs 8.0% for the reverse). Action trajectories carry the
primary verb signal for most manipulation verbs.

**Scene MLP is weak as a standalone classifier** (23.1%) but carries unique signal for
fixture verbs: turn on (95.8%), open (71.4%), turn off (82.4%). These verbs cause
distinctive state changes in the 24-d scene_obs (light switches, drawer positions) that
the MLP can detect even without action context.

**Fusion generates genuine complementary signal**: 68 samples (11.3%) are correctly
classified only by fusion, not by either unimodal model alone. The strongest complementary
gains are on:
- **grasp** (17/20 neither-samples rescued): the combination of gripper action pattern +
  state change uniquely identifies this verb
- **place** (17/27 rescued): action trajectory alone can't distinguish place from other
  positioning verbs; scene_obs alone can't either; but together the putting-down motion
  + resulting state change is diagnostic
- **sweep** (11/17 rescued): similar — the sweeping action + resulting object displacement
  together identify this verb

**Fusion has significant regressions on action-dominated verbs**: pick up (42.9% → 14.3%),
push (21.6% → 6.8%), put (44.0% → 16.0%). The scene token may introduce noise for verbs
where scene_obs is uninformative, distracting the transformer from the action signal.

**Overall**: actions provide the primary signal (29.6% unique), scene_obs provides a smaller
but non-redundant complement (8.0% unique), and their interaction creates additional
predictive power (11.3% complementary). However, fusion also causes regressions for
action-dominated verbs, explaining why fusion accuracy (43.1%) is slightly below action-only
(44.8%) despite having better MacF1 (41.0% vs 39.1%).

## Bug Fix

FAST test jobs initially crashed with `FileNotFoundError` during `load_action_tokenizer`.
Root cause: `test_transformer.py` line 121 had `"fast"` in the wrong tokenizer branch
(`action_tokenizers.load_action_tokenizer` which loads training episodes for normalization)
instead of the correct `fast_tokenizer.load_fast_tokenizer` branch at line 132 (dead code).
Fix: removed `"fast"` from line 121's condition so it falls through to the correct branch.

## Conclusions

1. **Scene-obs token fusion achieves the best MacF1** (41.0%, 21/21 classes active),
   beating MM VC-1 late2 (40.7%, 20/21) at a fraction of the compute cost. However,
   action-only on the same label set achieves higher accuracy (44.8% vs 43.1%).

2. **Token injection > Concat > FiLM** for scene fusion. Letting the transformer attend to
   scene info as a token works better than bypassing it (concat) or modulating it globally (FiLM).

3. **Action trajectories carry the primary verb signal** (29.6% uniquely correct samples),
   while scene_obs provides a smaller but non-redundant complement (8.0% unique), strongest
   for fixture verbs (turn on/off, open). Fusion generates 11.3% complementary gains where
   neither modality alone suffices.

4. **Fusion helps rare classes but hurts action-dominated verbs**: MacF1 improves (+1.9pp)
   because fusion rescues rare verbs (grasp, place, sweep), but accuracy drops (−1.7pp)
   because the scene token introduces noise for verbs where scene_obs is uninformative
   (pick up, push, put).

5. **Scene-obs and vision are complementary**, not redundant — each excels at different verb
   subsets. A future round could fuse all three (action + scene_obs + vision).

6. **Native actions remain essential for MacF1** — FAST and VQ-VLA tokenization both degrade
   MacF1 substantially. Exception: concat + VQ-VLA matches native accuracy (43.1%) by
   leveraging the independent scene MLP branch, but rare-class F1 still suffers.

7. **Gap to oracle**: 43.1% vs 48.4% RF. The remaining 5.3pp gap likely comes from
   (a) transformer vs RF on the scene signal, and (b) imperfect fusion with action tokens.
