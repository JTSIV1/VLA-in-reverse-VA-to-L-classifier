# Round 3c: Oracle Privileged-State Baselines (scene_obs, robot_obs)

**Date**: 2026-02-27
**Goal**: Understand how much verb identity is grounded in the action trajectory vs. physical state changes, using privileged simulator state as oracle inputs.

## Motivation

Previous rounds established unimodal baselines from action trajectories (R1–R2) and vision (R3). But we haven't answered a fundamental question: **how much of a verb is determined by the change in physical state vs. the motion to achieve it?**

CALVIN provides privileged simulator state per timestep:
- **`scene_obs`** (24-d): sliding door joint, drawer joint, button, switch, lightbulb, green light, red/blue/pink block poses (x,y,z,euler×3 each)
- **`robot_obs`** (15-d): TCP position (3), TCP orientation (3), gripper width (1), arm joint angles (7), gripper action (1)

These are oracle signals — unavailable on a real robot — but useful for analysis.

### Hypotheses
1. **scene_obs** should be a strong oracle for state-change verbs (open/close, turn on/off, slide) since it directly encodes the affected objects
2. **robot_obs** should correlate with action trajectories since it describes the robot's motion, but from the state perspective rather than delta-action perspective
3. 8-frame sampling should outperform 2-frame (first+last) by capturing intermediate dynamics

## Method

### Part 1: Transformer baselines

Oracle obs vectors are treated identically to native action tokens: `nn.Linear(obs_dim, d_model)` projects each timestep's vector to a transformer token. The sequence is simply `[CLS] + [obs_tokens]` (3 or 9 tokens for 2f/8f).

Same architecture as all previous rounds: 4-layer transformer, d_model=128, 8 heads, dropout=0.1. All use sp+wt recipe (21 sparse classes, weighted CE).

| Tag | Modality | Frames | Input dim | Tokens |
|-----|----------|--------|-----------|--------|
| scene_obs_2f j6463254 | scene_obs | 2 (first+last) | 24 | 3 |
| scene_obs_8f j6463255 | scene_obs | 8 (uniform) | 24 | 9 |
| robot_obs_2f j6463256 | robot_obs | 2 (first+last) | 15 | 3 |
| robot_obs_8f j6463257 | robot_obs | 8 (uniform) | 15 | 9 |

### Part 2: Simple classifier baselines (sklearn)

To verify the transformer results aren't an architecture bottleneck, we also trained simple classifiers (LogReg, Random Forest, GBM, MLP) directly on first→last obs deltas and concatenated [first, last] vectors. All use balanced class weights and StandardScaler. No sequence modeling — just a flat feature vector per episode.

Feature variants tested:
- **scene_delta** (24-d): `last_scene_obs - first_scene_obs`
- **scene_concat** (48-d): `[first_scene_obs, last_scene_obs]`
- **scene_engineered** (96-d): delta + |delta| + sign(delta) + binary(|delta|>0.01)
- **scene_fixtures_only** (6-d): door, drawer, button, switch, light, green_light dims only
- **scene_xyz_only** (9-d): block xyz positions only (no euler angles)
- **robot_delta** (15-d): `last_robot_obs - first_robot_obs`
- **robot_concat** (30-d): `[first_robot_obs, last_robot_obs]`

## Results

### Part 1: Transformer results (best val checkpoint)

| Tag | BestAcc | BestMF1 | Active | BestEp |
|-----|---------|---------|--------|--------|
| scene_obs_2f j6463254 | 8.0% | 1.3% | 3/21 | 8 |
| scene_obs_8f j6463255 | 12.2% | 2.3% | 4/21 | 6 |
| robot_obs_2f j6463256 | 34.0% | 26.2% | 16/21 | 25 |
| robot_obs_8f j6463257 | 38.5% | 34.9% | 18/21 | 24 |

Training curves show scene_obs **never learns** (underfitting, loss barely decreases), while robot_obs shows mild overfitting with clear learning.

### Part 2: Simple classifier results (Acc% / MacF1%)

| Feature | dim | LogReg | RF | GBM | MLP | Best |
|---------|-----|--------|-----|-----|-----|------|
| **scene_delta** | 24 | 31.4/33.5 | **48.4/38.9** | 44.4/38.0 | 40.6/34.8 | 48.4% (RF) |
| scene_concat | 48 | 27.6/28.7 | 40.9/28.8 | 40.1/32.8 | 28.5/24.6 | 40.9% (RF) |
| scene_engineered | 96 | 36.6/36.5 | 47.3/37.4 | 45.4/42.3 | 40.9/37.8 | 47.3% (RF) |
| scene_fixtures_only | 6 | 13.3/19.1 | 22.6/25.0 | 24.3/22.3 | 24.3/19.6 | 24.3% (GBM) |
| scene_xyz_only | 9 | 23.6/21.4 | 39.9/31.1 | 37.8/27.7 | 40.8/29.7 | 40.8% (MLP) |
| **robot_delta** | 15 | 19.5/20.1 | 35.9/25.7 | 34.9/26.0 | 32.6/28.7 | 35.9% (RF) |
| **robot_concat** | 30 | 30.4/32.1 | **44.4/37.3** | 40.6/33.4 | 39.3/36.6 | 44.4% (RF) |

### Per-class F1: GBM on scene_engineered (best MacF1 = 42.3%)

| Verb | F1 | Support | | Verb | F1 | Support |
|------|----|---------|---|------|----|---------|
| turn on | 89.8% | 24 | | place | 41.5% | 31 |
| rotate | 80.3% | 64 | | remove | 40.0% | 8 |
| turn off | 78.9% | 17 | | stack | 36.4% | 13 |
| open | 72.7% | 7 | | close | 33.3% | 7 |
| pick up | 48.0% | 77 | | slide | 29.3% | 64 |
| push | 43.9% | 88 | | put | 26.4% | 25 |
| take off | 46.2% | 8 | | lift | 20.0% | 32 |
| grasp | 41.4% | 54 | | turn | 8.3% | 16 |
| move | 40.0% | 19 | | lift up | 0.0% | 11 |
| store | 40.0% | 6 | | sweep | 0.0% | 26 |
| pull | 72.7% | 4 | | | | |

### Comparison with all unimodal baselines (sp+wt, best model per modality)

| Model | Architecture | BestAcc | BestMF1 | Active |
|-------|-------------|---------|---------|--------|
| MM VC-1 late2 sp+wt | Transformer | 42.4% | 40.7% | 20/21 |
| **scene_delta RF** | **Random Forest** | **48.4%** | **38.9%** | — |
| **scene_engineered GBM** | **GBM** | **45.4%** | **42.3%** | — |
| **robot_concat RF** | **Random Forest** | **44.4%** | **37.3%** | — |
| AO native sp+wt | Transformer | 39.5% | 38.7% | 21/21 |
| VO VC-1 delta16 sp+wt | Transformer | 38.9% | 36.2% | 20/21 |
| robot_obs_8f sp+wt | Transformer | 38.5% | 34.9% | 18/21 |
| robot_obs_2f sp+wt | Transformer | 34.0% | 26.2% | 16/21 |
| scene_obs_8f sp+wt | Transformer | 12.2% | 2.3% | 4/21 |
| scene_obs_2f sp+wt | Transformer | 8.0% | 1.3% | 3/21 |

## Analysis

### The transformer was the bottleneck, not the data

The most striking result is the **36pp gap** between scene_obs with a transformer (12%) vs. Random Forest (48%). The transformer completely fails on this input while a simple RF succeeds. This means our initial conclusion — that scene_obs lacks discriminative signal — was wrong.

Why does the transformer fail here?
- **Too few tokens**: 2–8 tokens of 24-d vectors gives the self-attention mechanism almost nothing to work with. The sequence is too short for positional patterns to emerge.
- **Wrong inductive bias**: Transformers excel at learning relationships between tokens in a sequence. But scene_obs deltas are a fixed-length feature vector — there's no sequential structure. Tree-based models (RF, GBM) are better suited for learning axis-aligned decision boundaries on tabular data (e.g., "if drawer_delta < -0.1 → close").

### Both motion kinematics and action outcomes carry verb information

With the right model, **scene_obs (action outcome) achieves 48.4% — the highest accuracy of any single modality.** This overturns the initial hypothesis that verbs are primarily grounded in motion.

| Signal type | Best model | Acc | MacF1 |
|-------------|-----------|-----|-------|
| Action outcome (scene_delta, RF) | RF | 48.4% | 38.9% |
| Action outcome (scene_eng, GBM) | GBM | 45.4% | 42.3% |
| Robot motion state (robot_concat, RF) | RF | 44.4% | 37.3% |
| Motion kinematics (AO native, Transformer) | Transformer | 39.5% | 38.7% |
| Vision (VC-1 delta16, Transformer) | Transformer | 38.9% | 36.2% |

### What scene_obs captures that motion doesn't (and vice versa)

**Scene_obs strengths** (fixture verbs with unique state signatures):
- turn on/off: 89.8% / 78.9% F1 — light state is a binary oracle
- open/close/pull: 72.7% / 33.3% / 72.7% — drawer joint is unambiguous
- rotate: 80.3% — block euler changes are distinctive for rotation

**Scene_obs weaknesses** (motion-style verbs):
- sweep: 0.0% — no unique state signature
- lift up: 0.0% — indistinguishable from lift/pick up in final state
- turn: 8.3% — "turn the switch" looks like any switch manipulation in state space

These are exactly the verbs where **how** you move matters more than **what** changed.

### Cluster analysis: scene_obs clusters by object, not verb

t-SNE of scene_obs deltas shows clusters organized by **which object changed** (door, drawer, lights, red/blue/pink block), not by verb. The block-manipulation cluster contains 15+ verbs jumbled together. Pairwise Cohen's d analysis confirms:
- **Separable pairs** (d > 1.0): open/close (d=15.4 on drawer), turn on/off (d=2.0 on lights)
- **Inseparable pairs** (d < 0.5 on all dims): push/slide, push/rotate, grasp/pick up, place/put

The RF succeeds despite this because it can learn **nonlinear combinations** of features — e.g., "block z increased AND block euler changed little → pick up" vs "block z increased AND block euler changed a lot → rotate." The transformer with 2–8 tokens cannot learn these decision boundaries.

### Revised hypothesis: motion vs. outcome framing

The clean separation we sought between motion kinematics and action outcome is:
- **Motion kinematics**: rel_actions (7-d per timestep), robot_obs (15-d per timestep)
- **Action outcome**: scene_obs delta (24-d), endpoint vision frames

Key finding: **both signals are informative, but require matched architectures.** Sequential models (transformers) work well for motion kinematics (temporal patterns in action trajectories). Tabular models (RF/GBM) work well for action outcomes (axis-aligned decision boundaries on state deltas). The poor transformer results on scene_obs reflect an architecture mismatch, not a lack of signal.

## Key Takeaway

**Verb identity is grounded in both motion kinematics and action outcomes**, but different representations require different model families. The transformer architecture used throughout this project is well-suited for sequential action/vision data but poorly suited for low-dimensional state vectors. With the right model (Random Forest), scene_obs achieves 48.4% — the highest single-modality accuracy, surpassing even multimodal transformers.

This suggests that **the ceiling for verb classification may be significantly higher than 42%** if we combine action trajectories (via transformer) with scene_obs features (via tree-based models or by engineering them into the transformer input).
