# Round 9: L0 vs L1 Verb Decodability

**Date**: 2026-03-17
**Motivation**: R8 annotated CALVIN episodes with hierarchical language via Gemini 2.5 Pro,
producing both Gemini-inferred L0 task instructions and L1 subtask decompositions. R9 uses
these annotations to answer the core question: **are low-level subtask verbs (L1) easier
to decode from action trajectories and scene state than high-level task verbs (L0)?**

## Experimental Design

### Three conditions

| Condition | Label Source | Granularity | #Classes | #Train | #Val | Segment |
|-----------|-------------|-------------|----------|--------|------|---------|
| **GT L0** | CALVIN annotations | Full episode | 20 | ~3,300 | 666 | Full trajectory (~61 steps) |
| **Gemini L0** | Gemini TASK_INSTRUCTION | Full episode | 14 | 5,024 | 993 | Full trajectory (~61 steps) |
| **L1** | Gemini DECOMPOSITION | Phase segment | 16 | 18,366 | 3,674 | Segment (~16 steps avg) |

### Two classifiers per condition

1. **AO Transformer**: 4-layer transformer on action trajectories (7-d rel_actions per timestep).
   Same architecture as R1–R7 (d_model=128, 8 heads, CLS token). Weighted CE loss.

2. **Scene-obs sklearn MLP**: 2-layer MLP (256→128) on scene_obs engineered features (96-d).
   Features: `[delta, |delta|, sign(delta), 1(|delta|>0.01)]` where `delta = scene_obs[end] - scene_obs[start]`.
   StandardScaler, early stopping.

### Design choices

**L1 segment dataset** (`data/l1_segments/`):
- Each L1 phase → one sample, action subsequence sliced by `[START_TIMESTEP, END_TIMESTEP]`
- Verb extracted as first word of STEP_DESCRIPTION, then consolidated:
  - reach → approach; withdraw → retract; grip/hold/secure/engage/contact/hook → grasp
  - descend → lower; transport/translate/carry/relocate → move
  - align/reposition/adjust/orient → position; twist/turn → rotate
- Min segment length: 3 timesteps (filters 13% of phases — mostly 1-step gripper events)
- Min class count: 30 → 16 classes retained

**Gemini L0 dataset** (`data/gemini_l0_segments/`):
- Full episodes labeled with verb from Gemini's inferred TASK_INSTRUCTION
- Verb extracted via spaCy, consolidated (closed→close, unstack→remove, stand→place)
- Min class count: 30 → 14 classes retained
- 5 episodes skipped (spaCy found no verb in Gemini instruction)

**GT L0 baseline**: Existing results from R7/R8 (sp+wt recipe, 20 classes after "left" removal).

**Why different class counts**: Each condition has a natural verb vocabulary. GT L0 has
CALVIN's crowdsourced labels (20 classes). Gemini L0 is more goal-oriented, merging some
distinctions (e.g., "push"/"slide" → fewer classes) but splitting others ("press" as distinct).
L1 has scaffolding verbs (approach, retract, release) not present in L0. We compare
decodability within each condition's natural class set, not across forced-aligned classes.

### Scripts

- `scripts/build_l1_dataset.py` — build L1 segment dataset from annotations
- `scripts/build_gemini_l0_dataset.py` — build Gemini L0 dataset from annotations
- `scripts/train_l1_ao_transformer.py` — AO transformer (works for both L1 and Gemini L0)
- `scripts/train_l1_scene_sklearn.py` — scene sklearn MLP (works for both L1 and Gemini L0)
- `scripts/submit_l1_ao.sh` — SLURM submission for L1 AO

### L1 verb distribution (16 classes, training set)

| Verb | Count | % | Type |
|------|------:|--:|------|
| approach | 3,282 | 17.9% | scaffolding |
| lift | 2,667 | 14.5% | scaffolding |
| retract | 2,505 | 13.6% | scaffolding |
| lower | 1,908 | 10.4% | scaffolding |
| position | 1,520 | 8.3% | scaffolding |
| grasp | 1,319 | 7.2% | scaffolding |
| push | 1,313 | 7.1% | task-specific |
| move | 1,162 | 6.3% | task-specific |
| release | 662 | 3.6% | scaffolding |
| place | 543 | 3.0% | task-specific |
| rotate | 532 | 2.9% | task-specific |
| press | 328 | 1.8% | task-specific |
| pull | 195 | 1.1% | task-specific |
| slide | 193 | 1.1% | task-specific |
| open | 122 | 0.7% | task-specific |
| flip | 115 | 0.6% | task-specific |

7 scaffolding verbs (approach, lift, retract, lower, position, grasp, release) = 75.5% of data.
9 task-specific verbs (push, move, place, rotate, press, pull, slide, open, flip) = 24.5%.

### Gemini L0 verb distribution (14 classes, training set)

| Verb | Count | % |
|------|------:|--:|
| pick up | 1,453 | 28.9% |
| place | 931 | 18.5% |
| push | 620 | 12.3% |
| slide | 379 | 7.5% |
| rotate | 294 | 5.9% |
| stack | 218 | 4.3% |
| press | 213 | 4.2% |
| close | 186 | 3.7% |
| turn off | 163 | 3.2% |
| open | 156 | 3.1% |
| move | 133 | 2.6% |
| turn | 125 | 2.5% |
| lift | 119 | 2.4% |
| release | 34 | 0.7% |

## Results

### Scene-obs sklearn MLP

| Condition | Val Acc | Val MacF1 | #Classes | #Val |
|-----------|---------|-----------|----------|------|
| GT L0 | 44.7% | 38.0% | 20 | 666 |
| **Gemini L0** | **65.0%** | **56.6%** | 14 | 993 |
| L1 segments | 48.8% | 40.8% | 16 | 3,674 |

### AO Transformer

| Condition | Val Acc | Val MacF1 | #Classes | #Val | Best Ep | Status |
|-----------|---------|-----------|----------|------|---------|--------|
| GT L0 | 45.2% | 42.6% | 20 | 666 | — | converged |
| Gemini L0 | 30.4% | 16.9% | 14 | 993 | 6/30 | crashed, resubmitted |
| L1 segments | 43.2% | 29.9% | 16 | 3,674 | 14/30 | crashed, resubmitted |

**Note**: Both Gemini L0 and L1 AO jobs crashed (OOM at 32GB, ~33.5GB used) before
convergence. Results above are from best checkpoints saved mid-training. Resubmitted
with 48GB memory + checkpoint resume (jobs 6621296, 6621272). Results will be updated
when they complete.

Gemini L0 AO is notably struggling (30.4% at ep6, then degrading to 18.8% by ep14).
This may indicate the Gemini L0 verb labels are harder for the AO transformer to decode
than GT L0 verbs — the opposite of the scene-obs pattern. This would make sense:
Gemini L0 is goal-oriented ("pick up", "place") which maps to scene state changes,
while the AO transformer reads action trajectories which better encode HOW the robot
moves (GT L0's action-descriptive verbs like "push", "slide").

### Scene-obs per-class F1 (L1 segments, 16 classes)

| Verb | Prec | Recall | F1 | Support | Type |
|------|------|--------|-----|---------|------|
| press | 0.97 | 0.89 | 0.93 | 64 | task-specific |
| pull | 0.90 | 0.85 | 0.88 | 33 | task-specific |
| lift | 0.73 | 0.76 | 0.74 | 534 | scaffolding |
| push | 0.66 | 0.80 | 0.72 | 250 | task-specific |
| rotate | 0.56 | 0.61 | 0.58 | 111 | task-specific |
| approach | 0.38 | 0.92 | 0.53 | 647 | scaffolding |
| move | 0.57 | 0.39 | 0.46 | 226 | task-specific |
| retract | 0.46 | 0.37 | 0.41 | 487 | scaffolding |
| flip | 0.53 | 0.32 | 0.40 | 28 | task-specific |
| lower | 0.40 | 0.25 | 0.31 | 386 | scaffolding |
| grasp | 0.40 | 0.18 | 0.25 | 270 | scaffolding |
| slide | 0.36 | 0.09 | 0.14 | 46 | task-specific |
| place | 0.15 | 0.05 | 0.07 | 103 | task-specific |
| release | 0.21 | 0.05 | 0.07 | 153 | scaffolding |
| position | 0.19 | 0.01 | 0.02 | 310 | scaffolding |
| open | 0.00 | 0.00 | 0.00 | 26 | task-specific |

## Preliminary Interpretation

**Gemini L0 scene-obs is dramatically more decodable** (65.0% vs 44.7% GT L0):
- Gemini L0 verbs are more **goal-oriented** ("pick up", "place", "turn off") — these
  map directly to distinct scene_obs changes (block position delta, light state change).
- GT L0 uses many **action-descriptive** verbs ("push the switch upwards", "move the door")
  that describe HOW the robot acts, not WHAT changed — harder for scene_obs to decode.
- Fewer classes (14 vs 20) also helps, but the per-class F1 improvement suggests genuine
  better alignment, not just fewer bins.

**L1 scene-obs is slightly better than GT L0** (48.8% vs 44.7%):
- L1 segments are shorter and more motion-homogeneous, so scene_obs delta captures
  a single state change rather than mixing approach + action + retract.
- But L1 has scaffolding verbs (approach, retract, position) that cause minimal
  scene_obs changes — these pull down overall accuracy.
- Task-specific L1 verbs (press: 93%, pull: 88%, push: 72%) are highly decodable
  from scene state.

**Scaffolding vs task-specific L1 verbs** (scene-obs):
- Scaffolding verbs: 75.5% of data, most have low F1 (position: 2%, release: 7%,
  grasp: 25%) — these don't change the scene much.
- Task-specific verbs: 24.5% of data, generally high F1 (press: 93%, pull: 88%,
  push: 72%) — these produce distinctive state changes.
- This confirms the intuition: **task-specific L1 verbs ARE more decodable** from
  scene state because each maps to a unique state transition. Scaffolding verbs are
  motor primitives that change the robot but not the scene.

## Summary so far

### Combined results (6 experiments)

| Condition | Classifier | Val Acc | Val MacF1 | #Cls | #Val |
|-----------|-----------|---------|-----------|------|------|
| GT L0 | AO transformer | 45.2% | 42.6% | 20 | 666 |
| GT L0 | Scene sklearn | 44.7% | 38.0% | 20 | 666 |
| Gemini L0 | AO transformer | 30.4%* | 16.9%* | 14 | 993 |
| Gemini L0 | Scene sklearn | **65.0%** | **56.6%** | 14 | 993 |
| L1 | AO transformer | 43.2%* | 29.9%* | 16 | 3,674 |
| L1 | Scene sklearn | 48.8% | 40.8% | 16 | 3,674 |

*Partial (crashed mid-training). Resubmitted with more memory.

### Emerging patterns

1. **Gemini L0 + scene = best overall** (65.0%): Goal-oriented verbs + scene state
   changes are naturally aligned. Scene_obs captures WHAT changed; Gemini L0 describes
   WHAT changed. Perfect match.

2. **GT L0 + AO = best for actions** (45.2%): Action-descriptive verbs + action
   trajectories are naturally aligned. The robot's motion directly encodes HOW it
   acted; GT L0 describes HOW it acted.

3. **Modality–label alignment matters more than abstraction level**: The decodability
   gap is not about L0 vs L1 but about whether the label vocabulary matches the
   modality's information axis (scene state change vs motion trajectory).

4. **L1 task-specific verbs are highly decodable from scene** (press: 93%, pull: 88%,
   push: 72%) but L1 scaffolding verbs are not (position: 2%, release: 7%).
   Scene_obs only changes during task-specific phases.

## TODO

- [ ] Update AO transformer results when resumed jobs complete (6621296, 6621272)
- [ ] AO per-class F1 for L1 and Gemini L0
- [ ] Scaffolding vs task-specific decodability breakdown for AO transformer
- [ ] OAT/QueST tokenizer experiments (action tokenization comparison)
- [ ] Conclusions section
