# Round 8: Verb Granularity — Are Low-Level Control Verbs Easier to Decode?

**Date**: 2026-03-16
**Motivation**: Recent work in robot learning decomposes high-level language instructions
(e.g. "pour food into bowl") into low-level control primitives (e.g. "move left, then rotate
wrist"). This raises a question for our VA→L classifier: **are low-level motor verbs easier
to decode from action trajectories than high-level goal verbs?**

Intuitively, low-level verbs should map more directly to action trajectories since they
describe the motion itself. High-level verbs describe outcomes (place, store, stack) that
could be achieved by different motions. But our existing per-class results suggest the
opposite may be true.
<!-- 
## Part 1: Preliminary Analysis from Existing Data

### Three-way taxonomy of CALVIN's 21 verb classes

| Category | Verbs | Description |
|----------|-------|-------------|
| **Fixture** | open, close, turn on, turn off | State changes on fixed objects (drawers, lights) |
| **Manipulation** | pick up, place, put, stack, store, grasp, take off, remove | Goal-oriented object manipulation |
| **Motion Primitive** | move, push, pull, slide, rotate, lift, turn, sweep, lift up | Direct motion descriptions |

### Per-class F1 (R7 Scene Token + Native, best fusion model)

| Category | Verb | F1 | N |
|----------|------|----|---|
| Fixture | turn on | 72.7 | 24 |
| Fixture | turn off | 66.7 | 17 |
| Fixture | close | 57.1 | 7 |
| Fixture | open | 40.0 | 7 |
| **Fixture avg** | | **59.1** | **55** |
| Manipulation | grasp | 61.3 | 54 |
| Manipulation | place | 58.0 | 31 |
| Manipulation | take off | 47.1 | 8 |
| Manipulation | remove | 46.2 | 8 |
| Manipulation | store | 40.0 | 6 |
| Manipulation | stack | 30.8 | 13 |
| Manipulation | pick up | 22.0 | 77 |
| Manipulation | put | 19.0 | 25 |
| **Manipulation avg** | | **40.5** | **222** |
| Motion Prim | rotate | 82.6 | 64 |
| Motion Prim | pull | 50.0 | 4 |
| Motion Prim | move | 44.0 | 19 |
| Motion Prim | sweep | 31.7 | 26 |
| Motion Prim | lift | 27.6 | 32 |
| Motion Prim | slide | 23.0 | 64 |
| Motion Prim | lift up | 20.7 | 11 |
| Motion Prim | push | 11.9 | 88 |
| Motion Prim | turn | 8.0 | 16 |
| **Motion Prim avg** | | **33.3** | **324** |

**Surprising finding**: Motion primitives are the *hardest* to decode (33.3% mean F1),
not the easiest. Fixture interactions are easiest (59.1%), with goal-oriented manipulation
in between (40.5%).

### The motor synonymy problem

Top confusion pairs from R8 AO-native predictions:

| True | Predicted as | Count | % of true class |
|------|-------------|------:|-------:|
| pick up | grasp | 57 | 67.1% |
| push | slide | 42 | 38.5% |
| push | sweep | 34 | 31.2% |
| turn | rotate | 15 | 93.8% |
| lift | grasp | 23 | 46.9% |
| put | place | 11 | 44.0% |
| slide | sweep | 17 | 21.0% |
| remove | take off | 5 | 62.5% | -->
<!-- 
Multiple low-level verbs map to the same action trajectory ("motor synonyms"):
- **Grasping**: pick up, grasp, lift, lift up → close gripper + lift
- **Lateral motion**: push, slide, sweep, move → translate object on table
- **Rotation**: turn, rotate → wrist rotation
- **Placement**: put, place, stack, store → move to location + release
- **Detachment**: remove, take off → separate object from fixture

Fixture verbs are easy because each interacts with a *different mechanism*
(drawer vs switch vs door), producing genuinely distinct trajectories.

### Limitation of this analysis

CALVIN only has **one level** of language annotation (single-sentence task instructions).
Artificially binning these verbs into "low-level" vs "high-level" is a post-hoc
categorization, not a true test of decodability at different abstraction levels. To properly
test this, we need **hierarchical language annotations** at multiple granularity levels
on the same trajectories. -->

## Part 2: Hierarchical Annotation via Gemini

### Background: VLM2VLA pipeline

"Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting"
(arXiv 2509.22195) uses Gemini 2.5 to decompose BridgeV2 trajectories into a hierarchy:

| Level | Name | Example | Scope |
|-------|------|---------|-------|
| L0 — Task | High-level instruction | "put the lid on the pot" | Full episode |
| L1 — Subtask | Semantic phase | "Grasp the Lid" | Segment of trajectory |
| L2 — Motion plan | Directional control | "move down and slightly right" | Same segment |
| L3 — Action chunk | Numerical actions | `[[dx, dy, dz, grip], ...]` | Same segment |

VLM2VLA has no public code. We replicate the pipeline for CALVIN.

### Key finding: L2 is a dead end for verb classification

We initially planned to annotate L0/L1/L2/L3. However, pilot experiments showed that
**L2 motion-plan labels have extremely low verb diversity** — nearly all segments reduce
to "move [direction]" + "open/close gripper". This makes L2 useless for studying verb
decodability (only ~3-4 distinct verb types).

This is expected: at the primitive motion level, all manipulation reduces to the same
small set of directional commands. The diverse, informative verbs live at L1 (subtask).

**Decision**: Focus on **L0 (task) vs L1 (subtask)** comparison only.

### Iterative prompt development

We ran multiple pilot rounds to develop the annotation prompt:

#### Pilot 1: VLM2VLA-style (L0 given, 12 frames at 5Hz, full action log)
- Good L1 decomposition with correct temporal boundaries
- But: Gemini hallucinated L1 labels for some episodes (e.g., ep5 "push the switch
  upwards" → Gemini said "Move cylinder towards socket" — completely wrong task)
- Root cause: CALVIN's 200×200 images are too low-res for Gemini to reliably identify
  objects; it sometimes ignores the L0 instruction and describes what it "sees"

#### Pilot 2: Vision-only (no L0, no actions, 12 frames at 5Hz)
- Tested whether Gemini can infer L0 from vision alone
- **Result**: Mostly wrong. E.g., "move the door to the left side" → "Pick up the pink
  block from the toolbox". Confirmed vision-only is unreliable for CALVIN.

#### Pilot 3: Fewer frames + scene_obs (6 frames, action + scene_obs per frame)
- Reduced to 6 evenly-spaced frames
- Added scene_obs (24-d) with dimension labels to each frame's data
- Scene_obs tells Gemini exactly what changed (drawer_pos, switch_state, block positions)
- **Result**: Hallucinations eliminated. Gemini correctly identifies tasks from state
  changes even when images are ambiguous.
- But: phase boundaries between sampled frames were **fabricated** (e.g., "t=26..37"
  when Gemini only saw data at t=0, 13, 26, 39, 52, 64)

#### Pilot 4: Full state log + 6 frames (FINAL)
- Send **all timesteps** of action + scene_obs as compact text log
- Only 6 images (to save cost), but full 30Hz numerical data
- **Result**: Precise boundaries (single-timestep "Release" phases at exact gripper
  events), correct scene grounding, diverse verbs.

#### Pilot 5: Gemini infers L0 (no GT instruction given)
- Removed GT L0 instruction from the prompt entirely
- Gemini generates both L0 and L1 from frames + state log
- **Result**: L0 inference quality is good with state data. Gemini tends to produce
  **more goal-oriented** L0 (e.g., "Turn on the light") vs CALVIN's **more action-
  descriptive** GT (e.g., "push the switch upwards"). Both are valid but at different
  abstraction levels.
- This is now the production pipeline since it removes annotation bias from L0.

### Final pipeline design

**Script**: `scripts/annotate_calvin_hierarchy.py`

**Input per episode** (sent to Gemini 2.5 Pro):
- 6 evenly-spaced RGB frames (JPEG, from `rgb_static`)
- Full state log at 30Hz: `rel_actions` (7-d) + `scene_obs` (24-d) per timestep
- Scene_obs dimensions explained: fixtures (door/drawer/button/switch/lights) + 3 block
  positions and orientations
- NO GT task instruction (Gemini infers L0)

**Output per episode** (JSON):
- `TASK_INSTRUCTION`: Gemini-inferred L0 (high-level task)
- `DECOMPOSITION`: array of L1 phases, each with:
  - `STEP_DESCRIPTION`: one-verb subtask label
  - `REASONING`: segmentation justification referencing state data
  - `START_TIMESTEP`, `END_TIMESTEP`: boundaries at 30Hz

**Saved fields** per episode (JSONL):
- `instruction_gt`: CALVIN's original L0 annotation
- `instruction_gemini`: Gemini-inferred L0
- `decomposition`: L1 phase array
- `start_idx`, `end_idx`, `n_steps`, `n_frames`, `frame_indices`

### Pilot 5 results (10 episodes)

| ep | GT L0 | Gemini L0 | Phases | Phase verbs |
|----|-------|-----------|--------|-------------|
| 0 | "move the door to the left side" | "Slide the door to the right" | 3 | approach, slide, retract |
| 1 | "slide the door to the left side" | "Slide the door open to the right" | 5 | release, approach, grasp, slide, retract |
| 3 | "toggle the button to turn on the green light" | "Slide the door...then press the button" | 6 | reposition, open, slide, approach, open, press |
| 4 | "toggle the light switch to turn on the yellow light" | "Turn on the light" | 5 | release, retract, approach, align, push |
| 5 | "push the switch upwards" | "Turn on the light by flipping the switch" | 4 | position, release, approach, push |
| 6 | "push down the button to turn on the led" | "Press the button to turn on the green light" | 5 | release, approach, open, lower, press |
| 7 | "open the cabinet drawer" | "Open the drawer" | 6 | release, retract, approach, align, pull, retract |
| 8 | "grasp the drawer handle and open it" | "Open the drawer" | 4 | place, release, approach, hook, pull |
| 9 | "move up the switch" | "Turn on the light by flipping the switch" | 3 | reposition, lower, push |

**Observations**:
1. Gemini L0 is **more goal-oriented** than GT (e.g., "Turn on the light" vs "push the
   switch upwards") — it describes the outcome, GT describes the action.
2. Ep3: Gemini detected a door slide [t=13..26] not mentioned in GT. This is a real
   state change (sliding_door_pos changed) — the robot did slide the door as part of
   reaching the button. Gemini captured this; CALVIN's GT missed it.
3. L1 verb diversity is good: approach, slide, release, grasp, push, pull, press, retract,
   align, hook, lower, position, place, open, reposition, flip.
4. Phase boundaries are precise: single-timestep phases for gripper events (e.g., ep5
   "Release the object" at exactly t=31).
5. 3–6 phases per episode (avg ~4.3), each covering ~15 timesteps.

**Visualization**: `figures/hierarchy_pilot_v3.png`

### Cost estimate

| Split | Episodes | Est. input tokens | Est. output tokens | Est. cost |
|-------|----------|------------------:|-------------------:|----------:|
| Training | 5,124 | ~21.3M | ~3.1M | ~$57 |
| Validation | 1,011 | ~4.2M | ~0.6M | ~$11 |
| **Total** | **6,135** | **~25.5M** | **~3.7M** | **~$69** |

### Full annotation run

**Status**: Complete (2026-03-17)
- Training: 5,112 / 5,124 annotated (12 API errors, 99.8% coverage)
- Validation: 1,011 / 1,011 annotated (100%)
- Parallelized with 8 SLURM shards on `cpu` partition (`scripts/submit_gemini_parallel.sh`)
- Actual runtime: ~6 hours wall-clock (8 parallel workers, ~38s/episode Gemini thinking time)
- Estimated spend: ~$69

Output files:
- `data/hierarchy_annotations/calvin_training.jsonl` (5,112 episodes, merged from shards)
- `data/hierarchy_annotations/calvin_validation.jsonl` (1,011 episodes)
- `data/hierarchy_annotations/errors_training.jsonl` (12 failed episode indices)

### Annotation statistics

| Metric | Training | Validation |
|--------|----------|------------|
| Episodes | 5,112 | 1,011 |
| Total L1 phases | 21,167 | 4,224 |
| Phases/episode (mean) | 4.1 | 4.2 |
| Phases/episode (range) | 1–9 | 1–8 |
| Unique L1 first-word verbs | 68 | — |

### L1 verb distribution (training, top 20)

| Verb | Count | % | Example descriptions |
|------|------:|--:|---------------------|
| approach | 3,054 | 14.4% | "Approach the door handle", "Approach the drawer" |
| retract | 2,688 | 12.7% | "Retract the arm", "Retract from the button" |
| lift | 2,676 | 12.6% | "Lift the blue block", "Lift arm above the switch" |
| grasp | 2,359 | 11.1% | "Grasp the red block", "Grasp the door handle" |
| lower | 1,833 | 8.7% | "Lower onto the handle", "Lower the pink block" |
| release | 1,761 | 8.3% | "Release the blue block", "Release the object" |
| push | 1,311 | 6.2% | "Push the door to the left", "Push the light switch up" |
| reposition | 732 | 3.5% | — |
| move | 730 | 3.4% | — |
| place | 561 | 2.7% | "Place the pink block on the table" |
| rotate | 516 | 2.4% | "Rotate the red block", "Rotate the block upright" |
| align | 410 | 1.9% | — |
| transport | 346 | 1.6% | — |
| position | 342 | 1.6% | — |
| press | 330 | 1.6% | "Press the button", "Press the switch with the block" |
| open | 286 | 1.4% | — |
| reach | 231 | 1.1% | — |
| pull | 195 | 0.9% | "Pull the drawer open" |
| slide | 193 | 0.9% | "Slide the door to the left" |
| flip | 114 | 0.5% | "Flip the switch up", "Flip the switch down" |

Long tail: descend (93), translate (65), close (48), hook (43), grip (37), hold (32),
adjust (23), sweep (18), engage (12), secure (12), ... (68 unique verbs total).

**Key observations**:
1. **Top 6 verbs account for 68.8%** of all L1 phases: approach, retract, lift, grasp,
   lower, release. These are the "scaffolding" verbs that appear in almost every episode.
2. **Task-specific verbs** (push, place, rotate, press, pull, slide, flip) are rarer
   (0.5–6.2%) but more informative — they correspond to the actual manipulation, not
   the approach/retract wrapper.
3. **68 unique verbs** vs CALVIN's 21 L0 classes — L1 is much richer but heavily
   skewed toward scaffolding.

### Gemini L0 vs GT L0 (qualitative)

| GT L0 | Gemini L0 |
|-------|-----------|
| "move the door to the left side" | "Slide the door to the left." |
| "slide down the switch" | "Turn off the light" |
| "toggle the button to turn on the green light" | "Slide the door to the left and press the button." |
| "push the switch upwards" | "Turn on the light" |
| "open the cabinet drawer" | "Open the drawer" |
| "grasp the drawer handle and open it" | "Open the drawer." |
| "move up the switch" | "Turn on the light by flipping the switch." |

Gemini L0 tends to be **more goal-oriented** (describes the outcome: "Turn on the light")
while CALVIN GT is **more action-descriptive** (describes the motion: "push the switch
upwards"). Both are valid but at different abstraction levels. Ep3 is notable: Gemini
detected a door slide not mentioned in the GT instruction.

## Part 3: Proposed Experiments

### Exp 8.1: L0 vs L1 Decodability Comparison

Train the VA→L classifier at two levels on the same action trajectories:
- **L0**: CALVIN GT verbs (21 classes, existing baseline: 45.2% acc / 42.6% MacF1)
- **L1**: Gemini subtask verbs (68 raw verbs → need consolidation)

For L1, each L1 phase becomes a separate training sample with its own action subsequence
(sliced by `START_TIMESTEP..END_TIMESTEP`). This means:
- ~21K samples (vs ~5K L0) — 4× more data
- ~16 steps/sample (vs ~61 steps) — shorter, more motion-homogeneous
- Heavily imbalanced: approach/retract/lift/grasp dominate

**Design decisions for L1 classifier**:
- Consolidate near-synonyms: reach→approach, descend→lower, transport→move,
  position→align, grip→grasp, hold→grasp, etc.
- Consider filtering scaffolding verbs (approach, retract, release) that are not
  task-informative, OR treat them as valid classes and measure their decodability
- Use sp+wt recipe (min_class_count 30 + weighted CE)

**Hypothesis**: L1 scaffolding verbs (approach, retract, grasp, release) will be
highly decodable because they map to distinctive motion patterns. Task-specific
L1 verbs (push, pull, slide, press, flip) should also be easier than L0 because
each L1 segment contains only that one motion, not the full episode trajectory
mixing approach + action + retract.

### Exp 8.2: Gemini L0 vs GT L0 Agreement

Quantify how well Gemini's inferred L0 matches CALVIN's GT:
- Semantic similarity (sentence embedding cosine similarity)
- Verb extraction agreement (extract verb from Gemini L0, compare to GT verb)
- Error analysis: when they disagree, which is "correct"?

This validates whether Gemini understood the trajectory well enough to produce reliable
L1 decompositions.

### Exp 8.3: Motor Synonym Cluster Classification

Collapse the 21 L0 verb classes into ~6 motor clusters and retrain:
- Reuse existing AO-native architecture, sp+wt recipe, 6 classes
- Expected: much higher accuracy since within-cluster confusion is eliminated
- Provides a ceiling for what action trajectories alone can discriminate

### Exp 8.4: Action Trajectory Distance Analysis

For each pair of verb classes, compute mean L2 distance between frozen CLS embeddings.
Visualize as a distance matrix sorted by motor cluster. Quantifies whether motor synonyms
are genuinely closer in action space.
