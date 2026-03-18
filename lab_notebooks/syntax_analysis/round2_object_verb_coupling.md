# Round 2: Object–Verb Coupling and Its Effect on Verb Decodability

**Date:** 2026-03-16
**Goal:** Use the syntactic decomposition (R1) to explain *why* scene_obs and action-only models succeed or fail on specific verb classes, and test whether the strong verb–object coupling predicted by syntax translates into actual classification performance patterns.

## Prediction from Syntax Analysis

Round 1 found that **object type strongly constrains verb choice**:
- **Fixture-only verbs** (open, close, pull, turn on/off, move): bound to specific fixtures (drawer, light, switch). If a model can identify *which fixture changed*, it can narrow the verb to 1–3 candidates.
- **Block-only verbs** (grasp, pick up, rotate, lift, sweep, stack, etc.): all act on the same object type (blocks). No object-level disambiguation — the model must distinguish *how* the block was manipulated.
- **Mixed verbs** (push, slide, place, put, store): apply to multiple object types.

**Hypothesis:** Scene_obs (which encodes fixture states directly) should strongly outperform action-only on fixture verbs, while action-only (which encodes motion trajectories) should do better on block-only verbs where scene_obs sees nearly identical state changes.

## Data Source

All comparisons use the **sklearn complementarity analysis** from R7 (2026-03-15), which pairs:
- **AO**: Transformer on action trajectories (r8_ao_native_best.pth), 45.2% acc
- **Scene**: sklearn MLP on scene_engineered features (96-d), 44.7% acc

Same val set: 666 episodes, 20 classes. Source: `results/episode_complementarity.csv`.

## Results by Object Category

### Aggregate

| Category | N | AO Acc | Scene Acc | Both OK | AO+ | SC+ | Neither | Oracle |
|----------|--:|-------:|----------:|--------:|----:|----:|--------:|-------:|
| **Fixture-only** | 82 | **74.4%** | **67.1%** | 51.2% | 23.2% | 15.9% | 9.8% | **90.2%** |
| **Block-only** | 328 | **54.6%** | 39.9% | 25.0% | 29.6% | 14.9% | 30.5% | 69.5% |
| **Mixed** | 256 | 23.8% | **43.8%** | 11.7% | 12.1% | 32.0% | 44.1% | 55.9% |

### The prediction is partially confirmed, partially surprising

**Confirmed: Fixture verbs are easiest for both models.** Both AO (74.4%) and scene (67.1%) achieve their highest accuracy on fixture verbs, and the oracle union reaches 90.2%. The strong verb–object coupling means both modalities can leverage object identity as a shortcut.

**Surprising: AO beats scene on fixtures overall (74.4% vs 67.1%).** The syntax-based prediction was that scene_obs should dominate here. But looking per-verb, the picture is more nuanced:

| Verb | N | AO Acc | Scene Acc | Winner |
|------|--:|-------:|----------:|--------|
| close | 9 | **100.0%** | 44.4% | AO |
| open | 9 | **100.0%** | 77.8% | AO |
| move | 19 | **68.4%** | 26.3% | AO |
| pull | 4 | 0.0% | **50.0%** | Scene |
| turn off | 17 | 52.9% | **88.2%** | Scene |
| turn on | 24 | 87.5% | **91.7%** | Scene |

**Scene dominates turn on/off** (88–92%): light state is a near-binary oracle in scene_obs. **AO dominates close/open** (100%): the drawer-opening motion trajectory is apparently more distinctive than the drawer joint position in scene_obs. **Both fail on pull** — only 4 samples, and AO scores 0%.

**Confirmed: Block-only verbs are hardest, AO leads.** AO (54.6%) substantially outperforms scene (39.9%) on block-only verbs. With all verbs acting on the same object type, scene_obs sees similar state changes (block moved from A to B) and must rely on subtle differences in *how* the block moved. Action trajectories directly encode this motion.

**Surprising: Scene wins on mixed verbs (43.8% vs 23.8%).** Push/slide/place/put act on multiple object types. Scene_obs can detect *which* object changed (door joint vs drawer joint vs block xyz) and use that to disambiguate, while action trajectories for pushing a door vs pushing a block look similar enough to confuse the AO model.

## Deep Dive: Block-Only Verbs

Since blocks account for 73% of all samples, this is where the classification battle is fought.

### Per-verb performance

| Verb | N | AO | Scene | Winner | Confusion pattern |
|------|--:|---:|------:|--------|-------------------|
| rotate | 57 | **100%** | 84.2% | AO | Distinctive motion; scene also good (euler changes) |
| grasp | 61 | **96.7%** | 29.5% | AO | AO: gripper closing pattern; Scene confuses with pick up (34/43) |
| take off | 13 | **76.9%** | 23.1% | AO | Lifting from stack has clear trajectory; scene sees generic z-change |
| stack | 13 | **69.2%** | 15.4% | AO | Placing on block: trajectory matters; scene sees generic block move |
| sweep | 26 | **61.5%** | 0.0% | AO | Scene 0% — sweep's state change (block displaced) looks like push/slide |
| lift | 49 | **51.0%** | 28.6% | AO | AO confuses w/ grasp(23); scene confuses w/ pick up(26) |
| remove | 8 | **37.5%** | 0.0% | AO | AO confuses w/ take off(5); scene: scattered |
| pick up | 85 | 0.0% | **54.1%** | **Scene** | AO: 0%! All predicted as grasp(57) or lift(27) |
| turn | 16 | 0.0% | 0.0% | TIE | Both fail — rotating a block is confused with rotate |

### Key finding: Acquire-verb syndrome

The "acquire" verbs {grasp, pick up, lift, take off} all involve the robot hand closing on a block and moving it. Their performances are almost perfectly complementary:

| Verb | AO | Scene | What distinguishes it |
|------|---:|------:|----------------------|
| grasp | **96.7%** | 29.5% | Gripper closing pattern in action trajectory |
| pick up | 0.0% | **54.1%** | Block z-increase in scene_obs |
| lift | 51.0% | 28.6% | Intermediate — motion somewhat distinct, state somewhat distinct |
| take off | **76.9%** | 23.1% | From-stack context in trajectory |

AO classifies **everything** that looks like "hand grabs block" as `grasp` — it gets 96.7% on grasp but 0% on pick up (all 85 pick-up episodes predicted as grasp or lift). Scene_obs does the opposite: it sees "block z went up" and classifies as `pick up`, getting 54.1% on pick up but only 29.5% on grasp.

**These verbs are near-synonyms with overlapping motions and overlapping state changes.** Each model picks up on the one distinguishing feature in its modality and collapses the rest.

### Key finding: Displace-verb complementarity

The "displace" verbs {push, slide, sweep} show even stronger complementarity:

| Verb | AO | Scene | What distinguishes it |
|------|---:|------:|----------------------|
| sweep | **61.5%** | 0.0% | Sweeping motion in trajectory (scene: looks like push) |
| push | 5.5% | **48.6%** | Scene detects which object moved (door/drawer/block) |
| slide | 35.8% | **45.7%** | Scene detects object + direction |

AO is blind to push (5.5%) because pushing a block, door, and drawer have different trajectories that the model confuses. Scene is blind to sweep (0%) because sweep's state change (block displaced horizontally) is identical to push.

## The Object-Type Explanation for Mixed Verbs

Why does scene_obs win on mixed verbs (43.8% vs 23.8%)? Because mixed verbs act on *different objects*, and scene_obs directly encodes which object changed:

| Verb | N | AO | Scene | Explanation |
|------|--:|---:|------:|-------------|
| push | 109 | 5.5% | **48.6%** | Scene detects door-joint-change vs drawer-joint-change vs block-xyz-change |
| slide | 81 | 35.8% | **45.7%** | Same — object type is detectable in scene_obs |
| place | 35 | **57.1%** | 48.6% | Placing motion is distinctive; scene also decent |
| put | 25 | 0.0% | **20.0%** | AO confuses with place/store; both struggle |
| store | 6 | **100%** | 0.0% | Small N; store's trajectory is unique but rare |

**Push** is the clearest example: 109 val episodes spanning blocks (50%), doors (25%), drawers (15%), and switches (10%). AO sees 4 different trajectory patterns and can't unify them under "push." Scene sees "door joint changed → something pushed the door" and succeeds.

## Summary: What the Syntax Predicts and What It Misses

| Prediction | Verdict | Detail |
|------------|---------|--------|
| Scene_obs should excel on fixture verbs | **Partially confirmed** | True for turn on/off (88–92%), but AO beats scene on open/close (100%) |
| AO should excel on block-only verbs | **Confirmed** | 54.6% vs 39.9%; AO dominates grasp, sweep, take off, stack |
| Object-type should help scene on mixed verbs | **Confirmed** | 43.8% vs 23.8%; scene leverages which-object-changed for push/slide |
| Verb synonyms should confuse both models | **Confirmed** | {grasp, pick up, lift} and {push, slide, sweep} show near-perfect complementarity |

### Implications for fusion

The complementarity is strongest exactly where verb–object coupling is weakest (block verbs, mixed verbs). A fusion model that can route:
- Motion-discriminative verbs (grasp, sweep, take off) → action signal
- State-discriminative verbs (turn on/off, push, pick up) → scene signal
- Synonym-ambiguous verbs (lift, slide) → both signals jointly

...should exceed the 66.8% oracle union ceiling of the current independent models.
The R7 fusion model (43.1%) captures only ~60% of this potential, suggesting significant room for improvement in fusion architecture.
