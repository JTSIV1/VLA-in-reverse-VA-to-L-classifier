# Round 3: Episode Task Type Classification

**Date:** 2026-03-16
**Script:** `scripts/build_episode_task_types.py`
**Output:** `data/episode_task_types.csv`

## Goal

Classify every CALVIN episode into one of three task types — `fixture_manip`, `block_acquire`, or `block_displace` — using only scene_obs and robot_obs (no instruction text). This lets us analyze how verb identity correlates with physical task type without circular reasoning.

## scene_obs layout (24-d)

| Dims | Object | Type |
|------|--------|------|
| 0 | sliding door joint | fixture |
| 1 | drawer joint | fixture |
| 2 | button | fixture |
| 3 | switch | fixture |
| 4 | lightbulb | fixture |
| 5 | green light | fixture |
| 6–8 | red block xyz | block |
| 9–11 | red block euler | block |
| 12–14 | blue block xyz | block |
| 15–17 | blue block euler | block |
| 18–20 | pink block xyz | block |
| 21–23 | pink block euler | block |

## Classification algorithm

For each episode, load start and end frames. Compute `scene_delta = scene_obs[end] - scene_obs[start]` and check `robot_obs[6]` (gripper width) at the end.

### Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FIXTURE_THRESH | 0.01 | Any fixture dim change above this |
| BLOCK_XYZ_THRESH | 0.01 | 1 cm block displacement |
| BLOCK_EULER_THRESH | 0.1 | ~5.7° rotation (needed for rotate/turn verbs that spin block in place) |
| GRIPPER_CLOSED_THRESH | 0.05 | robot_obs[6] below this → gripper is holding a block |
| DRAWER_Z_THRESH | 0.38 | Block z below this → block is sitting inside the drawer |
| FIXTURE_DISCRETE_THRESH | 0.5 | Fixture delta above this → binary state flip (e.g. light on/off) |

### Step 1: Detect what changed

- **Fixture changed:** any of scene_obs dims 0–5 has `|delta| > 0.01`
- **Block changed:** any block's xyz has `max(|delta|) > 0.01`, OR any block's euler has `max(|delta|) > 0.1`
- **Gripper closed:** `robot_obs[6] < 0.05` at episode end

The euler threshold was added because `rotate` and `turn` spin a block in place — xyz barely moves but euler angles change. Without it, 25 episodes (18 rotate, 3 turn, 3 take off, 1 remove) fell into "neither."

### Step 2: Resolve overlap (collateral contact)

When both fixture and block dims change (472 episodes), the movement of one is usually collateral to the other. Two rules handle this:

**Rule 2a — Discrete fixture state flip:** If any fixture dim has `|delta| > 0.5`, a binary state changed (e.g. lightbulb 0→1, switch flip). The block movement is incidental arm contact on the way to the fixture → classify as `fixture_manip`.

This was needed because turn on/off and move (switch) episodes often knock a block on the table while reaching for the light/switch. The block starts at z ≈ 0.46–0.56 (table height, above the drawer threshold) so Rule 2b alone would miss them. But the fixture delta is unmistakable: lightbulb/green_light deltas are exactly 1.0 (binary flip), vs block xyz deltas of 0.01–0.17.

**Rule 2b — Block inside drawer:** If the most-displaced block starts at z < 0.38 AND ends at z < 0.38, it was sitting inside the drawer and rode along → classify as `fixture_manip`.

This was validated by examining the z distribution of "both" episodes:

| z range | Count | Interpretation |
|---------|------:|----------------|
| 0.36–0.37 | 262 | Blocks inside drawer (collateral) |
| 0.38+ | 210 | Blocks on table/stacked (independent movement) |

The bimodal gap at z=0.38 provides a clean threshold. Examples:
- **close/open/pull drawer** (all blocks at z ≈ 0.362, std < 0.002): blocks sit in drawer, moved by drawer → fixture_manip
- **pick up block from drawer** (block starts low, ends high): block was pulled out by robot → block_acquire (gripper closed)
- **push block** with incidental fixture contact (block at z ≈ 0.46): block was on the table → block_acquire or block_displace depending on gripper

### Step 3: Acquire vs displace

For block episodes (or "both" episodes where block is primary):
- **Gripper closed at end** (robot_obs[6] < 0.05) → `block_acquire` (block is held)
- **Gripper open at end** (robot_obs[6] ≥ 0.05) → `block_displace` (block was released)

Validation of gripper signal (mean end gripper width by task type):

| Task type | N | Mean gripper | Min | Max |
|-----------|--:|---:|---:|---:|
| fixture_manip | 708 | 0.035 | -0.003 | 0.080 |
| block_acquire | 2,021 | 0.026 | -0.007 | 0.049 |
| block_displace | 1,341 | 0.075 | 0.051 | 0.080 |

Block_acquire and block_displace have clean separation on gripper width.

## Results

### Distribution

| task_type | train | val | total |
|-----------|------:|----:|------:|
| block_acquire | — | — | 2,021 (49.6%) |
| block_displace | — | — | 1,341 (32.9%) |
| fixture_manip | — | — | 708 (17.4%) |
| neither | — | — | 2 (0.05%) |

### Verb × task_type

| Verb | fixture_manip | block_acquire | block_displace | neither |
|------|-----:|-----:|-----:|-----:|
| close | 59 | 0 | 0 | 0 |
| collapse | 0 | 23 | 0 | 0 |
| grasp | 2 | 292 | 49 | 0 |
| lift | 1 | 263 | 42 | 0 |
| move | 98 | 12 | 6 | 0 |
| open | 51 | 1 | 0 | 0 |
| pick up | 1 | 439 | 73 | 0 |
| place | 0 | 4 | 188 | 0 |
| pull | 39 | 0 | 0 | 0 |
| push | 134 | 256 | 247 | 0 |
| put | 0 | 0 | 159 | 0 |
| remove | 0 | 40 | 5 | 1 |
| rotate | 0 | 246 | 101 | 0 |
| slide | 76 | 218 | 208 | 0 |
| stack | 0 | 1 | 85 | 0 |
| store | 0 | 2 | 46 | 0 |
| sweep | 0 | 74 | 91 | 0 |
| take off | 1 | 67 | 15 | 1 |
| turn | 0 | 59 | 23 | 0 |
| turn off | 117 | 0 | 0 | 0 |
| turn on | 129 | 0 | 0 | 0 |
| unstack | 0 | 24 | 3 | 0 |

### Visualization

![Verb × Task Type Distribution](../../figures/verb_task_type_distribution.png)

### Key observations

**Pure fixture verbs** (100% fixture_manip): close, pull, open, turn on, turn off
**Mostly fixture** (~85%): move — remaining 18 episodes have collateral block contact from reaching near the door/switch
**Polysemous verbs**: push (21% fixture, 40% acquire, 39% displace) and slide (15% fixture, 43% acquire, 41% displace) — these verbs physically act on both fixtures and blocks
**Pure block-acquire**: collapse, unstack (100%), grasp/pick up/lift/take off/remove (~85–90% acquire, rest displace)
**Pure block-displace**: put (100%), place/store/stack (~95%+ displace)
**Mixed block**: sweep (45% acquire, 55% displace), rotate (71% acquire, 29% displace), turn (72% acquire, 28% displace)

**Verb clusters by dominant task type:**

| Cluster | Verbs | Interpretation |
|---------|-------|----------------|
| Fixture manipulation | close, open, pull, move, turn on, turn off | Changing fixture state |
| Block acquiring | grasp, pick up, lift, take off, remove, collapse, unstack | Getting block into gripper |
| Block displacing | place, put, store, stack | Releasing block at target |
| Mixed block | rotate, turn, sweep, push(block), slide(block) | Could end held or released |

## CSV columns

| Column | Description |
|--------|-------------|
| split | train / val |
| start_idx, end_idx | CALVIN episode frame indices |
| instruction | Raw language annotation |
| task_type | `fixture_manip`, `block_acquire`, `block_displace`, or `neither` |
| verb | Primary verb (from `utils.load_calvin_to_dataframe`) |
| object_modifier | Adjective/state modifiers + source location (e.g., `red;in drawer`) |
| object | Direct object noun (e.g., `block`, `drawer`, `switch`) |
| direction | Directional goal (e.g., `left`, `right`, `up`, `into`, `on top`) |
| target_location | Named target (e.g., `drawer`, `top of stack`, `left side`) |
