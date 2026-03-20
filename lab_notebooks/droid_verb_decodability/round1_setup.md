# Round 1: DROID Verb Decodability — Setup & Action-Only Results

**Date:** 2026-03-19

## Motivation

CALVIN has limited task diversity (22 verbs, ~250 templated instructions). DROID offers
75K real-world demonstrations with 307 unique verbs from free-form annotations — a much
harder and more realistic testbed for action verb decodability.

## Dataset: DROID

- **Source:** `gs://gresearch/robotics/droid/1.0.1/` (2048 TFRecord shards, ~1.7TB total)
- **Annotations:** `KarlP/droid` on HuggingFace (75,144 episodes × 3 language annotations each)
- **Action space:** 7-dim (cartesian_position[6] + gripper_position[1]), avg ~385 steps/episode
- **No scene_obs** (real-world, no simulator state) — visual goal proxy = first/last frame

## Data Processing

### Action Extraction (job 6654450, DONE)
- Script: `scripts/extract_droid_actions.py`
- SLURM: `scripts/submit_droid_extraction.sh` (32 array tasks × 64 shards each)
- Output: `/data/user_data/wenjiel2/datasets/droid_actions/shard_*.npz` (~800MB total)
- Each .npz contains per-episode: actions (T,7), language instructions, episode path

### Full Shard Download (job 6656093, DONE)
- Script: `scripts/download_droid_shards.sh`
- Output: `/data/user_data/wenjiel2/datasets/droid_rlds/` (1.7TB, 2048 shards)
- Needed for: visual goal experiments (first/last frames), future BC training

### Frame Extraction (job 6658632, DONE)
- Script: `scripts/extract_droid_frames.py`
- Output: `/data/user_data/wenjiel2/datasets/droid_frames/frames_*.npz` (~4GB)
- First + last frame (JPEG bytes) from `exterior_image_1_left` per episode

### Consolidation
- Script: `scripts/consolidate_droid_actions.py`
- Output: `data/droid_episodes_filtered.csv` (48,015 single-verb episodes, 228 unique verbs)

### Verb Extraction Stats (from language_instruction1, spaCy)
| Category | Count | % |
|----------|-------|---|
| Total episodes loaded | 95,658 | 100% |
| Single-verb (kept) | 51,504 | 53.8% |
| Multi-verb (discarded) | 22,597 | 23.6% |
| Zero-verb (discarded) | 21,557 | 22.5% |
| After "then"/"and" filter | 48,015 | — |

### Verb Merging
Applied a merge map to collapse synonymous verb+particle variants:
- **Clear merges:** flip over→flip, fold up→fold, stack up→stack, etc.
- **Cautious merges:** press down→press, slide out→slide, put down→put, etc.
- **Directional merges:** pull out/up/down→pull, push down/up/in→push, pour out→pour
- **Cross-verb:** turn over→flip
- Result: 288→228 unique verbs

### Verb Distribution (post-merge)
- 228 unique verbs total; with min_class_count≥30: **47-53 classes** (depends on checkpoint era)
- Top 5: put (15,901), move (8,320), close (2,946), remove (2,891), open (2,635)
- Heavy imbalance — "put" alone is 33% of all episodes
- Long tail of noisy verbs from spaCy misparsing (filtered by min_class_count)
- See `figures/abcd_d_verb_distribution.png`

## Action-Only Verb Classification Results

### Experiment 1: d128 baseline (job 6656090)
- Model: ActionToVerbTransformer, d_model=128, 4 layers, action_only
- Config: max_seq_len=512, weighted CE, 30 epochs, lr=1e-4
- **Best val acc: 25.99%** @ epoch 27
- Macro recall: ~28%
- 53 verb classes (pre-merge checkpoint)

### Experiment 2: d256 + regularization (job 6657183)
- Model: d_model=256, 6 layers, action_only
- Config: max_seq_len=512, weighted CE, label_smoothing=0.1, weight_decay=0.01, patience=15
- **Best val acc: 29.03%** @ epoch 73, early stopped at epoch 88
- Significant overfitting: train acc 66% vs val 29%
- 47 verb classes (post-merge checkpoint)

### Comparison
| Model | Best Val Acc | Macro Recall | Train Acc (final) | Classes |
|-------|-------------|-------------|-------------------|---------|
| d128 (4L) | 25.99% | ~28% | 27.3% | 53 |
| d256 (6L) | 29.03% | ~29% | 66.1% | 47 |

Key observations:
- d256 gains +3pp val acc but at cost of severe overfitting (66% vs 29%)
- Extra capacity mostly memorizes training data
- Both plateau well below CALVIN's ~40% (expected: 228→47 verbs vs 21, noisier labels, more diverse scenes)

### t-SNE Embedding Visualization

![DROID Action t-SNE](../../figures/droid_action_tsne.png)

Side-by-side t-SNE of CLS token embeddings from d128 (left) and d256 (right), colored by top-15 verbs:

- **"turn off"**, **"turn on"**, **"hang"** form tight, well-separated clusters — physically distinct actions
- **"fold"** clusters distinctly — unique trajectory pattern
- **"open"** and **"close"** have identifiable regions
- **"put"** and **"move"** form a mixed central blob — kinematically similar, need visual context to disambiguate
- d256 produces tighter, more separable clusters than d128, consistent with +3pp accuracy gain

## Planned: Visual Goal Classification (Phase 2)

- Script: `train_droid_goal.py`
- SLURM: `scripts/submit_droid_goal.sh`
- Model: ActionToVerbTransformer in vision_only mode, delta_patches=0 (full scene)
- Encoders: DINOv2-S and VC-1, frozen, 2 frames (first + last)
- Hypothesis: verbs like "put" vs "move" should be easier to separate from visual change than from action trajectory

## Key Differences from CALVIN
| | CALVIN D→D | DROID |
|---|---|---|
| Episodes | 3,309 train / 665 val | ~40K train / ~7K val |
| Verbs (sparse) | 21 | 47 (≥30 samples, post-merge) |
| Avg trajectory length | ~61 steps | ~385 steps |
| Action dim | 7 (rel_actions) | 7 (cartesian + gripper) |
| Scene state (goal) | 24-d scene_obs (simulator) | First/last frame visual change |
| Instructions | Templated (~250 unique) | Free-form (~75K unique) |
| Environment | Single simulated desk | 564 real-world scenes |
| Best AO acc | 39.5% (d128, 21 cls) | 29.0% (d256, 47 cls) |
