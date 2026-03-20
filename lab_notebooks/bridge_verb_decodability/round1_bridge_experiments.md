# Round 1: BridgeV2 Verb Decodability

**Date:** 2026-03-19

## Motivation

DROID experiments probe verb decodability at the **task level** (one verb per full episode).
BridgeV2 lets us compare two granularity levels:
1. **High-level** — same as DROID: classify the task verb from the full episode trajectory
2. **Subtask-level** — classify atomic action verbs (grasp, lift, place, ...) from short segments

The subtask annotations come from **Emma-X GCOT** (declare-lab/Emma-X-GCOT), which used
Gemini 1.5 Pro to decompose 38K BridgeV2 episodes into temporally-segmented subtask sequences.

## Dataset: BridgeV2

- **Source:** TFDS format, 1024 shards (~112GB), downloaded to `/data/user_data/wenjiel2/datasets/bridge_v2/`
- **Action space:** 7-dim (joint velocities + gripper), avg ~30-50 steps/episode
- **Episodes:** ~53K total (train + val)

### Action Extraction
- Script: `scripts/extract_bridge_actions.py`
- SLURM: `scripts/submit_bridge_extraction.sh` (32 array tasks × 32 shards)
- Output: `/data/user_data/wenjiel2/datasets/bridge_actions/shard_*.npz` (1024 files)
- Each .npz: actions (T,7), state (T,7), episode_key, instruction, n_steps

## Experiment A: Subtask-Level Classification

### Data Pipeline
1. Downloaded Emma-X GCOT annotations (`plans_train.json`, 63MB, 38,660 episodes)
2. Parsed Gemini's raw text into structured segments (segment_num, start_frame, end_frame, subtask_label)
3. Cleaned verb labels:
   - Gerund → base form (grasping → grasp, lifting → lift)
   - Dropped non-actions (locate, identify, maintain, observe, pause, etc.)
   - Merged directional variants: move towards/away/back → move; position over → position
   - Dropped slash-combos (Gemini hedging artifacts)
4. Matched episode keys (100% match: all 37,785 Emma-X episodes found in BridgeV2 shards)
5. Extracted action sub-trajectories for each segment

### Stats
- **167,603 segments** across 37,785 episodes
- **48 verbs** (with min_class_count ≥ 30)
- Mean segment length: **9.1 frames** (with ±1 frame context padding), median 8
- Segments are very short: 80% fall between 4–12 frames
- Top 10 verbs cover **92.8%**: move (24.5%), grasp (19.0%), lift (14.0%), reach (9.0%),
  position (8.0%), place (6.2%), release (5.3%), approach (3.6%), sweep (2.0%), pull (1.4%)

### Isolated Segments (job 6661257)
- d_model=128, 4 layers, max_seq_len=20, batch_size=64
- Unweighted CE + label_smoothing=0.1 (weighted CE was unstable with 48-class imbalance)
- **Best val acc: 34.49%** @ epoch 35 (early stopped at epoch 49)
- **Macro recall: 8.12%** — model overwhelmingly predicts dominant classes
- 48 verbs, chance = ~2.1%

### Contextualized Subtask Classifier (job 6673121)

**Motivation:** Isolated 7–12 frame segments lack context. The contextualized model
sees the entire episode trajectory while classifying each segment individually.

**Architecture:** `ContextualVerbClassifier` in `train_bridge_ctx.py`
- Full episode actions (max_ep_len=64) encoded with learned positional embeddings
- **All action tokens attend to all action tokens** via self-attention (full episode context)
- Per-segment CLS tokens appended after action tokens (max_segments=10)
- **Each CLS_i only attends to its segment's action frames + itself** via 3D attention mask
- One forward pass classifies all segments in an episode simultaneously
- Episode-level train/val split (no data leakage between segments of the same episode)
- d_model=128, 4 layers, 8 heads, unweighted CE, label_smoothing=0.1
- Script: `train_bridge_ctx.py`, SLURM: `scripts/submit_bridge_ctx_train.sh`

**Results:** Val loss was NaN throughout training (best val acc 2.8%).
The 3D attention mask with `-inf` caused NaN propagation in softmax. Fixed by switching to
`-1e9` and ensuring all CLS tokens have self-attention, but the resubmitted job still produced
NaN — likely a deeper interaction between the per-sample 3D mask and PyTorch's TransformerEncoderLayer.
**Status: blocked on NaN bug.**

## Experiment B: High-Level Classification

### Data Pipeline
1. Consolidation: `scripts/consolidate_bridge_actions.py`
2. spaCy verb extraction from episode instructions
3. Two-step verb cleanup:
   - **Lemmatization**: conjugated forms (moved/moves/moving → move), typos (moove/movve → move),
     non-English (abriu, colocou, sacar → dropped), non-verbs (strawberry, front, pan → dropped)
   - **Directional merge**: put down/on/up/in → put, move up/down/in/out → move, etc.
4. **27,271 episodes**, **17 verbs** (min_count ≥ 30)

### t-SNE Embedding Visualization

![Bridge HL t-SNE](../../figures/bridge_hl_tsne.png)

- `sweep`, `open`, `pick up`, `close`, `turn` form clean separable clusters
- `move` and `put` are heavily fragmented — the model learns sub-patterns but can't cleanly separate them
- `fold`/`unfold` cluster together (similar motion, opposite direction)
- Script: `figures/plot_bridge_tsne.py`

## Experiment C: OAT Action Tokenizer

### Training (job 6660959)
- Fitted OAT (Object Action Tokenizer) from scratch on BridgeV2 action data
- Architecture: RegisterEncoder + FSQ quantizer + SinglePassDecoder, 5.8M params
- 32-step chunks → 8 tokens from vocab of 1000
- 427K train chunks, 47K val chunks, batch_size=256, 500 epochs, constant lr=5e-5
- **Final recon MSE: 0.000577** (val), still improving at epoch 500
- No overfitting: val loss consistently below train loss (EMA model)
- Checkpoint: `checkpoints/oat_bridge_j6660959_best.pth`
- Script: `train_oat_bridge.py`, SLURM: `scripts/submit_oat_bridge.sh`

![OAT Bridge Loss](../../figures/oat_bridge_loss.png)

### Representation Comparison: Raw vs OAT Latent vs OAT Discrete

**Goal:** Compare verb decodability across three action representations on the
same 17-verb high-level BridgeV2 task.

| Representation | Description |
|---|---|
| **Raw actions** | 7-d joint velocities per timestep, ~38 tokens/episode |
| **OAT latent** | 4-d continuous FSQ vectors, ~13 tokens/episode (8 per 32-step chunk) |
| **OAT discrete** | Integer token IDs from vocab of 1000, ~13 tokens/episode, fed through learned `nn.Embedding` |

Each representation is paired with its own classifier size sweep to find the
best architecture (different compression levels warrant different model capacities):

| Rep | d_model | Layers | Job |
|-----|---------|--------|-----|
| Raw | 128 | 4 | 6677135 |
| Raw | 256 | 4 | 6677136 |
| Raw | 256 | 6 | 6677137 |
| OAT latent | 64 | 2 | 6677138 |
| OAT latent | 128 | 2 | 6677139 |
| OAT latent | 128 | 4 | 6677140 |
| OAT discrete | 64 | 2 | 6677141 |
| OAT discrete | 128 | 2 | 6677142 |
| OAT discrete | 128 | 4 | 6677143 |

All use: weighted CE, label_smoothing=0.1, weight_decay=0.01, lr=1e-4, patience=15, batch_size=64.
Raw uses max_seq_len=64; OAT uses max_seq_len=32.
Script: `scripts/submit_bridge_rep_sweep.sh`

### Results

| Rep | Config | Val Acc | Macro Recall | Train Acc | Best Epoch |
|-----|--------|---------|-------------|-----------|------------|
| Raw | d128/4L | 24.8% | **62.0%** | 26.0% | 66 |
| Raw | d256/4L | 23.9% | 59.0% | 27.7% | 49 |
| Raw | d256/6L | 24.6% | 56.2% | 29.5% | 55 |
| OAT latent | d64/2L | 11.1% | 39.6% | 9.9% | 80 |
| OAT latent | d128/2L | 12.2% | 41.0% | 12.7% | 75 |
| OAT latent | d128/4L | 14.4% | **44.6%** | 17.5% | 78 |
| OAT discrete | d64/2L | 8.7% | 31.8% | 11.2% | 62 |
| OAT discrete | d128/2L | 9.7% | **32.8%** | 13.5% | 31 |
| OAT discrete | d128/4L | 11.3% | 30.3% | 25.9% | 49 |

![Representation Sweep Curves](../../figures/bridge_rep_sweep_curves.png)

**Best per representation** (selected by highest macro recall across architecture sweep):

| Representation | Best Config | Val Loss | Val Acc | Macro Recall |
|---|---|---|---|---|
| **Raw actions** (7-d × ~38 tokens) | d128 / 4 layers | 3.87 | 24.8% | **62.0%** |
| **OAT latent** (4-d × ~13 tokens) | d128 / 4 layers | 4.23 | 14.4% | **44.6%** |
| **OAT discrete** (1 ID × ~13 tokens) | d128 / 2 layers | 4.46 | 9.7% | **32.8%** |

**Observations:**
- Within each representation, all architecture sizes converge to similar macro recall
  (Raw: 56–62%, Latent: 40–45%, Discrete: 30–33%), justifying that we've let each
  representation show its full potential
- Raw actions dominate: 62% macro recall vs 45% (latent) vs 33% (discrete)
- More compression = worse decodability: OAT quantization discards verb-relevant information
- OAT discrete d128/4L shows loss-accuracy decoupling: val loss rises after epoch 10 while
  val accuracy keeps climbing — weighted CE penalizes rare class errors heavily, so the model
  improves on frequent classes (accuracy) while worsening on rare classes (loss)
- Macro recall curves are noisy due to rare classes (e.g., `cover` has only 6 val samples)
  swinging between epochs under weighted CE

## Data Overview

![Bridge Overview](../../figures/bridge_overview.png)

Key observations:
- **Subtask segments are very short** (mean 9.1 frames) vs high-level episodes (mean 36.7 frames)
- **Both have heavy class imbalance**: subtask top 3 (move/grasp/lift) = 57%, high-level top 2 (move/put) = 65%
- **max_seq_len=64 fits both**: subtask rarely exceeds 20 frames, high-level mostly under 60

## Key Questions

1. **Subtask vs task-level decodability**: Are atomic verbs (grasp, lift, place) easier to
   decode from actions than task verbs (put, move, fold)?
   - Hypothesis: Yes — subtask actions are shorter and more stereotyped
   - But: Emma-X annotations are noisy (Gemini-generated, no human validation)

2. **Cross-dataset comparison**: How does BridgeV2 compare to DROID and CALVIN?
   - CALVIN: 21 verbs, ~40% val acc (task-level)
   - DROID: 228 verbs, ~29% val acc (task-level)
   - BridgeV2 high-level (raw): 17 verbs, 24.8% val acc, 62.0% macro recall
   - BridgeV2 high-level (OAT latent): 17 verbs, 14.4% val acc, 44.6% macro recall
   - BridgeV2 high-level (OAT discrete): 17 verbs, 9.7% val acc, 32.8% macro recall
   - BridgeV2 subtask (isolated): 48 verbs, 34.5% val acc, 8.1% macro recall
   - BridgeV2 subtask (contextualized): blocked (NaN bug)

3. **Does OAT tokenization preserve verb information?**
   - **Answer: No — OAT loses significant verb info.** Raw 62% → Latent 45% → Discrete 33% macro recall.
   - OAT's reconstruction objective (minimize MSE) doesn't prioritize verb-discriminative features.
   - The 32-step → 8-token compression (4× temporal, 7-d → 4-d spatial) discards fine-grained
     motion patterns that distinguish verbs.

4. **Is the subtask vocabulary universal?** Can a verb classifier trained on BridgeV2 subtasks
   transfer to DROID or CALVIN?
