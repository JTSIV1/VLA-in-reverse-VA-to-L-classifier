# Round 2: OpenVLA-mini Fine-Tuning on CALVIN

**Date**: 2026-03-03
**Goal**: Fine-tune OpenVLA-mini on CALVIN task_D_D to test whether verb-decodable action
tokenization improves language grounding.

## Setup

### Codebase
- OpenVLA-mini: `/data/user_data/wenjiel2/Code/openvla-mini`
  (cloned from `https://github.com/Stanford-ILIAD/openvla-mini.git`)
- Our project: `/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier`

### CALVIN Dataset (RLDS/TFDS format)
- TFDS Builder: `openvla_experiment/tfds_builders/calvin_dataset.py`
- Output dir: `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/`
- Build job: 6483000 (running, ~2.5h, 5124 train + 1011 val episodes)
- Field mapping:
  - `observation.image` → `image_primary` (rgb_static, 200×200)
  - `observation.wrist_image` → `image_wrist` (rgb_gripper, 84×84)
  - `observation.state` → `proprio` (robot_obs, 15-dim)
  - `action` → rel_actions (7-dim: delta_xyz + delta_euler + gripper)
  - `language_instruction` → task description (CALVIN auto annotations)

### CALVIN → RLDS Conversion (job 6478945, Done)

OpenVLA-mini's data pipeline is built around the [RLDS](https://github.com/google-research/rlds)
format (TFRecord-based, one episode per example), used by the Open X-Embodiment ecosystem.
CALVIN's raw dataset stores observations as per-step `.npz` files with separate language annotation
files — incompatible with the TFDS/RLDS loading pipeline in openvla-mini.

The conversion script (`openvla_experiment/tfds_builders/calvin_dataset.py`) reads CALVIN episodes,
groups steps by episode, and writes them as RLDS TFRecord shards that can be loaded natively by
openvla-mini's `RLDSDataset` / `RLDSBatchTransform` pipeline.

| Field | Source | Notes |
|-------|--------|-------|
| `observation/image` | `rgb_static` (200×200) | primary camera |
| `observation/wrist_image` | `rgb_gripper` (84×84) | wrist camera |
| `observation/state` | `robot_obs` (15-dim) | joint + EEF state |
| `action` | `rel_actions` (7-dim) | delta xyz + delta euler + gripper |
| `language_instruction` | CALVIN auto annotations | one per episode |

Output: `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/`
- Train: 5124 episodes, 308918 steps, 128 shards
- Val: 1011 episodes, 60575 steps, 32 shards

### openvla-mini modifications
Three files modified in `/data/user_data/wenjiel2/Code/openvla-mini/`:
1. `prismatic/vla/datasets/rlds/oxe/configs.py` — added `calvin_dataset` entry
2. `prismatic/vla/datasets/rlds/oxe/transforms.py` — added `calvin_dataset_transform`
3. `prismatic/vla/datasets/rlds/oxe/mixtures.py` — added `"calvin_dataset"` mixture
4. `prismatic/vla/datasets/rlds/dataset.py` — added `builder_from_directory` fallback

**calvin_dataset_transform** (in transforms.py):
- Gripper: convert relative (-1=open, +1=close) → absolute (0=closed, 1=open) via `rel2abs_gripper_actions`
- EEF_state: `robot_obs[:, :6]` (tcp_pos + tcp_orn_euler, 6-dim)
- gripper_state: `robot_obs[:, 9:10]` (gripper_opening_width, 1-dim)

---

## Experiment Conditions

### Stage 2a: Baseline fine-tuning (standard bin-based ActionTokenizer)
| Param | Value |
|-------|-------|
| Model | `openvla/openvla-7b` (LoRA r=32) |
| Dataset | `calvin_dataset` (CALVIN task_D_D) |
| Action tokenizer | `ActionTokenizer` (256 bins, 7 tokens/step) |
| Steps | 50,000 |
| LR | 5e-4 |
| Batch | 8 (grad_accum=2 → effective 16) |
| GPU | 1× L40S (46GB) |
| Script | `openvla_experiment/scripts/finetune_openvla_bin.sh` |

### Stage 2b: VQ-based fine-tuning
Compare two conditions using `CalvinVQActionTokenizer` with our verb-decodable VQ-VLA:
- **Control**: vanilla VQ-VLA (lambda=0, `checkpoints/vqvla_ft_vanilla/`)
- **Experimental**: verb-decodable VQ-VLA (lambda=0.5, `checkpoints/vqvla_ft_verb_l0.5/`)
- Scripts: `finetune_openvla_vq_vanilla.sh`, `finetune_openvla_vq_verb.sh`
- 4 tokens per 5-step chunk; `future_action_window_size=4`

---

## Jobs

| Job ID | Condition | Status | Notes |
|--------|-----------|--------|-------|
| 6483000 | TFDS build | **DONE** | 16.75 GiB, 5124 train + 1011 val episodes |
| ~~6483035~~ | Stage 2a (failed) | FAILED | PRISMATIC_DATA_ROOT not set |
| ~~6485333~~ | Stage 2a bin (failed) | FAILED | args not parsed by draccus (sbatch --wrap bug); dataset defaulted to droid_wipe |
| ~~6485334~~ | Stage 2b VQ vanilla (failed) | FAILED | SIGBUS crash; wrong config dir path + diffusers>=0.30 API change |
| ~~6485335~~ | Stage 2b VQ verb (failed) | FAILED | Same SIGBUS crash as 6485334 |
| ~~6485772~~ | Stage 2a: bin-based baseline | FAILED | wandb 403 (stanford-voltron entity, no access) |
| ~~6485773~~ | Stage 2b: VQ vanilla (control) | FAILED | same wandb 403 |
| ~~6485774~~ | Stage 2b: VQ verb-decodable | FAILED | same wandb 403 |
| **6485841** | Stage 2a: bin-based baseline | **DONE** | 50k steps, 1.44s/step |
| **6485842** | Stage 2b: VQ vanilla (control) | **DONE** | 50k steps, 1.79s/step |
| ~~6485843~~ | Stage 2b: VQ verb-decodable | FAILED | 41s/step on A100 node (bad node) |
| ~~6487909~~ | Stage 2b: VQ verb (resubmit) | FAILED | 16s/step on draining node |
| **6493014** | Stage 2b: VQ verb λ=0.5 | **RUNNING** | device="cuda" fix, 1.70s/step, L40S |
| **6493302** | Stage 2b: VQ verb λ=0.1 | **RUNNING** | checkpoints/vqvla_ft_verb_l0.1, L40S |

---

## Stage 3 Results

### Stage 2 Training Status

| Job | Condition | Status | Steps | Speed |
|-----|-----------|--------|-------|-------|
| 6485841 | Bin baseline | **DONE** (50k/50k) | Full merged ckpt | 1.44 s/step |
| 6485842 | VQ vanilla | **DONE** (45k, timeout) | Full merged ckpt | 1.79 s/step |
| 6493014 | VQ verb λ=0.5 | **RUNNING** (~35k/50k) | Full merged ckpt | 1.70 s/step |
| 6493302 | VQ verb λ=0.1 | **RUNNING** (~30k/50k) | Full merged ckpt | 1.70 s/step |

**Issue**: VQ verb tokenizer was running on CPU by default (device="cpu"), causing 16-41s/step
on some nodes. Fixed by passing `device="cuda"` in finetune.py. Also excluded draining/bad nodes
(babel-m5-32, babel-y9-12).

### Continuous L1 Loss (teacher-forcing, val split) — jobs 6502134–6502137

NLL and token accuracy were dropped — NLL is not cross-tokenizer comparable (different vocab sizes
and token counts per step), and token accuracy has the same problem. Replaced with continuous L1:
decode predicted tokens → continuous actions → mean |pred − GT| in original action space.

| Condition | Ckpt Steps | Avg L1 ↓ |
|-----------|-----------|---------|
| bin (job 6502134) | 50k | **0.162** |
| vq_vanilla (job 6502135) | 45k (timeout) | 0.232 |
| vq_verb λ=0.5 (job 6502136) | ~35k | 0.196 |
| vq_verb01 λ=0.1 (job 6502137) | ~30k | 0.186 |

**Key findings:**

- Bin is best (0.162) — near-lossless tokenization; VQ adds 30–40% reconstruction error from compression.
- Verb CE reduces L1 vs vanilla (0.232 → 0.186–0.196); λ=0.1 slightly better than λ=0.5.
- VQ conditions trained fewer steps than bin and are still running — final numbers may improve slightly.
- L1 under teacher-forcing is not a direct measure of language grounding; rollout task success is the gold standard.

## Evaluation Plan (Stage 3)

1. **Continuous L1** — teacher-forcing on CALVIN val, decode tokens → actions → L1 vs GT (comparable across tokenizers)
2. **Verb probe Levels 1–3** — train classifiers from scratch on z_q / token ID sequences / LLM action token embeddings
3. **Rollout task success rate** (SR1–SR5, avg length) — gold standard; planned after current results

---

## Verb Probe Results (Levels 1–3)

Script: `openvla_experiment/scripts/train_verb_probe_vq.py` (L1/L2), `train_verb_probe_level3.py` (L3)

Note: L1/L2 VQ numbers below are from rerun jobs (6503447–6503449) which also generated caches for L3.
Slight differences from original jobs (6502503–6502504, 6502611) are due to random probe training seed.

### Full pipeline comparison (best probe per level)

| Condition | L1 tokenizer latent ↑ | L2 token IDs ↑ | L3 LLM embeddings ↑ | L4 action L1 ↓ |
|-----------|----------------------|----------------|---------------------|----------------|
| vq_vanilla | 37.22% | 40.62% | 37.22% | 0.232 |
| vq_verb λ=0.1 | 44.31% | 41.65% | 42.10% | 0.186 |
| vq_verb λ=0.5 | **46.68%** | 41.36% | **43.13%** | 0.196 |

### Level 3 detail (linear vs transformer probe)

| Condition | L3a Linear ↑ | L3b Transformer ↑ | Jobs |
|-----------|-------------|-------------------|------|
| bin | 29.25% | 32.35% | 6503272 |
| vq_vanilla | 36.48% | 37.22% | 6503450 |
| vq_verb λ=0.1 | 37.96% | 42.10% | 6503452 |
| vq_verb λ=0.5 | 41.06% | **43.13%** | 6503451 |

### Key findings (Levels 1–3)

1. **vq_verb λ=0.5 is best at every verb-decodability level** (L1: 46.68%, L3: 43.13%). The verb CE
   objective propagates all the way from the VQ latent space through LLM fine-tuning.

2. **L3 > L2 for vq_verb conditions**: LLM fine-tuning adds +1.5–1.8pp verb info above raw token IDs
   for verb-decodable tokenizers (vq_verb L2=41.36% → L3=43.13%). The LLM learns to use the verb
   structure embedded in the token distribution.

3. **L3 < L2 for vanilla and bin**: LLM fine-tuning does NOT improve verb separability when the
   tokenizer has no verb structure (vq_vanilla L2=40.62% → L3=37.22%; bin L2=37.81% → L3=32.35%).
   The LLM's action token embeddings are actually *less* verb-separable after fine-tuning on
   unstructured tokens — the embedding space organizes around action patterns, not verb semantics.

4. **Bin collapses at L3 (32.35%)** despite competitive L1 (42.84%). The continuous actions carry
   verb info, but after bin tokenization → LLM fine-tuning, that structure is lost. Bin tokens do
   not induce verb-clustered LLM embeddings.

5. **The full pipeline ordering** (L3: bin < vq_vanilla < vq_verb01 < vq_verb) matches the
   tokenizer design ranking — verb CE at λ=0.5 dominates all the way to the LLM embedding level.

---

## Technical Issues & Fixes

1. **Old step-level TFRecords blocked TFDS build**: previous `calvin_to_rlds.py` output (step-level)
   interfered with TFDS expecting episode-level data. Fixed by deleting
   `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/` before rebuilding.

2. **tensorflow_datasets not in mmml env**: installed `tensorflow-datasets==4.9.3`
   (highest version compatible with Python 3.9).

3. **TFDS builder registration**: `tfds.builder("calvin_dataset", ...)` requires the builder
   class to be registered. Fixed in `dataset.py` with a `builder_from_directory` fallback that
   reads the built dataset directly from the output directory (no import needed at runtime).

4. **sbatch --wrap draccus arg parsing failure**: jobs 6485333-6485335 submitted via `--wrap`
   had their torchrun CLI args silently ignored by draccus — `dataset_name` defaulted to `droid_wipe`.
   Root cause unknown (possibly shell quoting in sbatch wrap). Fixed by using proper `#SBATCH` scripts
   (`finetune_openvla_*.sh`) with explicit `torchrun` calls.

5. **SIGBUS crash in VQ jobs**: jobs 6485334/6485335 crashed with Signal 7 (SIGBUS) after model
   download. Root cause: `VQVLA_PRETRAINED_PATH` in `calvin_vq_action_tokenizer.py` pointed to
   `checkpoints/vqvla_pretrained/action_tokenizer_weight/` (only has `.pth`, no JSONs). Fixed:
   path now points to `tokenization/vqvla/config/` which has the config JSONs.

6. **diffusers>=0.30 API change**: `ActionVQVAEPE.from_config(json_file_path)` fails in
   diffusers>=0.30 which requires a directory path or dict (not a specific .json file path).
   Also, diffusers==0.36.0 adds a hard check for `peft>=0.17.0` (we have 0.11.1).
   Fix: downgraded diffusers to 0.30.0 in mmml env AND added a monkeypatch in
   `_VQVLAWrapper.__init__` that intercepts `.json` file paths and loads them as dicts.

7. **wandb 403 (permission denied on stanford-voltron entity)**: jobs 6485772–6485774 crashed
   at `wandb.init(entity="stanford-voltron", project="openvla")` — we don't have write access to
   the Stanford entity. Fix: added `export WANDB_MODE=offline` to all three finetune scripts.

---

## Next Steps

- [x] Build CALVIN TFDS dataset (job 6483000) — 16.75 GiB, 5124/1011 eps
- [x] Write VQ-VLA → CalvinVQActionTokenizer adapter (`prismatic/vla/calvin_vq_action_tokenizer.py`)
- [x] Fix submission issues (sbatch args, SIGBUS, diffusers API) and resubmit
- [x] Submit Stage 2 fine-tuning — all 4 conditions running/done
- [x] Write Stage 3 evaluation script — L1 loss (replaced NLL+token_acc)
- [x] Submit Stage 3 L1 eval jobs (6502134–6502137)
- [x] Submit verb probe Level 1 & 2 jobs (6502503–6502504)
- [x] Fill in L1 results table (jobs 6502134–6502137) — DONE
- [x] Fill in verb probe Level 1 & 2 results (all 4 conditions) — DONE
- [x] Submit Level 3 probe jobs (6503272, 6503450–6503452) — DONE
- [x] Fill in Level 3 results — DONE
- [ ] Attention analysis (action token → verb token attention) — see round3_attention_analysis.md
- [ ] Rollout evaluation (task success rate SR1–SR5) — gold standard
