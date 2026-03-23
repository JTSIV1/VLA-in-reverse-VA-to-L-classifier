# VLA-in-Reverse: From Vision-Action to Language

Are action verbs grounded in *how* the robot moves (motion dynamics), or *what*
changes in the world (action goal)? This project studies verb decodability from
robotic manipulation data and uses those findings to build better action
tokenizers for Vision-Language-Action (VLA) models.

## Datasets

| Dataset | Split | Train | Val | Verbs | Action dim | Avg steps | Path on cluster |
|---------|-------|------:|----:|------:|-----------:|----------:|-----------------|
| CALVIN  | D→D   | 3,309 | 665 | 21    | 7          | ~61       | `/data/user_data/yashagar/task_D_D/` |
| CALVIN  | ABCD→D| 15,207| 698 | 22    | 7          | ~61       | `/data/user_data/wenjiel2/datasets/task_ABCD_D/` |
| CALVIN RLDS | D→D | 5,124 | 1,011 | — | 7 | ~61 | `/data/user_data/wenjiel2/datasets/calvin_rlds/` |
| BridgeV2| —     | ~44K  | ~8K | ~25   | 7          | varies    | `/data/user_data/wenjiel2/datasets/bridge_actions/` |
| DROID   | —     | ~44K  | ~8K | ~54   | 7          | ~385      | `/data/user_data/wenjiel2/datasets/droid_rlds/` |

All dataset directories are world-readable (755).

## Repository structure

The four core directories are **`datasets/`**, **`tokenization/`**,
**`verb_probe/`**, and **`policy/`**. 

### `datasets/` — Data loading and building

PyTorch Dataset classes for CALVIN, BridgeV2, and DROID, plus scripts to build
derived datasets (Gemini annotation, RLDS/TFDS conversion).

| File | Purpose |
|------|---------|
| `calvin_dataset.py` | `CalvinVerbProbeDataset`, `CalvinTokenizerDataset`, `CalvinActionCropDataset` |
| `bridge_dataset.py` | `BridgeVerbDataset` for BridgeV2 subtask classification |
| `build_gemini_dataset.py` | Build L0/L1 verb-labeled datasets from Gemini annotations |
| `annotate_calvin_hierarchy.py` | Gemini VLM hierarchical episode annotation |
| `build_episode_task_types.py` | Episode classification (fixture/block/etc.) |
| `tfds_builders/` | CALVIN → RLDS/TFDS converters (D and ABCD splits) |
| `tfrecord_parser.py` | TFRecord parser (no TF dependency) |

### `tokenization/` — Action tokenizers

Unified interface for training and loading action tokenizers. Supports VQ-BeT,
OAT, QueST, FAST, VQ-VAE, VQ-VLA, and bin tokenizers.

| File | Purpose |
|------|---------|
| `train_tokenizer.py` | **Unified training/fitting** for all tokenizer types + aux losses |
| `action_tokenizers.py` | `TokenizerAdapter` — uniform `adapter(actions) -> token_ids` interface |
| `vqbet_tokenizer.py` | VQ-BeT (MLP + ResidualVQ) |
| `vqvae_tokenizer.py` | VQ-VAE chunk tokenizer |
| `fast_tokenizer.py` | FAST (DCT + BPE), vendored for Python 3.9 |
| `codebook_utilization.py` | Codebook usage analysis |
| `oat/` | Vendored OAT/QueST (RegisterEncoder + FSQ) |
| `vqvla/` | Vendored VQ-VLA (causal conv VAE + ResidualVQ) |
| `vq_bet_official/` | Vendored VQ-BeT reference implementation |

### `verb_probe/` — Verb classification

Transformer-based verb classifiers that test whether action representations
(raw or tokenized) preserve verb identity.

| File | Purpose |
|------|---------|
| `train_verb_probe.py` | **Unified entry point** for CALVIN / Bridge / DROID probes |
| `models.py` | `ActionToVerbTransformer`, vision encoders |
| `image_encoders.py` | Pluggable vision encoders (DINOv2, VC-1, R3M, scratch) |
| `training.py` | Shared training loop, criterion, optimizer, checkpointing |
| `train_bridge_ctx.py` | Contextualized per-segment classifier for BridgeV2 |
| `train_scene_mlp.py` | MLP baseline on oracle scene_obs features |
| `test_transformer.py` | Checkpoint evaluation + confusion matrices |

### `policy/` — VLA fine-tuning

Fine-tune OpenVLA or MiniVLA on CALVIN with different action tokenizers,
then evaluate with NLL, verb probing, or simulator rollouts.

| File | Purpose |
|------|---------|
| `train_policy.py` | **Unified launcher** for OpenVLA / MiniVLA training (generates SLURM jobs) |
| `eval_policy.py` | **Unified launcher** for evaluation (NLL, verb, rollout, attention) |
| `scripts/evaluate_openvla.py` | NLL + verb probe implementation |
| `scripts/evaluate_openvla_rollout.py` | CALVIN rollout (1000 sequences, SR1–SR5) |
| `scripts/analyze_attention.py` | Action→verb attention analysis (deprecated atm)|
| `scripts/build_calvin_*.sh` | TFDS dataset build scripts |
| `README.md` | Detailed pipeline guide |

### Supporting directories

| Directory | Purpose |
|-----------|---------|
| `analysis/` | Post-hoc analysis (RSMs, variance decomposition, clustering, ablations) |
| `scripts/` | SLURM launchers for sweeps (`submit_calvind_*.sh`, `submit_bridge_*.sh`) |
| `figures/` | Plot generation (sweeps, t-SNE, confusion matrices) |
| `data/` | Static data files (verb vocab, episode CSVs, Gemini annotations) |
| `lab_notebooks/` | Experiment reports organized by track |
| `config.py` | Paths, hyperparameters, constants |
| `utils.py` | Data loading, spaCy verb extraction, shared helpers |

## Usage

### Environment

```bash
conda create -n mmml python=3.9 -y
conda activate mmml
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
