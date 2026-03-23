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

On the cluster, the existing `mmml` env is ready to use:

```bash
conda activate mmml
```

### 1. Train an action tokenizer

`tokenization/train_tokenizer.py` is the unified entry point. It supports
gradient-based tokenizers (VQ-BeT, OAT, QueST) and fit-once tokenizers (FAST, bin).

```bash
# VQ-BeT on CALVIN (default dataset)
python tokenization/train_tokenizer.py --tokenizer vq_bet \
    --epochs 200 --batch_size 64 --save_dir checkpoints/vqbet_vanilla

# OAT on CALVIN
python tokenization/train_tokenizer.py --tokenizer oat \
    --epochs 200 --batch_size 64 --save_dir checkpoints/oat_vanilla

# QueST on CALVIN
python tokenization/train_tokenizer.py --tokenizer quest \
    --epochs 200 --batch_size 64 --save_dir checkpoints/quest_vanilla

# VQ-BeT on BridgeV2 (specify shard directory)
python tokenization/train_tokenizer.py --tokenizer vq_bet \
    --dataset bridge --shard_dir /data/user_data/wenjiel2/datasets/bridge_actions \
    --epochs 200 --save_dir checkpoints/vqbet_bridge

# Fit FAST tokenizer (one-off, no gradient training)
python tokenization/train_tokenizer.py --tokenizer fast \
    --save_dir checkpoints/fast_calvin
```

### 2. Add auxiliary losses

Add `--verb_cls_lambda` for a verb classification head and/or `--clip_lambda`
for a contrastive action-language head on the post-quantization latents.
Gradients flow back through the quantizer's straight-through estimator.

```bash
# VQ-BeT + verb classification loss (lambda=0.5)
python tokenization/train_tokenizer.py --tokenizer vq_bet \
    --verb_cls_lambda 0.5 \
    --epochs 200 --save_dir checkpoints/vqbet_verb05

# OAT + CLIP contrastive loss (lambda=1.0)
python tokenization/train_tokenizer.py --tokenizer oat \
    --clip_lambda 1.0 \
    --epochs 200 --save_dir checkpoints/oat_clip10

```


### 3. Probe verb decodability

Train a verb classifier on tokenized actions to measure how much verb
information survives quantization.

```bash
# Probe a trained VQ-BeT tokenizer
python verb_probe/train_verb_probe.py \
    --dataset calvin --modality action_only --action_rep vq_bet \
    --tokenizer_ckpt checkpoints/vqbet_verb05/best.pth \
    --min_class_count 30 --weighted_loss \
    --save_path checkpoints/probe_vqbet_verb05.pth

# Probe native (untokenized) actions as baseline
python verb_probe/train_verb_probe.py \
    --dataset calvin --modality action_only --action_rep native \
    --min_class_count 30 --weighted_loss \
    --save_path checkpoints/probe_native.pth
```

### 4. Train a VLA policy with a tokenizer (working on installing flash-attn)

`policy/train_policy.py` generates and submits SLURM jobs for OpenVLA or
MiniVLA fine-tuning on CALVIN.

```bash
# MiniVLA with vanilla VQ-VLA tokenizer
python policy/train_policy.py --model minivla --tokenizer vqvla_vanilla

# MiniVLA with verb-decodable VQ-VLA tokenizer
python policy/train_policy.py --model minivla --tokenizer vqvla_verb

# OpenVLA with bin tokenizer (LoRA fine-tuning)
python policy/train_policy.py --model openvla --tokenizer bin

# Dry run (print sbatch script without submitting)
python policy/train_policy.py --model minivla --tokenizer oat --dry_run
```

### 5. Evaluate a trained policy (these probably need debugging)

```bash
# Teacher-forcing NLL
python policy/eval_policy.py --mode nll --tokenizer vqvla_verb

# Verb decodability through tokenizer round-trip
python policy/eval_policy.py --mode verb --tokenizer vqvla_verb

# CALVIN rollout (1000 sequences)
python policy/eval_policy.py --mode rollout --tokenizer vqvla_verb

# Attention analysis
python policy/eval_policy.py --mode attention --tokenizer vqvla_verb
```

## Lab notebooks

Experiment details live in `lab_notebooks/`, organized by track:

- `verb_classification/` — 9 rounds of verb classifier experiments (R1–R9)
- `action_tokenizer_cls_head/` — Tokenizer decodability experiments (5 rounds)
- `calvind_sweep/` — CALVIN D tokenizer hyperparameter sweeps
- `bridge_verb_decodability/` — BridgeV2 verb classification and tokenizer sweep
- `action_tokenizer_clip/` — Contrastive action-language alignment
- `syntax_analysis/` — Syntactic decomposition, object-verb coupling
- `action_v_llm_rsm/` — Verb RSM vs LLM representations
- `droid_verb_decodability/` — Scaling to DROID real-world data
- `targeted_paper_experiments/` — Clean comparison tables for paper
