# VLA-in-Reverse: From Vision-Action to Language

Are action verbs grounded in *how* the robot moves (motion dynamics), or *what*
changes in the world (action goal)? This project studies verb decodability from
robotic manipulation data and uses those findings to build better action
tokenizers for Vision-Language-Action (VLA) models.

## Project overview

The project has three tracks:

1. **Verb classification** — Train classifiers to predict verb labels from
   action trajectories, visual observations, and simulator state. Establishes
   which signals carry verb information and how they complement each other.

2. **Action tokenizer decodability** — Measure how well different action
   tokenizers (VQ-VAE, VQ-VLA, FAST, OAT) preserve verb identity. Train
   verb-decodable tokenizers with auxiliary classification/contrastive losses.

3. **VLA fine-tuning** — Plug verb-decodable tokenizers into VLA models
   (OpenVLA-mini, RDT2, MiniVLA) and measure downstream task performance.

### Datasets

| Dataset | Split | Train | Val | Verbs | Action dim | Avg steps |
|---------|-------|------:|----:|------:|-----------:|----------:|
| CALVIN  | D→D   | 3,309 | 665 | 21    | 7          | ~61       |
| CALVIN  | ABCD→D| 15,207| 698 | 22    | 7          | ~61       |
| DROID   | —     | ~44K  | ~8K | ~54   | 7          | ~385      |

CALVIN provides simulator state (`scene_obs`, 24-d) as a privileged goal signal.
DROID is real-world and requires visual change (first/last frame delta) as a
goal proxy.

## Repository structure

```
.
├── Core ──────────────────────────────────────────────────────
├── train_transformer.py          # Verb classifier model + training (CALVIN)
├── train_droid.py                # Verb classifier training (DROID)
├── test_transformer.py           # Checkpoint evaluation + confusion matrices
├── config.py                     # Paths, hyperparameters, constants
├── utils.py                      # Data loading, spaCy verb extraction
├── image_encoders.py             # Vision: DINOv2-S, VC-1, R3M, scratch
│
├── Action tokenization ───────────────────────────────────────
├── tokenization/
│   ├── action_tokenizers.py      # Unified loader (native/FAST/VQ-VAE/OAT/bin)
│   ├── fast_tokenizer.py         # FAST tokenizer (DCT + BPE)
│   ├── vqvae_tokenizer.py        # VQ-VAE chunk tokenizer + verb-decodable variant
│   ├── clip_action_language.py   # Contrastive action-language alignment
│   ├── vqvla/                    # Vendored VQ-VLA (causal conv VAE + ResidualVQ)
│   └── oat/                      # Vendored OAT/QueST tokenizer
│
├── VLA fine-tuning ───────────────────────────────────────────
├── openvla_experiment/           # OpenVLA-mini fine-tuning on CALVIN
│   ├── scripts/                  # Training, evaluation, probing scripts
│   ├── tfds_builders/            # CALVIN → RLDS/TFDS converter
│   └── data_conversion/          # Raw data format converters
│
├── Analysis ──────────────────────────────────────────────────
├── analysis/                     # Post-hoc analysis modules
│   ├── compute_verb_rsm.py       # Verb representation similarity matrices
│   ├── unique_variance.py        # Information-theoretic decomposition
│   ├── cluster_gather_results.py # Result aggregation across experiments
│   ├── sklearn_*.py              # Baseline classifiers (RF, MLP)
│   └── ...
│
├── Scripts ────────────────────────────────────────────────────
├── scripts/
│   ├── build_*.py                # Dataset builders (task types, L1 segments)
│   ├── extract_droid_actions.py  # DROID RLDS → compact action .npz
│   ├── download_droid_shards.sh  # Full DROID download (2TB)
│   ├── submit_droid_*.sh         # Active SLURM job scripts
│   ├── train_scene_mlp.py        # Specialized trainers (scene MLP, L1)
│   ├── run_cluster_*.sh          # Generic SLURM launchers
│   └── archived/                 # Old round-specific submit scripts (R1–R9)
│
├── Data ───────────────────────────────────────────────────────
├── data/
│   ├── verb_classes.txt          # 21 sparse verb vocabulary
│   ├── episode_task_types.csv    # CALVIN task type labels
│   ├── episode_abcd_d.csv        # CALVIN ABCD→D annotations
│   ├── embeddings/               # Cached verb/instruction embeddings
│   ├── l1_segments/              # Phase-level verb segments
│   └── hierarchy_annotations/    # Gemini-annotated task hierarchy
│
├── Documentation ─────────────────────────────────────────────
├── lab_notebooks/                # Experiment reports (see below)
│
└── Outputs (gitignored) ──────────────────────────────────────
    ├── checkpoints/              # Model weights
    ├── results/                  # Per-experiment metrics JSON
    ├── figures/                  # Plots + confusion matrices
    └── logs/                     # SLURM job logs
```

## Key results (CALVIN D→D)

### Verb classification

| Model | Accuracy | Macro F1 | Active |
|-------|----------|----------|--------|
| Action-only (native) | 39.5% | 38.7% | 21/21 |
| Vision-only (VC-1 delta16) | 38.9% | 36.2% | 20/21 |
| Multimodal (VC-1 late2) | 42.4% | 40.7% | 20/21 |
| Scene obs (Random Forest) | 48.4% | — | — |
| Action + scene obs (token fusion) | **43.1%** | **41.0%** | 21/21 |

All transformer results use 21 sparse verb classes (`--min_class_count 30`)
with weighted cross-entropy (`--weighted_loss`).

### Motion vs goal

Action trajectories (motion) and scene state changes (goal) carry
**complementary** verb information:
- **Motion dominates**: rotate, move, pick up — distinctive trajectories
- **Goal dominates**: turn on, turn off, open, close — distinctive state changes
- **Fusion rescues**: grasp, place, sweep — ambiguous without both signals

### Action tokenizer verb decodability

| Tokenizer | Train verb acc | Codebook usage |
|-----------|---------------|----------------|
| VQ-VAE vanilla (lambda=0) | — | 15/512 (2.9%) |
| VQ-VAE verb (lambda=0.1) | **51.0%** | 73/512 (14.3%) |
| VQ-VLA fine-tuned (lambda=0.5) | 44.9% val | 256/256 |

Adding a verb classification loss to VQ-VAE training improves codebook
utilization 5x and produces verb-decodable codes without sacrificing
reconstruction quality.

## Lab notebooks

All experiment details are in `lab_notebooks/`, organized by track:

### Verb classification (9 rounds)
| Round | Focus | Key finding |
|-------|-------|-------------|
| R1 | Baselines | Vision-only collapses with scratch encoder |
| R2 | R3M + FAST | AO native sp+wt = 39.5% baseline established |
| R3 | Patch vision (DINOv2-S, VC-1) | VC-1 delta16 first vision model to match AO |
| R3b | FAST scale x vocab sweep | DCT quantization loss is bottleneck, not seq length |
| R3c | Oracle (scene_obs, robot_obs) | scene_obs RF=48.4%; transformer wrong arch for tabular |
| R4 | Multimodal fusion | VC-1 late2 = 42.4% best multimodal |
| R5 | MM ablations (d256, K49, mdrop) | No improvement; capacity mismatch is fundamental |
| R6 | MM tokenizer sweep | Vision fusion rescues FAST but not VQ-VLA |
| R7 | Scene obs + action fusion | Scene token + native = 43.1% best overall |
| R8 | Verb granularity taxonomy | Fixture verbs easiest (counter-intuitive) |
| R9 | L0 vs L1 decodability | Full-episode vs phase-segment comparison |

### Action tokenizer decodability (5 rounds)
| Round | Focus |
|-------|-------|
| R1 | VQ-VAE + verb loss lambda sweep |
| R2 | OpenVLA-mini fine-tuning with verb-decodable tokenizer |
| R3 | Attention analysis of verb grounding |
| R4 | RDT2-VQ fine-tuning (temporal token structure) |
| R5 | MiniVLA fine-tuning (0.5B model) |

### Other tracks
- **action_tokenizer_clip/** — Contrastive action-language alignment (2 rounds)
- **syntax_analysis/** — Syntactic decomposition, object-verb coupling (3 rounds)
- **action_v_llm_rsm/** — Verb RSM vs LLM representations (2 rounds)
- **droid_verb_decodability/** — Scaling to DROID real-world data (in progress)
- **targeted_paper_experiments/** — Clean comparison tables for paper

## Quick start

```bash
conda activate mmml

# Train action-only verb classifier (CALVIN)
python train_transformer.py \
    --modality action_only --action_rep native \
    --min_class_count 30 --weighted_loss \
    --save_path checkpoints/ao_native.pth \
    --epochs 100 --batch_size 16 --lr 1e-4 --max_seq_len 64

# Evaluate checkpoint
python test_transformer.py \
    --model_path checkpoints/ao_native_best.pth \
    --save_cm figures/ao_native_cm.png \
    --save_metrics results/ao_native_metrics.json

# Multimodal with VC-1 vision + late cross-attention
python train_transformer.py \
    --modality full --action_rep native \
    --image_encoder vc1 --num_frames 2 --delta_patches 16 \
    --cross_layers 2 --min_class_count 30 --weighted_loss \
    --save_path checkpoints/mm_vc1.pth

# Train on DROID (after data extraction)
python train_droid.py \
    --min_class_count 30 --weighted_loss \
    --max_seq_len 512 --epochs 30 \
    --save_path checkpoints/droid_ao.pth
```

## Architecture

```
[CLS] + [vision patches (optional)] + [action tokens] --> Transformer --> verb
```

- **Actions**: native (Linear 7->d), FAST (DCT+BPE), VQ-VAE, VQ-VLA, OAT, QueST
- **Vision**: VC-1 / DINOv2-S delta patches (top-K changed between frames)
- **Scene obs**: 24-d simulator state -> MLP -> 1 token (CALVIN only)
- **Fusion**: late cross-attention in final K transformer layers
- **Model**: d_model=128, 8 heads, 4 layers, dropout=0.1

## Environment

```bash
conda create -n mmml python=3.9 -y
conda activate mmml
pip install -r requirements.txt
```

Cluster: SLURM with `--partition=general` (GPU) or `--partition=cpu`.
