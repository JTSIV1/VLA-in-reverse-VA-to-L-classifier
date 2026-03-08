# VLA-in-Reverse: From Vision-Action to Language

Are action verbs primarily grounded in the goal of the action (state changes in scene_obs), or the kinematic dynamics of the motion (action trajectories)? This project studies verb classification from robotic manipulation trajectories on the
[CALVIN benchmark](https://github.com/mees/calvin), and uses those findings to
build better action tokenizers for Vision-Language-Action (VLA) models.

## Project structure

```
.
├── ── Core (classification pipeline) ─────────────────────────────
├── train_transformer.py        # Model definition + training loop
├── test_transformer.py         # Checkpoint evaluation + metrics
├── config.py                   # Paths, hyperparameters, constants
├── utils.py                    # CALVIN data loading, verb extraction (spaCy)
├── image_encoders.py           # Vision backends: DINOv2-S, VC-1, R3M, scratch
├── explore_data.ipynb          # Interactive data exploration notebook
│
├── ── Action tokenization ────────────────────────────────────────
├── tokenization/               # All action tokenizer code
│   ├── action_tokenizers.py    #   Unified loader (native, FAST, VQ-VAE, VQ-VLA, OAT)
│   ├── action_tokenizers_training.py  #   Tokenizer fitting on CALVIN data
│   ├── fast_tokenizer.py       #   FAST tokenizer (DCT + BPE)
│   ├── vqvae_tokenizer.py      #   VQ-VAE chunk tokenizer + verb-decodable variant
│   ├── vqvla/                  #   Vendored VQ-VLA (causal conv VAE + ResidualVQ)
│   └── oat/                    #   Vendored OAT/QueST tokenizer
│
├── ── OpenVLA experiment (in progress) ───────────────────────────
├── openvla_experiment/         # Verb-decodable tokenizer → OpenVLA-mini
│   └── README.md               #   Experiment design and plan
│
├── ── Artifacts (gitignored) ─────────────────────────────────────
├── checkpoints/                # Model checkpoints + tokenizer weights
├── results/                    # Per-experiment metrics JSON files
├── figures/                    # Confusion matrices + analysis plots
├── logs/                       # SLURM job logs
│
├── ── Archives ───────────────────────────────────────────────────
├── old_run_scripts/            # SLURM submit scripts (all rounds)
├── lab_notebooks/              # Per-round experiment reports (markdown)
├── analysis/                   # Post-hoc analysis + visualization scripts
└── report/                     # LaTeX paper draft
```

## Classification experiment (completed)

A Transformer classifier predicts one of 21 verb classes (e.g., "push", "lift",
"open") from CALVIN manipulation trajectories.

### Key results

| Model | Accuracy | Macro F1 | Active classes |
|-------|----------|----------|----------------|
| Action-only (native) | 39.5% | 38.7% | 21/21 |
| Scene obs (MLP) | 23.1% | 24.2% | 21/21 |
| Action + scene obs (token fusion) | **43.1%** | **41.0%** | 21/21 |

All results use 21 sparse verb classes (`--min_class_count 30`) with weighted
cross-entropy (`--weighted_loss`). See [lab_notebooks/](lab_notebooks/) for
more experiment reports.

### Modality contribution analysis

**Sample-level agreement** (action-only vs scene obs MLP vs token fusion,
N=601 val samples):

| Category | Count | % | Fusion correct |
|----------|------:|--:|---------------:|
| Both correct | 91 | 15.1% | 76 (83.5%) |
| Only action correct | 178 | 29.6% | 90 (50.6%) |
| Only scene correct | 48 | 8.0% | 25 (52.1%) |
| Neither correct | 284 | 47.3% | 68 (23.9%) |

Union unimodal accuracy (either model correct) is 52.7%, well above either
individual model, confirming action and scene obs carry complementary signal.
The fusion model rescues 68 samples (23.9%) where neither unimodal model
succeeds. Per-class patterns:

- **Action dominates**: rotate (100% vs 61%), move (74% vs 5%), pick up
  (43% vs 4%) — kinematic verbs with distinctive motion trajectories
- **Scene obs dominates**: turn on (96% vs 50%), open (71% vs 29%) —
  fixture interactions identifiable from state changes
- **Fusion rescues**: grasp (85% of "neither" rescued), place (63%),
  sweep (65%) — ambiguous verbs where combining signals helps

**NLL decomposition** (action vs scene obs, unique variance explained):

| Component | Nats | % of H(Y) |
|-----------|-----:|----------:|
| Unique action (given scene obs) | 0.989 | 36.9% |
| Unique scene obs (given action) | -0.018 | -0.7% |
| Shared | 0.407 | 15.2% |
| Irreducible | 1.305 | 48.6% |

Action dominates verb prediction at the population level: it explains 37%
of unique variance while scene obs adds no unique information beyond action.
Despite this, the sample-level agreement table shows scene obs is
complementary at the instance level — it correctly classifies different
individual samples than action does, even though it carries no unique
population-level signal. See [analysis/](analysis/) for full scripts and
per-class breakdowns.

### Architecture

```
[CLS] + [scene obs token] + [action tokens] → Transformer → verb
```

- **Action**: native (Linear 7→d), FAST (DCT+BPE), VQ-VAE (chunk MLP), or
  VQ-VLA (causal conv + ResidualVQ)
- **Scene obs**: 24-d scene obs → MLP → 1 token, prepended to action sequence
- **Model**: d=128, 8 heads, 4 layers, dropout=0.1

### Quick start

```bash
conda activate mmml

# Train action-only baseline
python train_transformer.py \
    --modality action_only --action_rep native \
    --min_class_count 30 --weighted_loss \
    --save_path ./checkpoints/example.pth \
    --epochs 100 --batch_size 16 --lr 1e-4 --max_seq_len 64

# Evaluate
python test_transformer.py \
    --model_path ./checkpoints/example_best.pth \
    --save_cm ./figures/example_cm.png \
    --save_metrics ./results/example_metrics.json

# Multimodal with VC-1 vision
python train_transformer.py \
    --modality full --action_rep native \
    --vision_encoder vc1 --num_frames 2 --delta_patches --topk_patches 16 \
    --cross_layers 2 --min_class_count 30 --weighted_loss \
    --save_path ./checkpoints/mm_vc1.pth
```

See [old_run_scripts/run_experiment.sh](old_run_scripts/run_experiment.sh) for
the full SLURM runner with all flags.

## OpenVLA experiment (in progress)

Can a verb-aware action tokenizer improve VLA task performance? Fine-tune the
VQ-VLA tokenizer with an auxiliary verb classification loss, then plug it into
[OpenVLA-mini](https://github.com/Stanford-ILIAD/openvla-mini) and measure
action prediction quality on CALVIN.

See [openvla_experiment/README.md](openvla_experiment/README.md) for the full
experiment design.

## Data

All experiments use **CALVIN split D** (`task_D_D`):
- ~3,300 training / ~660 validation trajectories
- 21 verb classes after sparse filtering (min 30 samples per class)
- 7-DoF relative actions, RGB observations (200x200), language instructions
- Data path configured in [config.py](config.py)

## Environment

```bash
conda create -n mmml python=3.9 -y
conda activate mmml
pip install -r requirements.txt
```

Cluster: SLURM with `--partition=general` (GPU) or `--partition=cpu`.
