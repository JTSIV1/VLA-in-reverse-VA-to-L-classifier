# CALVIN Action-to-Verb Transformer

Predict the primary action verb (e.g., "push", "lift", "open") from a robotic manipulation trajectory using the **CALVIN Dataset**. The model is a multimodal Transformer that fuses:

- **First & last video frames** — ViT-style patch embeddings (no pretrained CNN; learned from scratch on 200x200 CALVIN renders)
- **7D relative action trajectory** — either native continuous vectors or **FAST** discrete tokens (DCT + BPE)

Verbs are extracted automatically from CALVIN's natural language instructions via spaCy.

## Architecture

The Transformer encoder processes a sequence of:
```
[CLS] [start_frame patches × 64] [end_frame patches × 64] [action tokens × T]
```

Key design choices:
- **Patch embeddings** (Conv2d, 25×25 patches) preserve spatial info lost by ResNet global avg pool
- **Token type embeddings** (CLS / start-frame / end-frame / action) let the model distinguish modalities, like BERT segment embeddings
- **Separate positional embeddings** per modality: spatial for image patches, temporal for actions
- **d_model=128**, 8 heads (16 dims/head), 4 layers
- Classification from the CLS token via a 2-layer MLP head

## Setup

```bash
# Create environment
conda create -n mmml python=3.9 -y
conda activate mmml
pip install -r requirements.txt
```

The CALVIN dataset should be at the path configured in `config.py` (default: `/data/user_data/yashagar/task_D_D/`) with `training/` and `validation/` subdirectories.

## Experiments

### Modality ablations

We compare three input configurations to measure each modality's contribution:

| Variant | Flag | Description |
|---|---|---|
| **Full** (default) | `--modality full` | First/last frames + action trajectory |
| **Action-only** | `--modality action_only` | Action trajectory only, no images |
| **Vision-only** | `--modality vision_only` | First/last frames only, no actions |

### Action representations

| Representation | Flag | Description |
|---|---|---|
| **Native** (default) | `--action_rep native` | Raw 7D relative actions, projected via Linear |
| **FAST** | `--action_rep fast` | DCT + BPE discrete tokens (arxiv 2501.09747), embedded via nn.Embedding |

## Usage

### 1. Train the full model

```bash
# Single experiment
python train_transformer.py \
    --modality full --action_rep native \
    --save_path ./checkpoints/model.pth \
    --epochs 10 --batch_size 16 --lr 1e-4 --max_seq_len 64

# Or submit to SLURM
sbatch run.sh
```

### 2. Run all baselines

```bash
# Runs: FAST tokenizer fitting, action-only, vision-only, full+FAST, action-only+FAST
sbatch run_baselines.sh

# Smoke test with 32 samples
sbatch run_baselines.sh --debug 32
```

Each experiment saves its checkpoint, confusion matrix PNG, and metrics JSON to `./checkpoints/`. Results are saved incrementally so you can inspect them while later experiments run.

### 3. Evaluate a checkpoint

```bash
python test_transformer.py \
    --model_path ./checkpoints/model.pth \
    --save_cm ./checkpoints/cm.png \
    --save_metrics ./checkpoints/metrics.json
```

The modality and action_rep are read from the checkpoint automatically.

### 4. FAST tokenizer

```bash
# Fit on training data (required before --action_rep fast)
python fast_tokenizer.py --save_path ./checkpoints/fast_tokenizer

# Debug with fewer samples
python fast_tokenizer.py --debug 100
```

### 5. Exploratory analysis

`explore_data.ipynb` provides interactive visualizations:
1. Language annotations DataFrame with verb extraction
2. Verb distribution bar chart
3. Episode `.npz` structure inspection
4. First/last frame visualization
5. 7D action trajectory plots
6. Sequence length distribution per verb (box plots)

### 6. Cluster analysis

```bash
python cluster_analysis.py --max_len 64 --out_dir ./checkpoints
# Or: sbatch run_cluster.sh
```

PCA + K-Means on flattened action trajectories, with ARI/NMI metrics.

## File Structure

```
├── config.py                # Central configuration (paths, hyperparameters)
├── train_transformer.py     # Training script (model, dataset, training loop)
├── test_transformer.py      # Evaluation (per-class metrics, confusion matrix)
├── fast_tokenizer.py        # FAST tokenizer (DCT + BPE, fit/load/tokenize)
├── utils.py                 # Data loading, NLP verb extraction, visualization
├── explore_data.ipynb       # Interactive data exploration notebook
├── cluster_analysis.py      # PCA + K-Means cluster analysis
├── run.sh                   # SLURM script: train + test (full model)
├── run_baselines.sh         # SLURM script: all modality/action-rep ablations
├── run_cluster.sh           # SLURM script: cluster analysis
└── requirements.txt         # Python dependencies
```

## Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | config.DATA_DIR | Path to CALVIN training split |
| `--val_dir` | config.VAL_DIR | Path to CALVIN validation split |
| `--modality` | `full` | `full`, `action_only`, or `vision_only` |
| `--action_rep` | `native` | `native` (continuous 7D) or `fast` (discrete tokens) |
| `--batch_size` | 16 | Batch size |
| `--epochs` | 10 | Training epochs |
| `--lr` | 1e-5 | AdamW learning rate |
| `--max_seq_len` | 128 | Max action sequence length (pad/truncate) |
| `--save_path` | None | Checkpoint save path |
| `--debug N` | 0 | Use only N samples for smoke testing |


## Clustering

Run image clustering:
```bash
RUN_MODE=images bash /home/istepka/11777/run_cluster.sh
```

Run action clustering:
```bash
RUN_MODE=actions bash /home/istepka/11777/run_cluster.sh
```