# CALVIN Action-to-Verb Transformer

This repository contains a PyTorch transformer pipeline designed to predict the primary action verb (e.g., "push", "lift", "open") from a robotic trajectory sequence and first and last video frames using the **CALVIN Dataset**.

It uses a multimodal Transformer that takes in the starting visual state, the sequence of 7D kinematic actions, and the ending visual state, mapping them to a verb extracted automatically from CALVIN's natural language instructions using **spacy**.

## Installation & Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

To train the model, point the script to your data directory.

```bash
python train_transformer.py \
    --data_dir /path/to/calvin_data \
    --save_path ./checkpoints/model.pth \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --max_seq_len 64
```

**Training Arguments:**

* --data_dir: (Required) Path to the CALVIN dataset directory.
* --save_path: (Optional) Where to save the .pth weights.
* --batch_size: Batch size (default: 16).
* --epochs: Number of training epochs (default: 10).
* --lr: Learning rate (default: 1e-5).
* --max_seq_len: Maximum number of action steps to pad/truncate (default: 128).

### 2. Testing / Evaluation

```bash
python test_transformer.py \
    --data_dir /path/to/test_data \
    --model_path ./checkpoints/model.pth \
    --batch_size 16
```

### 3. Exploratory Analysis (Notebook)

`explore_data.ipynb` provides an interactive walkthrough of the dataset:

1. Load language annotations into a DataFrame and extract verbs with spaCy
2. Verb distribution across the training split
3. Inspect `.npz` episode structure (actions, images, depth, etc.)
4. Visualize first/last frames for sampled episodes
5. Plot 7D action trajectories over time
6. Sequence length distribution per verb

### 4. Cluster Analysis

`cluster_analysis.py` extends the notebook analysis as a standalone script suitable for compute nodes. It pads/truncates each episode's action trajectory to a fixed length, flattens it into a feature vector, then runs PCA + K-Means to evaluate whether trajectories naturally cluster by verb.

```bash
# Run locally
python cluster_analysis.py --max_len 64 --out_dir ./checkpoints

# Submit to SLURM
sbatch run_cluster.sh
```

Outputs saved to `checkpoints/`:
- `pca_trajectories.png` — PCA scatter plot colored by ground-truth verb
- `kmeans_vs_ground_truth.png` — side-by-side comparison of K-Means clusters vs ground truth
- Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) printed to stdout

## File Structure

* **train_transformer.py**: Main training entry point.
* **test_transformer.py**: Evaluation script.
* **explore_data.ipynb**: Interactive data exploration notebook.
* **cluster_analysis.py**: Standalone PCA + K-Means cluster analysis of action trajectories.
* **utils.py**: Data loading, NLP filtering, and visualization helpers.
* **config.py**: Central configuration for paths and hyperparameters.
* **run.sh**: SLURM submission script for training + testing.
* **run_cluster.sh**: SLURM submission script for cluster analysis.
* **requirements.txt**: Dependencies.