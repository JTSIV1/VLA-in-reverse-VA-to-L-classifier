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

## File Structure

* **train_calvin_verb.py**: Main training entry point.
* **test_calvin_verb.py**: Evaluation script.
* **utils.py**: Data loading, NLP filtering, and visualization helpers.
* **requirements.txt**: Dependencies.