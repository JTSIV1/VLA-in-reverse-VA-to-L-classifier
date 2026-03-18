"""
Train sklearn MLP on L1 segments using scene_obs features.

For each L1 segment, extracts scene_obs delta features:
  delta = scene_obs[end] - scene_obs[start]
  features = [delta, |delta|, sign(delta), 1(|delta| > 0.01)]  → 96-d

Mirrors the R7 scene_obs sklearn approach but on L1 segments instead of full episodes.

Usage:
  python scripts/train_l1_scene_sklearn.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE

SCENE_OBS_KEY = "scene_obs"


def load_segments(jsonl_path):
    """Load L1 segment metadata."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def extract_scene_features(data_dir, segments):
    """Extract scene_obs engineered features for each segment."""
    features = []
    labels = []
    for seg in segments:
        start_global = seg["global_start"]
        end_global = seg["global_end"]

        start_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(start_global))
        end_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(end_global))

        s_start = np.load(start_path, mmap_mode='r')[SCENE_OBS_KEY]
        s_end = np.load(end_path, mmap_mode='r')[SCENE_OBS_KEY]

        delta = s_end - s_start
        feat = np.concatenate([
            delta,                              # 24-d
            np.abs(delta),                      # 24-d
            np.sign(delta),                     # 24-d
            (np.abs(delta) > 0.01).astype(float),  # 24-d
        ])  # 96-d
        features.append(feat)
        labels.append(seg["label"])

    return np.array(features), np.array(labels)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_dir", type=str, default="data/l1_segments")
    parser.add_argument("--tag", type=str, default="l1_scene_sklearn")
    args = parser.parse_args()

    seg_dir = args.seg_dir
    tag = args.tag
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load label map
    with open(os.path.join(seg_dir, "label_map.json")) as f:
        label_info = json.load(f)
    id2verb = {int(k): v for k, v in label_info["id2verb"].items()}
    verb2id = label_info["verb2id"]

    # Load segments
    print("Loading segments...")
    train_segs = load_segments(os.path.join(seg_dir, "train.jsonl"))
    val_segs = load_segments(os.path.join(seg_dir, "val.jsonl"))
    print("  Train: {}, Val: {}".format(len(train_segs), len(val_segs)))

    # Extract features
    print("Extracting scene_obs features (training)...")
    X_train, y_train = extract_scene_features(TRAIN_DIR, train_segs)
    print("  X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))

    print("Extracting scene_obs features (validation)...")
    X_val, y_val = extract_scene_features(VAL_DIR, val_segs)
    print("  X_val: {}, y_val: {}".format(X_val.shape, y_val.shape))

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train MLP (same architecture as R7 scene sklearn)
    print("\nTraining MLP...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    class_names = [id2verb[i] for i in range(len(id2verb))]
    report = classification_report(y_val, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    print("\n=== L1 Scene-Obs sklearn MLP ===")
    print("Val Accuracy: {:.1f}%".format(acc * 100))
    print("Val Macro F1: {:.1f}%".format(report["macro avg"]["f1-score"] * 100))
    print()
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    # Save results
    result = {
        "model": "l1_scene_sklearn_mlp",
        "accuracy": acc * 100,
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "n_train": len(train_segs),
        "n_val": len(val_segs),
        "n_classes": len(class_names),
        "per_class": report,
    }
    out_path = os.path.join(results_dir, "{}_metrics.json".format(tag))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to", out_path)

    # Save predictions for complementarity analysis
    preds_path = os.path.join(results_dir, "{}_preds.json".format(tag))
    with open(preds_path, "w") as f:
        json.dump({
            "labels": y_val.tolist(),
            "preds": y_pred.tolist(),
            "id_to_verb": id2verb,
        }, f)
    print("Saved predictions to", preds_path)


if __name__ == "__main__":
    main()
