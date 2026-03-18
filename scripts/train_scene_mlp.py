"""
Train a PyTorch MLP on scene_obs engineered features for verb classification.

Architecture: 96-d input → 256 → ReLU → 128 → ReLU → n_classes
Same as the sklearn MLP but with proper train/val split and training loop.

Usage:
  python scripts/train_scene_mlp.py --tag gt_l0_scene_22cls
  python scripts/train_scene_mlp.py --tag gt_l0_scene_22cls --verb_file data/verb_classes.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE, CHECKPOINT_DIR
from utils import load_calvin_to_dataframe


class SceneFeatureDataset(Dataset):
    """Pre-computed scene_obs engineered features."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SceneMLP(nn.Module):
    def __init__(self, input_dim=96, hidden1=256, hidden2=128, n_classes=22, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_features(df, data_dir, verb2id):
    X, y = [], []
    for _, row in df.iterrows():
        v = row['primary_verb']
        if v not in verb2id:
            continue
        s0 = np.load(os.path.join(data_dir, EPISODE_TEMPLATE.format(row['start_idx'])))['scene_obs']
        s1 = np.load(os.path.join(data_dir, EPISODE_TEMPLATE.format(row['end_idx'])))['scene_obs']
        d = s1 - s0
        X.append(np.concatenate([d, np.abs(d), np.sign(d), (np.abs(d) > 0.01).astype(float)]))
        y.append(verb2id[v])
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="gt_l0_scene")
    parser.add_argument("--verb_file", type=str, default="data/verb_classes.txt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load verb classes
    verb_classes = open(args.verb_file).read().strip().split('\n')
    verb2id = {v: i for i, v in enumerate(verb_classes)}
    id2verb = {i: v for i, v in enumerate(verb_classes)}
    n_classes = len(verb_classes)

    # Load data
    print("Loading data...")
    df_train = load_calvin_to_dataframe(TRAIN_DIR)
    df_val = load_calvin_to_dataframe(VAL_DIR)
    df_train = df_train[df_train['primary_verb'].isin(verb_classes)]
    df_val = df_val[df_val['primary_verb'].isin(verb_classes)]

    X_train, y_train = build_features(df_train, TRAIN_DIR, verb2id)
    X_val, y_val = build_features(df_val, VAL_DIR, verb2id)

    # Standardize (fit on train only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    print("Train: {}, Val: {}, Classes: {}".format(len(X_train), len(X_val), n_classes))

    train_ds = SceneFeatureDataset(X_train, y_train)
    val_ds = SceneFeatureDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Weighted CE
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    weights = torch.zeros(n_classes)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 1.0
    weights = weights.to(device)

    model = SceneMLP(input_dim=X_train.shape[1], n_classes=n_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "{}_best.pth".format(args.tag))

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)
        scheduler.step()

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print("Epoch {:3d}/{}: train_loss={:.4f} train_acc={:.1f}% | val_loss={:.4f} val_acc={:.1f}%".format(
            epoch, args.epochs, train_loss / train_total, train_acc * 100,
            val_loss / val_total, val_acc * 100))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'n_classes': n_classes,
                'tag': args.tag,
                'mean': mean.tolist(),
                'std': std.tolist(),
            }, ckpt_path)
            print("  -> Best val acc: {:.1f}% (saved)".format(val_acc * 100))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("  Early stopping at epoch {}".format(epoch))
                break

    # Final eval with best checkpoint
    print("\nLoading best checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=verb_classes,
                                   output_dict=True, zero_division=0)
    print("\n=== Best Checkpoint (epoch {}) ===".format(ckpt['epoch']))
    print("Val Acc: {:.1f}%  Macro F1: {:.1f}%".format(
        report['accuracy'] * 100, report['macro avg']['f1-score'] * 100))
    print(classification_report(all_labels, all_preds, target_names=verb_classes, zero_division=0))

    result = {
        'model': args.tag,
        'accuracy': report['accuracy'] * 100,
        'macro_f1': report['macro avg']['f1-score'] * 100,
        'best_epoch': ckpt['epoch'],
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_classes': n_classes,
        'per_class': report,
    }
    with open('results/{}_best_metrics.json'.format(args.tag), 'w') as f:
        json.dump(result, f, indent=2)

    with open('results/{}_preds.json'.format(args.tag), 'w') as f:
        json.dump({
            'labels': all_labels,
            'preds': all_preds,
            'id_to_verb': id2verb,
        }, f)

    print("Saved to results/{}_best_metrics.json".format(args.tag))


if __name__ == "__main__":
    main()
