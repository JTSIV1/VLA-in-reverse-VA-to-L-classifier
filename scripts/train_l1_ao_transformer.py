"""
Train ActionToVerbTransformer on L1 segments (action-only).

Each L1 phase is a separate training sample with action subsequence
sliced by [phase_start, phase_end]. Uses the same transformer architecture
as the main train_transformer.py but with a simpler dataset loader.

Usage:
  python scripts/train_l1_ao_transformer.py [--epochs 30] [--tag l1_ao]

Submitting via SLURM:
  sbatch scripts/submit_l1_ao.sh
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ACTION_KEY, EPISODE_TEMPLATE, TRAIN_DIR, VAL_DIR,
    D_MODEL, NHEAD, NUM_LAYERS, DROPOUT_RATE, ACTION_DIM,
    BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, CHECKPOINT_DIR,
)


# ── Dataset ─────────────────────────────────────────────────────────────────

class L1SegmentDataset(Dataset):
    """Load action subsequences for L1 segments."""

    def __init__(self, jsonl_path, data_dir, label_map_path, max_seq_len=64):
        with open(jsonl_path) as f:
            self.segments = [json.loads(line) for line in f]
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        with open(label_map_path) as f:
            info = json.load(f)
        self.verb2id = info["verb2id"]
        self.id2verb = {int(k): v for k, v in info["id2verb"].items()}
        self.n_classes = len(self.verb2id)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        start_global = seg["global_start"]
        end_global = seg["global_end"]
        label = seg["label"]

        # Load actions
        actions = []
        for t in range(start_global, end_global + 1):
            ep_path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(t))
            actions.append(np.load(ep_path, mmap_mode='r')[ACTION_KEY])
        actions = np.array(actions)  # (T, 7)

        L = len(actions)
        if L < self.max_seq_len:
            padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)), mode='constant')
        else:
            padded = actions[:self.max_seq_len]

        actions_tensor = torch.tensor(padded, dtype=torch.float32)
        real_len = min(L, self.max_seq_len)

        return actions_tensor, real_len, label


# ── Model (simplified from train_transformer.py) ────────────────────────────

class ActionToVerbTransformerL1(nn.Module):
    """Action-only transformer for L1 segment classification."""

    def __init__(self, n_classes, action_dim=ACTION_DIM, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE,
                 max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.action_proj = nn.Linear(action_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, actions, real_lens):
        B = actions.size(0)
        x = self.action_proj(actions)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]

        # Create padding mask: True = masked position
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0)
        # CLS at position 0 is never masked; action tokens masked after real_len
        mask = mask >= (real_lens.unsqueeze(1) + 1)  # +1 for CLS

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_out = x[:, 0]  # CLS token
        return self.classifier(cls_out)


# ── Training ────────────────────────────────────────────────────────────────

def compute_class_weights(segments, n_classes):
    """Compute inverse-frequency class weights."""
    counts = Counter(s["label"] for s in segments)
    total = sum(counts.values())
    weights = torch.zeros(n_classes)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 1.0
    return weights


def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for actions, real_lens, labels in tqdm(loader, desc="Train", leave=False):
        actions = actions.to(device)
        real_lens = real_lens.to(device)
        labels = labels.to(device)

        logits = model(actions, real_lens)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for actions, real_lens, labels in loader:
        actions = actions.to(device)
        real_lens = real_lens.to(device)
        labels = labels.to(device)

        logits = model(actions, real_lens)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--tag", type=str, default="l1_ao")
    parser.add_argument("--seg_dir", type=str, default="data/l1_segments")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from best checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load datasets
    label_map_path = os.path.join(args.seg_dir, "label_map.json")
    train_ds = L1SegmentDataset(
        os.path.join(args.seg_dir, "train.jsonl"), TRAIN_DIR,
        label_map_path, max_seq_len=args.max_seq_len)
    val_ds = L1SegmentDataset(
        os.path.join(args.seg_dir, "val.jsonl"), VAL_DIR,
        label_map_path, max_seq_len=args.max_seq_len)

    n_classes = train_ds.n_classes
    print("Train: {}, Val: {}, Classes: {}".format(len(train_ds), len(val_ds), n_classes))
    print("Classes:", [train_ds.id2verb[i] for i in range(n_classes)])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = ActionToVerbTransformerL1(
        n_classes=n_classes, d_model=args.d_model,
        num_layers=args.num_layers, max_seq_len=args.max_seq_len,
    ).to(device)
    print("Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Weighted CE
    with open(os.path.join(args.seg_dir, "train.jsonl")) as f:
        train_segs = [json.loads(line) for line in f]
    weights = compute_class_weights(train_segs, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    start_epoch = 1
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "{}_best.pth".format(args.tag))

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print("Resumed from epoch {}, best_val_acc={:.1f}%".format(
            start_epoch - 1, best_val_acc * 100))

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP_NORM)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        print("Epoch {:2d}/{}: train_loss={:.4f} train_acc={:.1f}% | "
              "val_loss={:.4f} val_acc={:.1f}%".format(
                  epoch, args.epochs, train_loss, train_acc * 100,
                  val_loss, val_acc * 100))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "n_classes": n_classes,
                "d_model": args.d_model,
                "tag": args.tag,
            }, ckpt_path)
            print("  -> Best val acc: {:.1f}% (saved)".format(val_acc * 100))

    # Final evaluation with best checkpoint
    print("\nLoading best checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc, val_preds, val_labels = eval_epoch(
        model, val_loader, criterion, device)

    from sklearn.metrics import classification_report
    class_names = [train_ds.id2verb[i] for i in range(n_classes)]
    report = classification_report(val_labels, val_preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print("\n=== Best Checkpoint Results ===")
    print("Val Accuracy: {:.1f}%".format(val_acc * 100))
    print("Val Macro F1: {:.1f}%".format(report["macro avg"]["f1-score"] * 100))
    print()
    print(classification_report(val_labels, val_preds, target_names=class_names,
                                zero_division=0))

    # Save results
    result = {
        "model": "l1_ao_transformer",
        "accuracy": val_acc * 100,
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "best_epoch": ckpt["epoch"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_classes": n_classes,
        "per_class": report,
    }
    out_path = "results/{}_best_metrics.json".format(args.tag)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to", out_path)

    # Save predictions
    preds_path = "results/{}_preds.json".format(args.tag)
    with open(preds_path, "w") as f:
        json.dump({
            "labels": val_labels,
            "preds": val_preds,
            "id_to_verb": {int(k): v for k, v in train_ds.id2verb.items()},
        }, f)
    print("Saved predictions to", preds_path)


if __name__ == "__main__":
    main()
