"""Train a Transformer verb classifier on DROID manipulation trajectories.

Reuses ActionToVerbTransformer from train_transformer.py but with a DROID-
specific dataset loader that reads pre-extracted action shards.

Action-only modality for now (probing verb decodability from actions).

Usage:
    python train_droid.py --min_class_count 30 --weighted_loss --max_seq_len 512
"""

import os
import json
import glob
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    D_MODEL, NHEAD, NUM_LAYERS, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM,
)

# Import model from existing code
from train_transformer import ActionToVerbTransformer


# ---------- DROID constants ----------
DROID_ACTION_DIM = 7
DROID_ACTIONS_DIR = "/data/user_data/wenjiel2/datasets/droid_actions"
DROID_CSV = "data/droid_episodes_filtered.csv"
DROID_MAX_SEQ_LEN = 512  # avg ~385 steps, cap at 512


class DroidVerbDataset(Dataset):
    """Dataset for DROID verb classification from pre-extracted action shards."""

    def __init__(self, df, actions_cache, max_seq_len=DROID_MAX_SEQ_LEN,
                 verb_to_id=None):
        self.df = df.reset_index(drop=True)
        self.actions_cache = actions_cache
        self.max_seq_len = max_seq_len

        if verb_to_id is not None:
            self.verb_to_id = verb_to_id
        else:
            unique_verbs = sorted(self.df["verb"].unique())
            self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        print(f"Vocab mapped: {len(self.verb_to_id)} unique verbs.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ep_idx = row["episode_idx"]
        verb = row["verb"]

        actions = self.actions_cache[f"actions_{ep_idx}"]
        L = actions.shape[0]

        # Pad or truncate
        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                    mode="constant")
        else:
            actions_padded = actions[:self.max_seq_len]

        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        action_real_len = min(L, self.max_seq_len)

        label_id = self.verb_to_id.get(verb, 0)
        label = torch.tensor(label_id, dtype=torch.long)

        # Dummy frames and scene_vec for compatibility with model forward
        frames = torch.zeros(2, 3, 224, 224)
        scene_vec = torch.zeros(48)

        # seq_len = CLS + action tokens
        seq_len = 1 + action_real_len

        return frames, actions_tensor, scene_vec, label, seq_len


def load_actions_cache(actions_dir):
    """Load all shard .npz files into a single dict."""
    shard_files = sorted(glob.glob(os.path.join(actions_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} action shards from {actions_dir}...")

    cache = {}
    total_eps = 0

    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            cache[f"actions_{total_eps}"] = data[f"actions_{i}"]
            total_eps += 1

    print(f"Loaded {total_eps} episode action trajectories")
    return cache, total_eps


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Load CSV ---
    if not os.path.exists(args.csv_path):
        print(f"CSV not found at {args.csv_path}. Run scripts/consolidate_droid_actions.py first.")
        return

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} episodes from {args.csv_path}")
    print(f"Unique verbs: {df['verb'].nunique()}")

    # --- Filter sparse classes ---
    if args.min_class_count > 0:
        verb_counts = df["verb"].value_counts()
        keep_verbs = set(verb_counts[verb_counts >= args.min_class_count].index)
        n_before = len(df)
        df = df[df["verb"].isin(keep_verbs)].reset_index(drop=True)
        dropped = verb_counts.index.difference(keep_verbs)
        print(f"Filtered classes with <{args.min_class_count} samples: "
              f"{len(verb_counts)}->{len(keep_verbs)} classes, "
              f"{n_before}->{len(df)} episodes")
        if len(dropped) > 0:
            print(f"  Dropped: {sorted(dropped.tolist())}")

    # --- Train/val split ---
    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=42,
        stratify=df["verb"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # --- Load action trajectories ---
    actions_cache, _ = load_actions_cache(args.actions_dir)

    # --- Build datasets ---
    train_dataset = DroidVerbDataset(train_df, actions_cache,
                                     max_seq_len=args.max_seq_len)
    val_dataset = DroidVerbDataset(val_df, actions_cache,
                                   max_seq_len=args.max_seq_len,
                                   verb_to_id=train_dataset.verb_to_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    num_verbs = len(train_dataset.verb_to_id)
    model = ActionToVerbTransformer(
        num_verbs=num_verbs,
        d_model=args.d_model,
        num_layers=args.num_layers,
        action_dim=DROID_ACTION_DIM,
        max_action_len=args.max_seq_len,
        img_size=224,
        modality="action_only",
        action_rep="native",
        cross_layers=0,
        image_encoder="scratch",
        action_vocab_size=None,
        freeze_vision=True,
        num_frames=2,
        delta_patches=0,
        modal_dropout=0.0,
        aux_loss_weight=0.0,
        scene_dim=0,
    ).to(device)

    # --- Loss ---
    if args.weighted_loss:
        class_counts = train_df["verb"].value_counts()
        weights = torch.zeros(num_verbs)
        for verb, cid in train_dataset.verb_to_id.items():
            count = class_counts.get(verb, 1)
            weights[cid] = 1.0 / count
        weights = weights / weights.sum() * num_verbs
        criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                         label_smoothing=args.label_smoothing)
        print(f"Using weighted CE loss (min={weights.min():.3f}, max={weights.max():.3f}), "
              f"label_smoothing={args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_pct = min(args.warmup_epochs / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos")

    # --- Training loop ---
    print(f"\nStarting training: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}, max_seq_len={args.max_seq_len}")
    id_to_verb = train_dataset.id_to_verb
    training_log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (frames, actions, scene_vecs, labels, seq_lengths) in pbar:
            actions, labels = actions.to(device), labels.to(device)
            frames = frames.to(device)
            scene_vecs = scene_vecs.to(device)
            seq_lengths = seq_lengths.to(device)

            optimizer.zero_grad()
            logits = model(frames, actions, seq_lengths=seq_lengths,
                           scene_vec=scene_vecs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                class_total[lbl] += 1
                class_correct[lbl] += int(pred == lbl)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100*correct/total:.2f}%")

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e} ---")

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)

        with torch.no_grad():
            for frames, actions, scene_vecs, labels, seq_lengths in tqdm(
                    val_loader, desc="  Validating"):
                actions, labels = actions.to(device), labels.to(device)
                frames = frames.to(device)
                scene_vecs = scene_vecs.to(device)
                seq_lengths = seq_lengths.to(device)

                logits = model(frames, actions, seq_lengths=seq_lengths,
                               scene_vec=scene_vecs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    val_class_total[lbl] += 1
                    val_class_correct[lbl] += int(pred == lbl)

        val_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        # Macro F1
        per_class_f1 = []
        for cid in range(num_verbs):
            tp = val_class_correct.get(cid, 0)
            n = val_class_total.get(cid, 0)
            recall = tp / n if n > 0 else 0
            # precision requires knowing predicted counts
            per_class_f1.append(recall)  # approximate with recall for logging
        macro_recall = np.mean(per_class_f1) * 100

        print(f"    Val Loss: {val_avg:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Macro Recall: {macro_recall:.2f}%")

        # Per-class metrics
        per_class_train = {}
        per_class_val = {}
        for cid in range(num_verbs):
            verb = id_to_verb.get(cid, str(cid))
            t = class_total.get(cid, 0)
            per_class_train[verb] = {
                "acc": 100 * class_correct.get(cid, 0) / t if t > 0 else 0,
                "count": t,
            }
            vt = val_class_total.get(cid, 0)
            per_class_val[verb] = {
                "acc": 100 * val_class_correct.get(cid, 0) / vt if vt > 0 else 0,
                "count": vt,
            }

        # Save best checkpoint
        if val_acc > best_val_acc and args.save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            best_ckpt = {
                "state_dict": model.state_dict(),
                "num_verbs": num_verbs,
                "verb_to_id": train_dataset.verb_to_id,
                "id_to_verb": train_dataset.id_to_verb,
                "d_model": args.d_model,
                "action_dim": DROID_ACTION_DIM,
                "nhead": NHEAD,
                "num_layers": args.num_layers,
                "max_action_len": args.max_seq_len,
                "modality": "action_only",
                "action_rep": "native",
                "min_class_count": args.min_class_count,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "dataset": "droid",
            }
            save_dir = os.path.dirname(best_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(best_ckpt, best_path)
            print(f"    * New best val acc: {val_acc:.2f}% @ epoch {epoch+1} -> {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"    Early stopping: no improvement for {args.patience} epochs")
                break

        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_avg,
            "val_acc": val_acc,
            "per_class_train": per_class_train,
            "per_class_val": per_class_val,
        }
        training_log.append(epoch_metrics)

        if args.log_path:
            log_dir = os.path.dirname(args.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": vars(args), "epochs": training_log}, f, indent=2)

    # Final checkpoint
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "state_dict": model.state_dict(),
            "num_verbs": num_verbs,
            "verb_to_id": train_dataset.verb_to_id,
            "id_to_verb": train_dataset.id_to_verb,
            "d_model": args.d_model,
            "action_dim": DROID_ACTION_DIM,
            "nhead": NHEAD,
            "num_layers": args.num_layers,
            "max_action_len": args.max_seq_len,
            "modality": "action_only",
            "action_rep": "native",
            "min_class_count": args.min_class_count,
            "dataset": "droid",
        }
        torch.save(checkpoint, args.save_path)
        print(f"\nFinal checkpoint saved to {args.save_path}")

    print(f"\nBest val acc: {best_val_acc:.2f}% @ epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=DROID_CSV)
    parser.add_argument("--actions_dir", type=str, default=DROID_ACTIONS_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=DROID_MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of data to use for validation")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()
    main(args)
