"""Train a Transformer verb classifier on BridgeV2 subtask segments.

Uses Emma-X GCOT subtask annotations matched to BridgeV2 action trajectories.
Each sample is a short action segment (~7 steps avg) labeled with a verb.

Usage:
    python train_bridge.py --min_class_count 30 --weighted_loss
"""

import os
import json
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
from train_transformer import ActionToVerbTransformer


# ---------- BridgeV2 constants ----------
BRIDGE_ACTION_DIM = 7
BRIDGE_CSV = "data/bridge_verb_segments.csv"
BRIDGE_ACTIONS_NPZ = "/data/user_data/wenjiel2/datasets/bridge_actions/segment_actions.npz"
BRIDGE_MAX_SEQ_LEN = 64  # segments are short (mean=7, max=117)


class BridgeVerbDataset(Dataset):
    """Dataset for BridgeV2 subtask verb classification."""

    def __init__(self, df, actions_cache, max_seq_len=BRIDGE_MAX_SEQ_LEN,
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
        print(f"Vocab: {len(self.verb_to_id)} verbs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seg_idx = row["seg_idx"]
        verb = row["verb"]

        actions = self.actions_cache[f"actions_{seg_idx}"]
        L = actions.shape[0]

        if L < self.max_seq_len:
            actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                    mode="constant")
        else:
            actions_padded = actions[:self.max_seq_len]

        actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
        action_real_len = min(L, self.max_seq_len)
        label = torch.tensor(self.verb_to_id.get(verb, 0), dtype=torch.long)

        frames = torch.zeros(2, 3, 224, 224)
        scene_vec = torch.zeros(48)
        seq_len = 1 + action_real_len

        return frames, actions_tensor, scene_vec, label, seq_len


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} segments, {df['verb'].nunique()} verbs")

    # Filter sparse classes
    if args.min_class_count > 0:
        vc = df["verb"].value_counts()
        keep = set(vc[vc >= args.min_class_count].index)
        n_before = len(df)
        df = df[df["verb"].isin(keep)].reset_index(drop=True)
        print(f"Filtered: {n_before}->{len(df)} segments, "
              f"{len(vc)}->{len(keep)} verbs")

    # Train/val split
    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=42, stratify=df["verb"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load actions eagerly into a plain dict (NpzFile is not safe for multi-worker)
    print(f"Loading segment actions from {args.actions_npz}...")
    npz = np.load(args.actions_npz, allow_pickle=True)
    actions_cache = {k: npz[k] for k in npz.files if k.startswith("actions_")}
    n_segs = int(npz["n_segments"])
    npz.close()
    print(f"Loaded {n_segs} segments into memory")

    # Datasets
    train_dataset = BridgeVerbDataset(train_df, actions_cache,
                                       max_seq_len=args.max_seq_len)
    val_dataset = BridgeVerbDataset(val_df, actions_cache,
                                     max_seq_len=args.max_seq_len,
                                     verb_to_id=train_dataset.verb_to_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Model
    num_verbs = len(train_dataset.verb_to_id)
    model = ActionToVerbTransformer(
        num_verbs=num_verbs,
        d_model=args.d_model,
        num_layers=args.num_layers,
        action_dim=BRIDGE_ACTION_DIM,
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

    # Loss
    if args.weighted_loss:
        class_counts = train_df["verb"].value_counts()
        weights = torch.zeros(num_verbs)
        for verb, cid in train_dataset.verb_to_id.items():
            count = class_counts.get(verb, 1)
            weights[cid] = 1.0 / count
        weights = weights / weights.sum() * num_verbs
        criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                         label_smoothing=args.label_smoothing)
        print(f"Weighted CE (min={weights.min():.3f}, max={weights.max():.3f}), "
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

    # Training
    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}, max_seq_len={args.max_seq_len}")
    id_to_verb = train_dataset.id_to_verb
    training_log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (frames, actions, scene_vecs, labels, seq_lengths) in pbar:
            actions, labels = actions.to(device), labels.to(device)
            frames, scene_vecs = frames.to(device), scene_vecs.to(device)
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

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={train_acc:.1f}% LR={lr:.2e}")

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)

        with torch.no_grad():
            for frames, actions, scene_vecs, labels, seq_lengths in val_loader:
                actions, labels = actions.to(device), labels.to(device)
                frames, scene_vecs = frames.to(device), scene_vecs.to(device)
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
        val_acc = 100 * val_correct / val_total if val_total else 0

        macro_recall = np.mean([
            val_class_correct.get(c, 0) / val_class_total[c]
            for c in range(num_verbs) if val_class_total.get(c, 0) > 0
        ]) * 100

        print(f"    Val: Loss={val_avg:.4f} Acc={val_acc:.1f}% MacroRecall={macro_recall:.1f}%")

        # Per-class
        per_class_val = {}
        for cid in range(num_verbs):
            verb = id_to_verb.get(cid, str(cid))
            vt = val_class_total.get(cid, 0)
            per_class_val[verb] = {
                "acc": 100 * val_class_correct.get(cid, 0) / vt if vt else 0,
                "count": vt,
            }

        # Best checkpoint
        if val_acc > best_val_acc and args.save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            ckpt = {
                "state_dict": model.state_dict(),
                "num_verbs": num_verbs,
                "verb_to_id": train_dataset.verb_to_id,
                "id_to_verb": train_dataset.id_to_verb,
                "d_model": args.d_model,
                "action_dim": BRIDGE_ACTION_DIM,
                "nhead": NHEAD,
                "num_layers": args.num_layers,
                "max_action_len": args.max_seq_len,
                "modality": "action_only",
                "dataset": "bridge_v2_subtask",
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
            os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
            torch.save(ckpt, best_path)
            print(f"    * Best val acc: {val_acc:.1f}% @ epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"    Early stopping after {args.patience} epochs no improvement")
                break

        training_log.append({
            "epoch": epoch + 1, "lr": lr,
            "train_loss": avg_loss, "train_acc": train_acc,
            "val_loss": val_avg, "val_acc": val_acc,
            "macro_recall": macro_recall,
            "per_class_val": per_class_val,
        })

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": vars(args), "epochs": training_log}, f, indent=2)

    # Final checkpoint
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "num_verbs": num_verbs,
            "verb_to_id": train_dataset.verb_to_id,
            "id_to_verb": train_dataset.id_to_verb,
            "d_model": args.d_model,
            "dataset": "bridge_v2_subtask",
        }, args.save_path)

    print(f"\nBest val acc: {best_val_acc:.1f}% @ epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=BRIDGE_CSV)
    parser.add_argument("--actions_npz", default=BRIDGE_ACTIONS_NPZ)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=BRIDGE_MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    main(args)
