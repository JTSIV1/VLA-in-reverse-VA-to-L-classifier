"""Contextualized subtask verb classifier for BridgeV2.

Full episode action trajectory is encoded with self-attention.
Per-segment CLS tokens are appended — each only attends to its segment's
action frames. One forward pass classifies all segments in an episode.

Usage:
    python train_bridge_ctx.py --min_class_count 30
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

from config import GRAD_CLIP_NORM, NHEAD


# ---------- Constants ----------
ACTION_DIM = 7
MAX_EP_LEN = 64   # bridge episodes avg 37 steps, cap at 64
MAX_SEGMENTS = 10  # episodes have avg 4.4 segments, max ~14


# ---------- Dataset ----------

class BridgeEpisodeDataset(Dataset):
    """Returns full episodes with all segment annotations."""

    def __init__(self, episode_groups, episode_actions, max_ep_len=MAX_EP_LEN,
                 max_segments=MAX_SEGMENTS, verb_to_id=None):
        """
        Args:
            episode_groups: list of (episode_key, [(start, end, verb), ...])
            episode_actions: dict episode_key -> (T, 7) array
            verb_to_id: shared vocab mapping
        """
        self.episodes = episode_groups
        self.actions = episode_actions
        self.max_ep_len = max_ep_len
        self.max_segments = max_segments

        if verb_to_id is not None:
            self.verb_to_id = verb_to_id
        else:
            all_verbs = set()
            for _, segs in self.episodes:
                for _, _, v in segs:
                    all_verbs.add(v)
            self.verb_to_id = {v: i for i, v in enumerate(sorted(all_verbs))}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep_key, segments = self.episodes[idx]
        actions = self.actions[ep_key]
        T = len(actions)

        # Pad/truncate actions to max_ep_len
        if T < self.max_ep_len:
            actions_padded = np.pad(actions, ((0, self.max_ep_len - T), (0, 0)),
                                    mode="constant")
        else:
            actions_padded = actions[:self.max_ep_len]
        real_len = min(T, self.max_ep_len)

        # Segment info (pad to max_segments)
        K = min(len(segments), self.max_segments)
        seg_starts = np.zeros(self.max_segments, dtype=np.int64)
        seg_ends = np.zeros(self.max_segments, dtype=np.int64)
        seg_labels = np.zeros(self.max_segments, dtype=np.int64)
        seg_mask = np.zeros(self.max_segments, dtype=np.float32)

        for i in range(K):
            start, end, verb = segments[i]
            # Clamp to valid range
            start = max(0, min(start, real_len - 1))
            end = max(start, min(end, real_len - 1))
            seg_starts[i] = start
            seg_ends[i] = end
            seg_labels[i] = self.verb_to_id.get(verb, 0)
            seg_mask[i] = 1.0

        return (
            torch.tensor(actions_padded, dtype=torch.float32),
            torch.tensor(real_len, dtype=torch.long),
            torch.tensor(seg_starts, dtype=torch.long),
            torch.tensor(seg_ends, dtype=torch.long),
            torch.tensor(seg_labels, dtype=torch.long),
            torch.tensor(seg_mask, dtype=torch.float32),
        )


# ---------- Model ----------

class ContextualVerbClassifier(nn.Module):
    """Transformer with per-segment CLS readout tokens.

    Action tokens have full self-attention (see entire episode).
    Each CLS token only attends to its segment's action frames.
    """

    def __init__(self, num_verbs, d_model=128, nhead=8, num_layers=4,
                 action_dim=ACTION_DIM, max_ep_len=MAX_EP_LEN,
                 max_segments=MAX_SEGMENTS, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_ep_len = max_ep_len
        self.max_segments = max_segments

        # Action embedding
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos = nn.Parameter(torch.zeros(1, max_ep_len, d_model))

        # Learnable CLS tokens (shared across segments)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.trunc_normal_(self.action_pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True,
                activation='gelu', dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs),
        )

    def forward(self, actions, real_lens, seg_starts, seg_ends, seg_mask):
        """
        Args:
            actions: (B, T, action_dim) full episode actions
            real_lens: (B,) real action lengths
            seg_starts: (B, K) segment start indices
            seg_ends: (B, K) segment end indices (inclusive)
            seg_mask: (B, K) 1.0 for valid segments, 0.0 for padding

        Returns:
            logits: (B, K, num_verbs)
        """
        B, T, _ = actions.shape
        K = seg_starts.shape[1]
        device = actions.device

        # Embed action tokens
        action_emb = self.action_proj(actions) + self.action_pos[:, :T, :]

        # Expand CLS tokens for each segment
        cls_tokens = self.cls_token.expand(B, K, -1)  # (B, K, d_model)

        # Full sequence: [action_0, ..., action_{T-1}, CLS_0, ..., CLS_{K-1}]
        full_seq = torch.cat([action_emb, cls_tokens], dim=1)  # (B, T+K, d)
        S = T + K

        # Build 3D attention mask (B*nhead, S, S) — vectorized
        # Use large negative (not -inf) to avoid NaN from softmax on all-masked rows
        NEG = -1e9
        attn_mask = torch.full((B, S, S), NEG, device=device)

        # Action tokens attend to all valid action tokens
        time_idx = torch.arange(T, device=device)
        valid_action = time_idx.unsqueeze(0) < real_lens.unsqueeze(1)  # (B, T)
        aa_mask = valid_action.unsqueeze(2) & valid_action.unsqueeze(1)  # (B, T, T)
        attn_mask[:, :T, :T].masked_fill_(aa_mask, 0.0)
        # All positions attend to themselves (prevents NaN from all-masked rows)
        diag = torch.eye(S, device=device).bool().unsqueeze(0).expand(B, -1, -1)
        attn_mask.masked_fill_(diag, 0.0)

        # CLS_i attends only to action tokens in segment_i + itself
        seg_action_mask = (
            (time_idx.view(1, 1, T) >= seg_starts.unsqueeze(2)) &
            (time_idx.view(1, 1, T) <= seg_ends.unsqueeze(2)) &
            (seg_mask.unsqueeze(2) > 0.5)
        )  # (B, K, T)
        attn_mask[:, T:, :T].masked_fill_(seg_action_mask, 0.0)

        # Expand for nhead: (B, S, S) -> (B*nhead, S, S)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        attn_mask = attn_mask.reshape(B * self.nhead, S, S)

        # Forward through transformer
        # Note: src_key_padding_mask omitted — the 3D attn_mask already encodes padding
        # (padded positions only self-attend via diagonal, so their outputs are harmless)
        x = full_seq
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        # Extract CLS outputs
        cls_out = x[:, T:, :]  # (B, K, d_model)

        # Classify
        logits = self.classifier(cls_out)  # (B, K, num_verbs)
        return logits


# ---------- Data Loading ----------

def load_episode_data(shard_dir, csv_path, min_class_count=30):
    """Load episodes and segment annotations, return grouped data."""
    df = pd.read_csv(csv_path)

    # Filter verbs
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= min_class_count].index)
    df = df[df["verb"].isin(keep)].reset_index(drop=True)
    print(f"After filtering: {len(df)} segments, {df['verb'].nunique()} verbs")

    # Group segments by episode
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["episode_key"]].append(
            (int(row["start_frame"]), int(row["end_frame"]), row["verb"])
        )

    # Sort segments within each episode by start_frame
    for key in grouped:
        grouped[key].sort(key=lambda x: x[0])

    # Load episode actions from shards
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} shards...")
    episode_actions = {}
    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            key = str(data[f"episode_key_{i}"])
            if key in grouped:
                episode_actions[key] = data[f"actions_{i}"].astype(np.float32)

    print(f"Loaded {len(episode_actions)} episodes with segments")

    # Build episode list
    episode_list = [(key, segs) for key, segs in grouped.items()
                    if key in episode_actions]

    return episode_list, episode_actions, df


# ---------- Training ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    episode_list, episode_actions, df = load_episode_data(
        args.shard_dir, args.csv_path, args.min_class_count)

    # Train/val split by episode
    np.random.seed(42)
    n_eps = len(episode_list)
    perm = np.random.permutation(n_eps)
    n_val = max(1, int(n_eps * args.val_fraction))
    train_episodes = [episode_list[i] for i in perm[n_val:]]
    val_episodes = [episode_list[i] for i in perm[:n_val]]
    n_train_segs = sum(len(s) for _, s in train_episodes)
    n_val_segs = sum(len(s) for _, s in val_episodes)
    print(f"Train: {len(train_episodes)} eps ({n_train_segs} segments), "
          f"Val: {len(val_episodes)} eps ({n_val_segs} segments)")

    # Datasets
    train_dataset = BridgeEpisodeDataset(
        train_episodes, episode_actions,
        max_ep_len=args.max_ep_len, max_segments=args.max_segments)
    val_dataset = BridgeEpisodeDataset(
        val_episodes, episode_actions,
        max_ep_len=args.max_ep_len, max_segments=args.max_segments,
        verb_to_id=train_dataset.verb_to_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Model
    num_verbs = len(train_dataset.verb_to_id)
    model = ContextualVerbClassifier(
        num_verbs=num_verbs,
        d_model=args.d_model,
        nhead=NHEAD,
        num_layers=args.num_layers,
        max_ep_len=args.max_ep_len,
        max_segments=args.max_segments,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params, {num_verbs} verbs")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_pct = min(3 / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos")

    # Training
    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}, max_ep_len={args.max_ep_len}")
    id_to_verb = train_dataset.id_to_verb
    training_log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for actions, real_lens, seg_starts, seg_ends, seg_labels, seg_mask in pbar:
            actions = actions.to(device)
            real_lens = real_lens.to(device)
            seg_starts = seg_starts.to(device)
            seg_ends = seg_ends.to(device)
            seg_labels = seg_labels.to(device)
            seg_mask = seg_mask.to(device)

            optimizer.zero_grad()
            logits = model(actions, real_lens, seg_starts, seg_ends, seg_mask)

            # Flatten valid segments for loss
            valid = seg_mask.bool()  # (B, K)
            flat_logits = logits[valid]  # (N_valid, num_verbs)
            flat_labels = seg_labels[valid]  # (N_valid,)

            if flat_logits.shape[0] == 0:
                continue

            loss = criterion(flat_logits, flat_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * flat_logits.shape[0]
            preds = flat_logits.argmax(dim=1)
            correct += (preds == flat_labels).sum().item()
            total += flat_logits.shape[0]

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100*correct/max(total,1):.1f}%")

        avg_loss = total_loss / max(total, 1)
        train_acc = 100 * correct / max(total, 1)
        lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={train_acc:.1f}% LR={lr:.2e}")

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)

        with torch.no_grad():
            for actions, real_lens, seg_starts, seg_ends, seg_labels, seg_mask in val_loader:
                actions = actions.to(device)
                real_lens = real_lens.to(device)
                seg_starts = seg_starts.to(device)
                seg_ends = seg_ends.to(device)
                seg_labels = seg_labels.to(device)
                seg_mask = seg_mask.to(device)

                logits = model(actions, real_lens, seg_starts, seg_ends, seg_mask)

                valid = seg_mask.bool()
                flat_logits = logits[valid]
                flat_labels = seg_labels[valid]

                if flat_logits.shape[0] == 0:
                    continue

                loss = criterion(flat_logits, flat_labels)
                val_loss += loss.item() * flat_logits.shape[0]
                preds = flat_logits.argmax(dim=1)
                val_correct += (preds == flat_labels).sum().item()
                val_total += flat_logits.shape[0]

                for lbl, pred in zip(flat_labels.cpu().tolist(), preds.cpu().tolist()):
                    val_class_total[lbl] += 1
                    val_class_correct[lbl] += int(pred == lbl)

        val_avg = val_loss / max(val_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)

        macro_recall = np.mean([
            val_class_correct.get(c, 0) / val_class_total[c]
            for c in range(num_verbs) if val_class_total.get(c, 0) > 0
        ]) * 100

        print(f"    Val: Loss={val_avg:.4f} Acc={val_acc:.1f}% MacroRecall={macro_recall:.1f}%")

        # Best checkpoint
        if val_acc > best_val_acc and args.save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "num_verbs": num_verbs,
                "verb_to_id": train_dataset.verb_to_id,
                "id_to_verb": id_to_verb,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "max_ep_len": args.max_ep_len,
                "max_segments": args.max_segments,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "dataset": "bridge_v2_subtask_ctx",
            }, best_path)
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
        })

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": vars(args), "epochs": training_log}, f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.1f}% @ epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/bridge_verb_segments.csv")
    parser.add_argument("--shard_dir", default="/data/user_data/wenjiel2/datasets/bridge_actions")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_ep_len", type=int, default=MAX_EP_LEN)
    parser.add_argument("--max_segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    main(args)
