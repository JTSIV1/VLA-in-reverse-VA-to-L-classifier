"""Train verb classifier on BridgeV2 using OAT token representations.

Encodes episodes with a pretrained OAT tokenizer, then trains a transformer
classifier on the resulting discrete tokens or continuous latent vectors.

Usage:
    python train_bridge_oat.py --oat_ckpt checkpoints/oat_bridge_j6660959_best.pth
"""

import os
import sys
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenization"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tokenization", "oat"))

from train_oat_bridge import build_oat
from config import GRAD_CLIP_NORM, NHEAD


# ---------- Constants ----------
OAT_HORIZON = 32
OAT_NUM_REGISTERS = 8
OAT_LATENT_DIM = 4  # FSQ output dim per token
MAX_SEQ_LEN = 32  # max OAT tokens per episode (8 tokens per 32-step chunk)


# ---------- OAT Encoding ----------

def load_oat_model(ckpt_path, device):
    """Load pretrained OAT model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_oat(
        action_dim=ckpt["action_dim"],
        horizon=ckpt["horizon"],
        num_registers=ckpt["num_registers"],
        emb_dim=ckpt["emb_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded OAT: vocab={ckpt['vocab_size']}, horizon={ckpt['horizon']}, "
          f"recon_mse={ckpt['best_recon_mse']:.6f}")
    return model, ckpt


@torch.no_grad()
def encode_episodes(oat_model, episode_actions, horizon, device, batch_size=256):
    """Encode all episodes into OAT tokens and latents.

    Returns:
        tokens_dict: {episode_key: (N_tokens,) int array}
        latents_dict: {episode_key: (N_tokens, latent_dim) float array}
    """
    tokens_dict = {}
    latents_dict = {}

    keys = list(episode_actions.keys())
    # Batch episodes by chunking
    all_chunks = []  # (chunk_tensor, episode_key, chunk_idx)
    chunk_map = defaultdict(list)  # episode_key -> list of chunk indices in all_chunks

    for key in keys:
        actions = episode_actions[key]
        T = len(actions)
        if T < 2:
            continue
        # Split into non-overlapping chunks of horizon
        n_chunks = max(1, (T + horizon - 1) // horizon)
        for c in range(n_chunks):
            start = c * horizon
            end = min(start + horizon, T)
            chunk = actions[start:end]
            if len(chunk) < horizon:
                chunk = np.pad(chunk, ((0, horizon - len(chunk)), (0, 0)), mode="edge")
            chunk_map[key].append(len(all_chunks))
            all_chunks.append(torch.tensor(chunk, dtype=torch.float32))

    # Batch encode
    all_chunks_tensor = torch.stack(all_chunks)  # (N_total_chunks, horizon, action_dim)
    all_tokens = []
    all_latents = []

    for i in range(0, len(all_chunks_tensor), batch_size):
        batch = all_chunks_tensor[i:i + batch_size].to(device)
        latents, tokens = oat_model.encode(batch)
        all_tokens.append(tokens.cpu().numpy())
        all_latents.append(latents.cpu().numpy())

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N_total, num_registers)
    all_latents = np.concatenate(all_latents, axis=0)  # (N_total, num_registers, latent_dim)

    # Reassemble per episode
    for key in keys:
        if key not in chunk_map:
            continue
        idxs = chunk_map[key]
        ep_tokens = np.concatenate([all_tokens[i] for i in idxs], axis=0)
        ep_latents = np.concatenate([all_latents[i] for i in idxs], axis=0)
        tokens_dict[key] = ep_tokens
        latents_dict[key] = ep_latents

    return tokens_dict, latents_dict


# ---------- Dataset ----------

class OATVerbDataset(Dataset):
    """Dataset for verb classification from OAT representations."""

    def __init__(self, keys, labels, tokens_dict, latents_dict,
                 max_seq_len=MAX_SEQ_LEN, use_latents=True):
        self.keys = keys
        self.labels = labels
        self.tokens_dict = tokens_dict
        self.latents_dict = latents_dict
        self.max_seq_len = max_seq_len
        self.use_latents = use_latents

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.labels[idx]

        tokens = self.tokens_dict[key]
        latents = self.latents_dict[key]
        L = len(tokens)

        # Pad/truncate
        if self.use_latents:
            latent_dim = latents.shape[1]
            if L < self.max_seq_len:
                padded = np.pad(latents, ((0, self.max_seq_len - L), (0, 0)), mode="constant")
            else:
                padded = latents[:self.max_seq_len]
            features = torch.tensor(padded, dtype=torch.float32)
        else:
            if L < self.max_seq_len:
                padded = np.pad(tokens, (0, self.max_seq_len - L), mode="constant",
                                constant_values=0)
            else:
                padded = tokens[:self.max_seq_len]
            features = torch.tensor(padded, dtype=torch.long)

        real_len = min(L, self.max_seq_len)
        return features, torch.tensor(label, dtype=torch.long), torch.tensor(real_len, dtype=torch.long)


# ---------- Model ----------

class OATVerbClassifier(nn.Module):
    """Transformer classifier operating on OAT token representations."""

    def __init__(self, num_verbs, d_model=128, nhead=8, num_layers=4,
                 input_dim=4, vocab_size=1000, use_latents=True,
                 max_seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.use_latents = use_latents
        self.max_seq_len = max_seq_len

        if use_latents:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.token_embed = nn.Embedding(vocab_size, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            activation='gelu', dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs),
        )

    def forward(self, features, real_lens):
        B = features.shape[0]
        device = features.device

        if self.use_latents:
            x = self.input_proj(features)  # (B, S, d_model)
        else:
            x = self.token_embed(features)  # (B, S, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+S, d_model)
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Padding mask: True = ignore
        S = self.max_seq_len + 1  # +1 for CLS
        positions = torch.arange(S, device=device).unsqueeze(0)
        pad_mask = positions >= (real_lens.unsqueeze(1) + 1)  # +1 for CLS

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)


# ---------- Data Loading ----------

def load_highlevel_data(shard_dir, csv_path, min_class_count=30):
    """Load high-level BridgeV2 verb data (one verb per episode)."""
    df = pd.read_csv(csv_path)

    # Filter by min count
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= min_class_count].index)
    df = df[df["verb"].isin(keep)].reset_index(drop=True)

    # Load episode actions from shards (only episodes in CSV)
    needed_keys = set(df["episode_key"])
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} shards (need {len(needed_keys)} episodes)...")
    episode_actions = {}
    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            key = str(data[f"episode_key_{i}"])
            if key in needed_keys:
                episode_actions[key] = data[f"actions_{i}"].astype(np.float32)

    # Match with CSV
    valid = df["episode_key"].isin(episode_actions)
    df = df[valid].reset_index(drop=True)
    print(f"After filtering: {len(df)} episodes, {df['verb'].nunique()} verbs")

    return df, episode_actions


# ---------- Training ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df, episode_actions = load_highlevel_data(
        args.shard_dir, args.csv_path, args.min_class_count)

    # Load and encode with OAT
    print(f"\nEncoding episodes with OAT ({args.oat_ckpt})...")
    oat_model, oat_ckpt = load_oat_model(args.oat_ckpt, device)
    tokens_dict, latents_dict = encode_episodes(
        oat_model, episode_actions, oat_ckpt["horizon"], device)
    del oat_model  # free GPU memory
    torch.cuda.empty_cache()

    # Print encoding stats
    lens = [len(tokens_dict[k]) for k in tokens_dict if k in set(df["episode_key"])]
    print(f"OAT tokens per episode: mean={np.mean(lens):.1f}, "
          f"median={np.median(lens):.0f}, max={max(lens)}")

    # Train/val split by episode
    verb_to_id = {v: i for i, v in enumerate(sorted(df["verb"].unique()))}
    id_to_verb = {i: v for v, i in verb_to_id.items()}
    num_verbs = len(verb_to_id)

    np.random.seed(42)
    all_keys = df["episode_key"].values
    all_labels = np.array([verb_to_id[v] for v in df["verb"]])
    perm = np.random.permutation(len(df))
    n_val = max(1, int(len(df) * args.val_fraction))

    train_keys = all_keys[perm[n_val:]]
    train_labels = all_labels[perm[n_val:]]
    val_keys = all_keys[perm[:n_val]]
    val_labels = all_labels[perm[:n_val]]
    print(f"Train: {len(train_keys)}, Val: {len(val_keys)}, Verbs: {num_verbs}")

    use_latents = args.rep_type == "latent"
    train_dataset = OATVerbDataset(train_keys, train_labels, tokens_dict, latents_dict,
                                    max_seq_len=args.max_seq_len, use_latents=use_latents)
    val_dataset = OATVerbDataset(val_keys, val_labels, tokens_dict, latents_dict,
                                  max_seq_len=args.max_seq_len, use_latents=use_latents)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Model
    input_dim = OAT_LATENT_DIM if use_latents else None
    model = OATVerbClassifier(
        num_verbs=num_verbs,
        d_model=args.d_model,
        nhead=NHEAD,
        num_layers=args.num_layers,
        input_dim=input_dim,
        vocab_size=oat_ckpt["vocab_size"],
        use_latents=use_latents,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params, rep={args.rep_type}")

    if args.weighted_loss:
        # Inverse-frequency weights
        train_df = df.iloc[perm[n_val:]]
        class_counts = train_df["verb"].value_counts()
        weights = torch.zeros(num_verbs)
        for verb, cid in verb_to_id.items():
            count = class_counts.get(verb, 1)
            weights[cid] = 1.0 / count
        weights = weights / weights.sum() * num_verbs
        criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                         label_smoothing=args.label_smoothing)
        print(f"Using weighted CE loss (min={weights.min():.3f}, max={weights.max():.3f})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_pct = min(3 / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos")

    # Training
    print(f"\nTraining: {num_verbs} verbs, {args.epochs} epochs, "
          f"d_model={args.d_model}, max_seq_len={args.max_seq_len}")
    training_log = []
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for features, labels, real_lens in pbar:
            features = features.to(device)
            labels = labels.to(device)
            real_lens = real_lens.to(device)

            optimizer.zero_grad()
            logits = model(features, real_lens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/max(total,1):.1f}%")

        avg_loss = total_loss / max(total, 1)
        train_acc = 100 * correct / max(total, 1)
        lr = scheduler.get_last_lr()[0]

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)

        with torch.no_grad():
            for features, labels, real_lens in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                real_lens = real_lens.to(device)

                logits = model(features, real_lens)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    val_class_total[lbl] += 1
                    val_class_correct[lbl] += int(pred == lbl)

        val_avg = val_loss / max(val_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)
        macro_recall = np.mean([
            val_class_correct.get(c, 0) / val_class_total[c]
            for c in range(num_verbs) if val_class_total.get(c, 0) > 0
        ]) * 100

        print(f"--- Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_avg:.4f} Acc={val_acc:.1f}% MacroR={macro_recall:.1f}% LR={lr:.2e}")

        # Best checkpoint
        if val_acc > best_val_acc and args.save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "num_verbs": num_verbs,
                "verb_to_id": verb_to_id,
                "id_to_verb": id_to_verb,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "max_seq_len": args.max_seq_len,
                "rep_type": args.rep_type,
                "oat_ckpt": args.oat_ckpt,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "dataset": "bridge_v2_oat",
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
            "val_loss": val_avg, "val_acc": val_acc, "macro_recall": macro_recall,
        })

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": vars(args), "epochs": training_log}, f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.1f}% @ epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oat_ckpt", required=True, help="Path to pretrained OAT checkpoint")
    parser.add_argument("--csv_path", default="data/bridge_episodes_filtered.csv")
    parser.add_argument("--shard_dir", default="/data/user_data/wenjiel2/datasets/bridge_actions")
    parser.add_argument("--rep_type", choices=["latent", "discrete"], default="latent",
                        help="Use continuous latents or discrete token indices")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
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
    parser.add_argument("--weighted_loss", action="store_true")
    args = parser.parse_args()
    main(args)
