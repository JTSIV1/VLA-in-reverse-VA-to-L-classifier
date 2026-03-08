"""train_verb_probe_vq.py

Train verb classifiers directly on VQ-VLA representations to measure how much
verb information survives the quantization bottleneck.

Two probe levels per condition:
  Level 1 (latent): classify on z_q sequences  (128-d per chunk, continuous)
                    → did verb CE push latents into verb-separable positions?
  Level 2 (token):  classify on code ID sequences  (integer in [0,255] per group)
                    → are discrete token assignments separable by verb?
                    → this is what the LLM actually sees

Conditions: vq_vanilla (lambda=0), vq_verb (lambda=0.5)

Usage:
    python -m openvla_experiment.scripts.train_verb_probe_vq \\
        --condition vq_vanilla \\
        --vqvla_checkpoint_dir checkpoints/vqvla_ft_vanilla \\
        --output_dir results/verb_probe_vq/vanilla

    python -m openvla_experiment.scripts.train_verb_probe_vq \\
        --condition vq_verb \\
        --vqvla_checkpoint_dir checkpoints/vqvla_ft_verb_l0.5 \\
        --output_dir results/verb_probe_vq/vq_verb
"""

import csv
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import load_calvin_to_dataframe
from config import DATA_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE
from tokenization.vqvae_tokenizer import (
    VQVLA_WINDOW_SIZE, VQVLA_NUM_TOKENS, VQVLA_VOCAB_SIZE,
    VQVLA_CONFIG_DIR, VQVLA_CHECKPOINT_PATH,
)


# ── VQ-VLA loader ─────────────────────────────────────────────────────────────

def load_vqvae(checkpoint_dir, config_dir=VQVLA_CONFIG_DIR):
    """Load fine-tuned VQ-VLA inner model (frozen, eval mode)."""
    from tokenization.vqvla import ActionVQVAELossWrapper
    wrapper = ActionVQVAELossWrapper(
        model_path=config_dir,
        checkpoint_path=VQVLA_CHECKPOINT_PATH,
        is_eval=True,
        freeze=True,
        use_action_type_pe=True,
        use_time_pe=True,
    )
    weights_path = os.path.join(checkpoint_dir, "vqvla_weights.pth")
    inner_weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    wrapper.vqvae.load_state_dict(inner_weights, strict=True)
    wrapper.vqvae.eval()
    for p in wrapper.vqvae.parameters():
        p.requires_grad = False
    return wrapper.vqvae


# ── Dataset ───────────────────────────────────────────────────────────────────

class VQVerbDataset(Dataset):
    """
    Pre-tokenizes CALVIN trajectories through the VQ-VLA encoder and stores
    both z_q (quantized latents) and code IDs for each trajectory.

    Each item:
        zq_seq:    (n_chunks, 128)    float32  — quantized latent per 5-step chunk
        code_seq:  (n_chunks, 4)      int64    — VQ code indices [0, 255] per group
        verb_id:   int
        n_chunks:  int
    """

    def __init__(self, df, data_dir, vqvae, device,
                 window_size=VQVLA_WINDOW_SIZE, max_windows=16):
        self.items = []
        self.verb_to_id = {}
        # verb_to_id is set externally before use
        self._data = []

        self.df = df
        self.data_dir = data_dir
        self.vqvae = vqvae
        self.device = device
        self.window_size = window_size
        self.max_windows = max_windows

    def precompute(self, verb_to_id, cache_path=None):
        """Run all trajectories through VQ encoder once and cache results.

        If cache_path is given and the file exists, load from disk instead of
        re-running the VQ encoder (fast path).  After computing, save to disk.
        """
        self.verb_to_id = verb_to_id
        self._data = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading pre-tokenized cache from {cache_path} ...")
            raw = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Update verb_ids to match current verb_to_id mapping
            for item in raw:
                item['verb_id'] = verb_to_id.get(item['primary_verb'], -1)
                self._data.append(item)
            print(f"  Done. {len(self._data)} trajectories loaded from cache.")
            return

        print(f"  Pre-tokenizing {len(self.df)} trajectories...")
        self.vqvae.eval()

        with torch.no_grad():
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                actions = []
                for i in range(row['start_idx'], row['end_idx'] + 1):
                    path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
                    data = np.load(path, mmap_mode='r')
                    actions.append(np.array(data[ACTION_KEY]))
                actions = np.array(actions, dtype=np.float32)  # (T, 7)

                T = len(actions)
                n_chunks = T // self.window_size
                if n_chunks == 0:
                    padded = np.pad(actions, ((0, self.window_size - T), (0, 0)), mode='edge')
                    windows = padded[np.newaxis]  # (1, 5, 7)
                    n_chunks = 1
                else:
                    windows = actions[:n_chunks * self.window_size].reshape(n_chunks, self.window_size, 7)

                n_chunks = min(n_chunks, self.max_windows)
                windows = windows[:n_chunks]

                x = torch.from_numpy(windows).float().to(self.device)  # (n_chunks, 5, 7)

                # Encode → quantize
                latents = self.vqvae.encode(x).latents           # (n_chunks, 128)
                state_rep = latents.view(latents.size(0), -1, latents.size(1))
                quantized, vq_codes, _ = self.vqvae.vq_layer(state_rep)
                zq = quantized.view(latents.size(0), -1)         # (n_chunks, 128)
                # vq_codes: (n_chunks, 1, n_groups) → (n_chunks, n_groups)
                codes = vq_codes.squeeze(1).long()               # (n_chunks, 4)

                verb_id = verb_to_id.get(row['primary_verb'], -1)
                self._data.append({
                    'zq':          zq.cpu(),
                    'codes':       codes.cpu(),
                    'verb_id':     verb_id,
                    'primary_verb': row['primary_verb'],  # keep for cache reload
                    'n_chunks':    n_chunks,
                })

        print(f"  Done. {len(self._data)} trajectories cached.")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self._data, cache_path)
            print(f"  Cache saved to {cache_path}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        return item['zq'], item['codes'], item['verb_id'], item['n_chunks']


class BinVerbDataset(Dataset):
    """
    Loads CALVIN trajectories and stores raw continuous actions + uniform bin IDs.

    Level 1 analog: raw continuous actions (T, 7) — what information is in the action sequence?
    Level 2 analog: 256-bin token IDs (T, 7)     — what survives discretization?

    Each item uses the same keys as VQVerbDataset for collate compatibility:
        zq:       (T, 7)   float32  — raw continuous actions
        codes:    (T, 7)   int64    — 256-bin indices per action dim
        verb_id:  int
        n_chunks: int      — actual number of timesteps (reuses n_chunks key)
    """

    ACTION_DIM = 7
    N_BINS = 256

    def __init__(self, df, data_dir, max_steps=64):
        self._data = []
        self.df = df
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.action_min = None  # (7,) computed from training data
        self.action_max = None

    def precompute(self, verb_to_id, action_stats=None, cache_path=None):
        """Load and bin-tokenize trajectories, with optional disk cache."""
        self.verb_to_id = verb_to_id
        self._data = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading pre-tokenized cache from {cache_path} ...")
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.action_min = saved['action_min']
            self.action_max = saved['action_max']
            for item in saved['data']:
                item['verb_id'] = verb_to_id.get(item['primary_verb'], -1)
                self._data.append(item)
            print(f"  Done. {len(self._data)} trajectories loaded from cache.")
            return

        print(f"  Loading {len(self.df)} trajectories...")
        traj_actions = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            actions = []
            for i in range(row['start_idx'], row['end_idx'] + 1):
                path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
                data = np.load(path, mmap_mode='r')
                actions.append(np.array(data[ACTION_KEY]))
            traj_actions.append(np.array(actions, dtype=np.float32))

        # Compute or apply action normalization stats
        if action_stats is not None:
            self.action_min, self.action_max = action_stats
        else:
            all_steps = np.concatenate(traj_actions, axis=0)  # (N_total, 7)
            self.action_min = all_steps.min(axis=0)
            self.action_max = all_steps.max(axis=0)
            # Avoid zero-range dims (e.g. constant gripper in some splits)
            same = self.action_max == self.action_min
            self.action_max[same] = self.action_min[same] + 1.0

        for idx, actions in enumerate(traj_actions):
            row = self.df.iloc[idx]
            n_steps = min(len(actions), self.max_steps)
            acts = actions[:n_steps]  # (n_steps, 7)

            # Normalize to [0, 1] using training stats, then bin to [0, N_BINS-1]
            acts_norm = np.clip(
                (acts - self.action_min) / (self.action_max - self.action_min),
                0.0, 1.0,
            )
            bin_ids = np.clip(
                np.floor(acts_norm * self.N_BINS).astype(np.int64),
                0, self.N_BINS - 1,
            )

            verb_id = verb_to_id.get(row['primary_verb'], -1)
            self._data.append({
                'zq':          torch.from_numpy(acts).float(),     # (n_steps, 7)
                'codes':       torch.from_numpy(bin_ids).long(),   # (n_steps, 7)
                'verb_id':     verb_id,
                'primary_verb': row['primary_verb'],
                'n_chunks':    n_steps,
            })

        print(f"  Done. {len(self._data)} trajectories cached.")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                'action_min': self.action_min,
                'action_max': self.action_max,
                'data': self._data,
            }, cache_path)
            print(f"  Cache saved to {cache_path}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        return item['zq'], item['codes'], item['verb_id'], item['n_chunks']


def collate_vq(batch, max_windows):
    zqs, codes, verb_ids, n_chunks_list = zip(*batch)
    B = len(zqs)
    latent_dim = zqs[0].size(1)
    n_groups   = codes[0].size(1)

    zq_pad    = torch.zeros(B, max_windows, latent_dim)
    code_pad  = torch.zeros(B, max_windows, n_groups, dtype=torch.long)
    for i, (zq, code, nc) in enumerate(zip(zqs, codes, n_chunks_list)):
        zq_pad[i, :nc]   = zq[:nc]
        code_pad[i, :nc] = code[:nc]

    return (zq_pad,
            code_pad,
            torch.tensor(verb_ids, dtype=torch.long),
            torch.tensor(n_chunks_list, dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────────

class VerbProbeTransformer(nn.Module):
    """
    Shared transformer architecture for both probe levels.

    mode='latent': input is (B, T, 128) z_q vectors → linear projection → transformer
    mode='token':  input is (B, T, 4)  code IDs     → embedding + sum per chunk → transformer
    """

    def __init__(self, num_verbs, mode, latent_dim=128, n_groups=4,
                 vocab_size=256, d_model=128, nhead=8, num_layers=4,
                 max_seq_len=32, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if mode == 'latent':
            self.input_proj = nn.Linear(latent_dim, d_model)
        else:  # token
            # One embedding per group (RVQ groups encode different residuals)
            self.group_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, d_model) for _ in range(n_groups)
            ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_verbs)

    def forward(self, x, n_chunks):
        """
        x:        (B, T, latent_dim) for mode='latent'
                  (B, T, n_groups)   for mode='token'
        n_chunks: (B,) actual sequence lengths
        """
        B, T = x.shape[:2]

        if self.mode == 'latent':
            tok = self.input_proj(x.float())                     # (B, T, d_model)
        else:
            # Sum group embeddings per chunk position
            tok = sum(self.group_embeddings[g](x[:, :, g])
                      for g in range(x.size(2)))                 # (B, T, d_model)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, d_model)
        seq = torch.cat([cls, tok], dim=1)                       # (B, T+1, d_model)

        # Positional encoding
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0)
        seq = seq + self.pos_embed(pos)

        # Padding mask: True = ignore (pad positions)
        pad_mask = torch.ones(B, T + 1, dtype=torch.bool, device=x.device)
        pad_mask[:, 0] = False  # CLS always visible
        for i, nc in enumerate(n_chunks):
            pad_mask[i, 1:nc + 1] = False  # real chunks visible

        out = self.transformer(seq, src_key_padding_mask=pad_mask)
        cls_out = self.norm(out[:, 0])                           # (B, d_model)
        return self.head(cls_out)                                # (B, num_verbs)


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train=True, max_windows=16):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for zq, codes, verb_ids, n_chunks in loader:
            verb_ids = verb_ids.to(device)
            n_chunks = n_chunks.to(device)

            if model.mode == 'latent':
                x = zq.to(device)
            else:
                x = codes.to(device)

            logits = model(x, n_chunks)
            loss = criterion(logits, verb_ids)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)
            n_batches += 1

    return total_loss / n_batches, 100.0 * correct / max(total, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_probe(mode, train_ds, val_ds, num_verbs, verb_weights,
                device, args, output_dir, max_len):
    """Train one probe (latent or token) and return best val metrics."""
    os.makedirs(output_dir, exist_ok=True)

    collate = lambda b: collate_vq(b, max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate)

    # Infer dims from data (works for both VQ and bin)
    latent_dim = train_ds._data[0]['zq'].size(1)
    n_groups   = train_ds._data[0]['codes'].size(1)

    model = VerbProbeTransformer(
        num_verbs=num_verbs, mode=mode,
        latent_dim=latent_dim, n_groups=n_groups, max_seq_len=max_len + 1,
        d_model=args.d_model, nhead=8, num_layers=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=verb_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    best_metrics = {}

    print(f"\n  Training {mode} probe ({num_verbs} classes, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"    Ep {epoch+1:3d}/{args.epochs}: train {tr_loss:.4f}/{tr_acc:.1f}% | val {va_loss:.4f}/{va_acc:.1f}%")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{tr_loss:.5f}", f"{tr_acc:.2f}",
                                    f"{va_loss:.5f}", f"{va_acc:.2f}", f"{lr:.8f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_metrics = {"val_acc": va_acc, "val_loss": va_loss,
                            "train_acc": tr_acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))

    print(f"  Best val acc ({mode}): {best_val_acc:.2f}% at epoch {best_metrics['epoch']}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        help="bin | vq_vanilla | vq_verb | vq_verb01")
    parser.add_argument("--vqvla_checkpoint_dir", default=None,
                        help="VQ-VLA checkpoint dir (not needed for bin condition)")
    parser.add_argument("--data_dir",  default=DATA_DIR)
    parser.add_argument("--val_dir",   default=VAL_DIR)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--max_windows", type=int,  default=16,
                        help="Max VQ chunks per trajectory (VQ conditions)")
    parser.add_argument("--max_steps",   type=int,  default=64,
                        help="Max timesteps per trajectory (bin condition)")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to save/load pre-tokenized caches. "
                             "Defaults to <vqvla_checkpoint_dir>/vprobe_cache/ "
                             "or results/verb_probe_vq/bin/cache/ for bin.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Condition: {args.condition}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = load_calvin_to_dataframe(args.data_dir)
    val_df   = load_calvin_to_dataframe(args.val_dir)

    if args.min_class_count > 0:
        verb_counts = train_df['primary_verb'].value_counts()
        keep = set(verb_counts[verb_counts >= args.min_class_count].index)
        train_df = train_df[train_df['primary_verb'].isin(keep)].reset_index(drop=True)
        val_df   = val_df[val_df['primary_verb'].isin(keep)].reset_index(drop=True)
        print(f"Kept {len(keep)} verb classes: {sorted(keep)}")

    unique_verbs = sorted(train_df['primary_verb'].unique())
    verb_to_id   = {v: i for i, v in enumerate(unique_verbs)}
    id_to_verb   = {i: v for v, i in verb_to_id.items()}
    num_verbs    = len(verb_to_id)

    # Weighted CE
    counts = train_df['primary_verb'].value_counts()
    weights = torch.zeros(num_verbs)
    for v, i in verb_to_id.items():
        weights[i] = 1.0 / counts.get(v, 1)
    weights = weights / weights.sum() * num_verbs

    # ── Build datasets ────────────────────────────────────────────────────────
    if args.condition == 'bin':
        cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
        train_cache = os.path.join(cache_dir, "train_actions.pt")
        val_cache   = os.path.join(cache_dir, "val_actions.pt")
        max_len = args.max_steps
        level1_label = "continuous actions"
        level2_label = "256-bin token IDs"

        print(f"\nPre-tokenizing train set (cache: {train_cache}):")
        train_ds = BinVerbDataset(train_df, args.data_dir, max_steps=args.max_steps)
        train_ds.precompute(verb_to_id, action_stats=None, cache_path=train_cache)

        print(f"Pre-tokenizing val set (cache: {val_cache}):")
        val_ds = BinVerbDataset(val_df, args.val_dir, max_steps=args.max_steps)
        val_ds.precompute(verb_to_id,
                          action_stats=(train_ds.action_min, train_ds.action_max),
                          cache_path=val_cache)

    else:
        if args.vqvla_checkpoint_dir is None:
            raise ValueError("--vqvla_checkpoint_dir is required for VQ conditions")
        print(f"VQ-VLA checkpoint: {args.vqvla_checkpoint_dir}")

        cache_dir = args.cache_dir or os.path.join(args.vqvla_checkpoint_dir, "vprobe_cache")
        train_cache = os.path.join(cache_dir, "train_zq_codes.pt")
        val_cache   = os.path.join(cache_dir, "val_zq_codes.pt")
        max_len = args.max_windows
        level1_label = "z_q latent"
        level2_label = "token IDs"

        print("\nLoading VQ-VLA...")
        vqvae = load_vqvae(args.vqvla_checkpoint_dir).to(device)

        print(f"\nPre-tokenizing train set (cache: {train_cache}):")
        train_ds = VQVerbDataset(train_df, args.data_dir, vqvae, device,
                                 max_windows=args.max_windows)
        train_ds.precompute(verb_to_id, cache_path=train_cache)

        print(f"Pre-tokenizing val set (cache: {val_cache}):")
        val_ds = VQVerbDataset(val_df, args.val_dir, vqvae, device,
                               max_windows=args.max_windows)
        val_ds.precompute(verb_to_id, cache_path=val_cache)

    # Filter out unknown verbs (verb_id == -1)
    train_ds._data = [d for d in train_ds._data if d['verb_id'] >= 0]
    val_ds._data   = [d for d in val_ds._data   if d['verb_id'] >= 0]
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Train Level 1: continuous/z_q latent probe ───────────────────────────
    print("\n" + "="*60)
    print(f"LEVEL 1: {level1_label} probe")
    print("="*60)
    latent_dir = os.path.join(args.output_dir, "latent")
    latent_metrics = train_probe('latent', train_ds, val_ds, num_verbs, weights,
                                 device, args, latent_dir, max_len=max_len)

    # ── Train Level 2: token ID probe ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"LEVEL 2: {level2_label} probe")
    print("="*60)
    token_dir = os.path.join(args.output_dir, "token")
    token_metrics = train_probe('token', train_ds, val_ds, num_verbs, weights,
                                device, args, token_dir, max_len=max_len)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "condition":     args.condition,
        "num_verbs":     num_verbs,
        "id_to_verb":    {str(k): v for k, v in id_to_verb.items()},
        "level1_label":  level1_label,
        "level2_label":  level2_label,
        "level1_latent": latent_metrics,
        "level2_token":  token_metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"SUMMARY — {args.condition}")
    print(f"  Level 1 ({level1_label}): {latent_metrics['val_acc']:.2f}% val acc")
    print(f"  Level 2 ({level2_label}):  {token_metrics['val_acc']:.2f}% val acc")
    print("="*60)
    print(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
