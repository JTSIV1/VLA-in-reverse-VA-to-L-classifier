"""train_verb_probe_level3.py

Level 3 verb decodability probe: classify verbs from the fine-tuned LLM's action token embeddings.

Does NOT run the full 7B model. Instead:
  1. Loads only embed_tokens.weight from the fine-tuned checkpoint safetensors
  2. For each CALVIN val trajectory, tokenizes actions (same as during fine-tuning) → LLM token IDs
  3. Looks up the 4096-d embedding for each action token ID
  4. Aggregates per timestep (bin) or per chunk (VQ): mean over 7 dims / 4 groups → (T, 4096)
  5. Trains two probes on these embedding sequences:
       a. Transformer probe: Linear(4096, d_model) → transformer CLS → Linear(d_model, num_verbs)
       b. Linear probe:      mean-pool over time   → Linear(4096, num_verbs)

Tokenization (must match fine-tuning exactly):
  bin: clip action to [-1, 1], digitize with np.linspace(-1, 1, 256) → token_id = tokenizer_len - bin_idx
  vq:  load cached code IDs from <vqvla_checkpoint_dir>/vprobe_cache/  → token_id = tokenizer_len - 1 - code_id

Usage:
  python -u -m openvla_experiment.scripts.train_verb_probe_level3 \\
      --condition bin \\
      --openvla_checkpoint_dir runs/openvla/openvla-7b+...+calvin_bin... \\
      --output_dir results/verb_probe_level3/bin

  python -u -m openvla_experiment.scripts.train_verb_probe_level3 \\
      --condition vq_verb \\
      --openvla_checkpoint_dir runs/openvla/openvla-7b+...+calvin_vq_verb... \\
      --vqvla_checkpoint_dir checkpoints/vqvla_ft_verb_l0.5 \\
      --output_dir results/verb_probe_level3/vq_verb
"""

import csv
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import load_calvin_to_dataframe
from config import DATA_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE
from tokenization.vqvae_tokenizer import VQVLA_WINDOW_SIZE


# ── Embedding table loader ─────────────────────────────────────────────────────

def load_action_embeddings(checkpoint_dir):
    """Load only the action token rows from embed_tokens.weight.

    Returns:
        embed:   (256, 4096) float32 tensor — the 256 action token embeddings
        tok_len: int — tokenizer_len (= vocab_size used during fine-tuning), used to compute token IDs
    """
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)

    shard = idx["weight_map"]["language_model.model.embed_tokens.weight"]
    weights = load_file(os.path.join(checkpoint_dir, shard))
    embed_full = weights["language_model.model.embed_tokens.weight"].float()  # (vocab_size, 4096)

    # The tokenizer uses: action_token_begin_idx = tokenizer_len - (n_bins + 1)
    # where tokenizer_len = tokenizer.vocab_size = 32000 for Llama-2-based models
    # Action tokens span [begin+1 ... begin+256] = [31744 ... 31999]
    # But the actual token IDs used are tokenizer_len - bin_idx, where bin_idx ∈ [1, 256]
    # → token IDs ∈ [32000-256, 32000-1] = [31744, 31999]
    # We load the 256 rows at positions [31744, 31999]
    # Also need tokenizer_len to convert code IDs → LLM token IDs.

    # Infer tokenizer_len from config.json
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    # Llama-2 based: vocab_size field or text_config.vocab_size
    tok_len = cfg.get("text_config", cfg).get("vocab_size", embed_full.shape[0])

    # Action token IDs: tokenizer_len - bin_idx  (bin_idx ∈ [1, 256])
    # → rows [tok_len - 256, tok_len - 1]
    n_bins = 256
    start = tok_len - n_bins
    action_embed = embed_full[start : tok_len].clone()  # (256, 4096)
    print(f"Loaded action embeddings: shape={action_embed.shape}, "
          f"from rows [{start}, {tok_len}) of vocab size {embed_full.shape[0]}")
    print(f"tokenizer_len (= {tok_len}), action_token_begin_idx = {tok_len - n_bins - 1}")
    return action_embed, tok_len


# ── Dataset ────────────────────────────────────────────────────────────────────

def _bin_to_local_idx(actions, n_bins=256):
    """Convert continuous actions (T, 7) in [-1, 1] to local embedding indices [0, 255].

    Mirrors ActionTokenizer.__call__:
        bin_idx = np.digitize(action, linspace(-1, 1, n_bins))   # [1, 256]
        token_id = tokenizer_len - bin_idx                        # e.g. 32000 - bin_idx
        local_idx = n_bins - bin_idx = token_id - (tokenizer_len - n_bins) # [0, 255]
    """
    bins = np.linspace(-1.0, 1.0, n_bins)
    actions = np.clip(actions, -1.0, 1.0)
    bin_idx = np.digitize(actions, bins)          # (T, 7), values in [1, 256]
    local_idx = n_bins - bin_idx                  # [0, 255];  bin 1 → 255, bin 256 → 0
    return np.clip(local_idx, 0, n_bins - 1).astype(np.int64)


def _vq_codes_to_local_idx(codes):
    """Convert VQ code IDs (n_chunks, 4) [0, 255] to local embedding indices.

    CalvinVQActionTokenizer.__call__:
        token_id = tokenizer_len - 1 - code_id
        local_idx = n_bins - 1 - code_id    (same formula, shifted by 1 vs bin)
    """
    return (255 - codes).astype(np.int64)


class Level3Dataset(Dataset):
    """
    Builds trajectory-level embedding sequences for the Level 3 probe.

    Each item:
        emb_seq:  (T_or_chunks, 4096) float32  — mean action token embedding per timestep/chunk
        verb_id:  int
        n_len:    int  — actual sequence length
    """

    def __init__(self, df, data_dir, action_embed, condition,
                 max_len, vq_cache_path=None):
        """
        action_embed: (256, 4096) — action token embedding table slice
        condition:    'bin' | 'vq_*'
        max_len:      64 (bin) or 16 (vq)
        vq_cache_path: path to vprobe_cache/*.pt file (for vq conditions)
        """
        self._data = []
        self.df = df
        self.data_dir = data_dir
        self.action_embed = action_embed   # (256, 4096)
        self.condition = condition
        self.max_len = max_len
        self.vq_cache_path = vq_cache_path

    def precompute(self, verb_to_id, cache_path=None):
        """Build embedding sequences for all trajectories, with disk cache."""
        self._data = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading Level 3 cache from {cache_path} ...")
            raw = torch.load(cache_path, map_location="cpu", weights_only=False)
            for item in raw:
                item["verb_id"] = verb_to_id.get(item["primary_verb"], -1)
                self._data.append(item)
            print(f"  Done. {len(self._data)} trajectories loaded.")
            return

        # ── bin: tokenize actions from .npz files ─────────────────────────────
        if self.condition == "bin":
            print(f"  Building bin embedding sequences for {len(self.df)} trajectories...")
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                actions = []
                for i in range(row["start_idx"], row["end_idx"] + 1):
                    path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
                    data = np.load(path, mmap_mode="r")
                    actions.append(np.array(data[ACTION_KEY]))
                actions = np.array(actions, dtype=np.float32)  # (T, 7)

                T = min(len(actions), self.max_len)
                acts = actions[:T]

                # → local embedding indices (T, 7), each in [0, 255]
                local_idx = _bin_to_local_idx(acts)  # (T, 7)
                # look up embeddings: (T, 7, 4096) → mean over 7 dims → (T, 4096)
                emb = self.action_embed[local_idx]      # (T, 7, 4096)
                emb_seq = emb.mean(dim=1)               # (T, 4096)

                verb_id = verb_to_id.get(row["primary_verb"], -1)
                self._data.append({
                    "emb_seq":     emb_seq,
                    "verb_id":     verb_id,
                    "primary_verb": row["primary_verb"],
                    "n_len":       T,
                })

        # ── vq: reuse cached code IDs, convert to embeddings ──────────────────
        else:
            assert self.vq_cache_path and os.path.exists(self.vq_cache_path), \
                f"VQ code cache not found: {self.vq_cache_path}. Run train_verb_probe_vq.py first."
            print(f"  Loading VQ code cache from {self.vq_cache_path} ...")
            raw_cache = torch.load(self.vq_cache_path, map_location="cpu", weights_only=False)
            print(f"  Building VQ embedding sequences for {len(raw_cache)} trajectories...")

            for item in raw_cache:
                codes = item["codes"].numpy()      # (n_chunks, 4), int
                n_chunks = min(item["n_chunks"], self.max_len)
                codes = codes[:n_chunks]

                # → local embedding indices (n_chunks, 4), each in [0, 255]
                local_idx = _vq_codes_to_local_idx(codes)  # (n_chunks, 4)
                emb = self.action_embed[local_idx]          # (n_chunks, 4, 4096)
                emb_seq = emb.mean(dim=1)                   # (n_chunks, 4096)

                verb_id = verb_to_id.get(item["primary_verb"], -1)
                self._data.append({
                    "emb_seq":     emb_seq,
                    "verb_id":     verb_id,
                    "primary_verb": item["primary_verb"],
                    "n_len":       n_chunks,
                })

        print(f"  Done. {len(self._data)} trajectories cached.")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self._data, cache_path)
            print(f"  Level 3 cache saved to {cache_path}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        return item["emb_seq"], item["verb_id"], item["n_len"]


def collate_l3(batch, max_len):
    embs, verb_ids, n_lens = zip(*batch)
    B = len(embs)
    emb_dim = embs[0].size(1)

    emb_pad = torch.zeros(B, max_len, emb_dim)
    for i, (emb, nl) in enumerate(zip(embs, n_lens)):
        emb_pad[i, :nl] = emb[:nl]

    return (emb_pad,
            torch.tensor(verb_ids, dtype=torch.long),
            torch.tensor(n_lens, dtype=torch.long))


# ── Models ─────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Mean-pool over time → Linear(4096, num_verbs)."""

    def __init__(self, emb_dim, num_verbs):
        super().__init__()
        self.head = nn.Linear(emb_dim, num_verbs)

    def forward(self, x, n_lens):
        """x: (B, T, emb_dim), n_lens: (B,)"""
        B, T, D = x.shape
        mask = torch.zeros(B, T, device=x.device)
        for i, nl in enumerate(n_lens):
            mask[i, :nl] = 1.0
        # mean pool over valid positions
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        x_mean = (x * mask.unsqueeze(-1)).sum(1) / denom  # (B, D)
        return self.head(x_mean.float())


class TransformerProbe(nn.Module):
    """Linear(emb_dim, d_model) → CLS + TransformerEncoder → Linear(d_model, num_verbs)."""

    def __init__(self, emb_dim, num_verbs, d_model=128, nhead=8, num_layers=4,
                 max_seq_len=65, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed  = nn.Embedding(max_seq_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_verbs)

    def forward(self, x, n_lens):
        """x: (B, T, emb_dim), n_lens: (B,)"""
        B, T = x.shape[:2]
        tok = self.input_proj(x.float())                  # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)                # (B, T+1, d_model)
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0)
        seq = seq + self.pos_embed(pos)

        pad_mask = torch.ones(B, T + 1, dtype=torch.bool, device=x.device)
        pad_mask[:, 0] = False
        for i, nl in enumerate(n_lens):
            pad_mask[i, 1:nl + 1] = False

        out = self.transformer(seq, src_key_padding_mask=pad_mask)
        cls_out = self.norm(out[:, 0])
        return self.head(cls_out)


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total, n_batches = 0.0, 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for emb, verb_ids, n_lens in loader:
            emb      = emb.to(device)
            verb_ids = verb_ids.to(device)
            n_lens   = n_lens.to(device)

            logits = model(emb, n_lens)
            loss   = criterion(logits, verb_ids)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == verb_ids).sum().item()
            total      += verb_ids.size(0)
            n_batches  += 1

    return total_loss / n_batches, 100.0 * correct / max(total, 1)


def train_probe(model_type, train_ds, val_ds, num_verbs, verb_weights,
                device, args, output_dir, max_len, emb_dim):
    os.makedirs(output_dir, exist_ok=True)
    collate = lambda b: collate_l3(b, max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)

    if model_type == "linear":
        model = LinearProbe(emb_dim, num_verbs).to(device)
    else:
        model = TransformerProbe(emb_dim, num_verbs, d_model=args.d_model,
                                 nhead=8, num_layers=4,
                                 max_seq_len=max_len + 1).to(device)

    criterion = nn.CrossEntropyLoss(weight=verb_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc, best_metrics = 0.0, {}
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  Training {model_type} probe ({num_verbs} cls, {n_params:.2f}M params)")

    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step()

        print(f"    Ep {epoch+1:3d}/{args.epochs}: train {tr_loss:.4f}/{tr_acc:.1f}% | "
              f"val {va_loss:.4f}/{va_acc:.1f}%")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{tr_loss:.5f}", f"{tr_acc:.2f}",
                                    f"{va_loss:.5f}", f"{va_acc:.2f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_metrics = {"val_acc": va_acc, "val_loss": va_loss,
                            "train_acc": tr_acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))

    print(f"  Best val acc ({model_type}): {best_val_acc:.2f}% at epoch {best_metrics['epoch']}")
    return best_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        help="bin | vq_vanilla | vq_verb | vq_verb01")
    parser.add_argument("--openvla_checkpoint_dir", required=True,
                        help="Fine-tuned OpenVLA checkpoint (for embed_tokens)")
    parser.add_argument("--vqvla_checkpoint_dir", default=None,
                        help="VQ-VLA tokenizer checkpoint (for vq_* conditions)")
    parser.add_argument("--data_dir",  default=DATA_DIR)
    parser.add_argument("--val_dir",   default=VAL_DIR)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--max_steps",  type=int,   default=64,
                        help="Max timesteps per trajectory (bin)")
    parser.add_argument("--max_windows", type=int,  default=16,
                        help="Max chunks per trajectory (vq)")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--cache_dir", default=None,
                        help="Cache dir for precomputed embedding sequences")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Condition: {args.condition}")
    print(f"OpenVLA checkpoint: {args.openvla_checkpoint_dir}")

    # ── Load embedding table (just action token rows) ──────────────────────────
    action_embed, tok_len = load_action_embeddings(args.openvla_checkpoint_dir)
    emb_dim = action_embed.shape[1]   # 4096
    print(f"Action embedding dim: {emb_dim}")

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df = load_calvin_to_dataframe(args.data_dir)
    val_df   = load_calvin_to_dataframe(args.val_dir)

    if args.min_class_count > 0:
        verb_counts = train_df["primary_verb"].value_counts()
        keep = set(verb_counts[verb_counts >= args.min_class_count].index)
        train_df = train_df[train_df["primary_verb"].isin(keep)].reset_index(drop=True)
        val_df   = val_df[val_df["primary_verb"].isin(keep)].reset_index(drop=True)
        print(f"Kept {len(keep)} verb classes: {sorted(keep)}")

    unique_verbs = sorted(train_df["primary_verb"].unique())
    verb_to_id   = {v: i for i, v in enumerate(unique_verbs)}
    id_to_verb   = {i: v for v, i in verb_to_id.items()}
    num_verbs    = len(verb_to_id)

    counts  = train_df["primary_verb"].value_counts()
    weights = torch.zeros(num_verbs)
    for v, i in verb_to_id.items():
        weights[i] = 1.0 / counts.get(v, 1)
    weights = weights / weights.sum() * num_verbs

    # ── Build datasets ─────────────────────────────────────────────────────────
    if args.condition == "bin":
        max_len = args.max_steps
        cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
        vq_train_cache = vq_val_cache = None
    else:
        max_len = args.max_windows
        assert args.vqvla_checkpoint_dir, "--vqvla_checkpoint_dir required for vq_* conditions"
        vq_cache_root = os.path.join(args.vqvla_checkpoint_dir, "vprobe_cache")
        vq_train_cache = os.path.join(vq_cache_root, "train_zq_codes.pt")
        vq_val_cache   = os.path.join(vq_cache_root, "val_zq_codes.pt")
        cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")

    train_l3_cache = os.path.join(cache_dir, "train_emb.pt")
    val_l3_cache   = os.path.join(cache_dir, "val_emb.pt")

    print(f"\nBuilding train embedding dataset (cache: {train_l3_cache}):")
    train_ds = Level3Dataset(train_df, args.data_dir, action_embed,
                             args.condition, max_len, vq_cache_path=vq_train_cache)
    train_ds.precompute(verb_to_id, cache_path=train_l3_cache)

    print(f"Building val embedding dataset (cache: {val_l3_cache}):")
    val_ds = Level3Dataset(val_df, args.val_dir, action_embed,
                           args.condition, max_len, vq_cache_path=vq_val_cache)
    val_ds.precompute(verb_to_id, cache_path=val_l3_cache)

    train_ds._data = [d for d in train_ds._data if d["verb_id"] >= 0]
    val_ds._data   = [d for d in val_ds._data   if d["verb_id"] >= 0]
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Train probes ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LEVEL 3a: Linear probe (mean-pool → linear)")
    print("="*60)
    linear_metrics = train_probe(
        "linear", train_ds, val_ds, num_verbs, weights,
        device, args,
        output_dir=os.path.join(args.output_dir, "linear"),
        max_len=max_len, emb_dim=emb_dim,
    )

    print("\n" + "="*60)
    print("LEVEL 3b: Transformer probe")
    print("="*60)
    transformer_metrics = train_probe(
        "transformer", train_ds, val_ds, num_verbs, weights,
        device, args,
        output_dir=os.path.join(args.output_dir, "transformer"),
        max_len=max_len, emb_dim=emb_dim,
    )

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "condition":       args.condition,
        "openvla_ckpt":    args.openvla_checkpoint_dir,
        "num_verbs":       num_verbs,
        "id_to_verb":      {str(k): v for k, v in id_to_verb.items()},
        "level3a_linear":  linear_metrics,
        "level3b_transformer": transformer_metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"SUMMARY — {args.condition} (Level 3)")
    print(f"  3a Linear (mean-pool): {linear_metrics['val_acc']:.2f}% val acc")
    print(f"  3b Transformer:        {transformer_metrics['val_acc']:.2f}% val acc")
    print("="*60)
    print(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
