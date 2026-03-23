"""Stage 1: Fine-tune VQ-VLA action tokenizer on CALVIN with optional verb loss.

Fine-tunes the pretrained VQ-VLA causal conv VAE + ResidualVQ tokenizer on
CALVIN split D actions. Two conditions:
  - Vanilla (lambda=0): recon + vq loss only (domain adaptation control)
  - Verb-decodable (lambda>0): recon + vq + lambda * verb_CE

The verb classifier operates on mean-pooled quantized latents across all
5-step windows in a trajectory.

Usage:
    # Vanilla fine-tuning (control)
    python -m openvla_experiment.scripts.finetune_tokenizer \
        --verb_loss_weight 0.0 --tag vanilla

    # Verb-decodable fine-tuning
    python -m openvla_experiment.scripts.finetune_tokenizer \
        --verb_loss_weight 0.5 --tag verb_l0.5

Outputs:
    checkpoints/vqvla_ft_{tag}/          -- VQ-VLA weights (ActionVQVAEPE state_dict)
    checkpoints/vqvla_ft_{tag}/full.pth  -- Full checkpoint (VQ-VLA + verb head + optimizer)
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
from tqdm import tqdm

# Project root imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import load_calvin_to_dataframe
from config import DATA_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE
from tokenization.vqvae_tokenizer import (
    VQVLA_WINDOW_SIZE, VQVLA_NUM_TOKENS, VQVLA_VOCAB_SIZE,
    VQVLA_CONFIG_DIR, VQVLA_CHECKPOINT_PATH,
)

# ─── Dataset ──────────────────────────────────────────────────────────────────

class CalvinVQVLADataset(Dataset):
    """CALVIN trajectories chunked into 5-step windows for VQ-VLA training."""

    def __init__(self, df, data_dir, window_size=VQVLA_WINDOW_SIZE,
                 verb_to_id=None, max_windows=16):
        self.df = df
        self.data_dir = data_dir
        self.window_size = window_size
        self.verb_to_id = verb_to_id or {}
        self.max_windows = max_windows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        actions = []
        for i in range(row['start_idx'], row['end_idx'] + 1):
            path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
            data = np.load(path, mmap_mode='r')
            actions.append(np.array(data[ACTION_KEY]))
        actions = np.array(actions, dtype=np.float32)  # (T, 7)

        T = actions.shape[0]
        n_windows = T // self.window_size
        if n_windows == 0:
            # Pad short trajectories
            padded = np.pad(actions, ((0, self.window_size - T), (0, 0)), mode='edge')
            windows = padded[np.newaxis]  # (1, window_size, 7)
            n_windows = 1
        else:
            usable = n_windows * self.window_size
            windows = actions[:usable].reshape(n_windows, self.window_size, 7)

        # Clip to max_windows
        if n_windows > self.max_windows:
            windows = windows[:self.max_windows]
            n_windows = self.max_windows

        # Pad to max_windows for batching
        padded_windows = np.zeros((self.max_windows, self.window_size, 7), dtype=np.float32)
        padded_windows[:n_windows] = windows

        verb_id = self.verb_to_id.get(row['primary_verb'], 0)
        return (torch.from_numpy(padded_windows),
                torch.tensor(verb_id, dtype=torch.long),
                torch.tensor(n_windows, dtype=torch.long))


# ─── Verb classification head ────────────────────────────────────────────────

class VerbHead(nn.Module):
    """MLP verb classifier on mean-pooled quantized latents."""

    def __init__(self, latent_dim, num_verbs, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_verbs),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ────────────────────────────────────────────────────────────────

def forward_vqvla(vqvae, windows_batch, n_windows_batch, device):
    """Forward pass through VQ-VLA, returning losses and quantized latents.

    Args:
        vqvae: ActionVQVAEPE model (the inner .vqvae of the wrapper)
        windows_batch: (B, max_windows, 5, 7) padded windows
        n_windows_batch: (B,) number of real windows per trajectory
    Returns:
        dict with recon_loss, vq_loss, traj_reprs (B, 128)
    """
    B = windows_batch.size(0)
    max_win = windows_batch.size(1)

    # Collect all real windows into one batch
    all_windows = []
    window_counts = []
    for i in range(B):
        nw = n_windows_batch[i].item()
        all_windows.append(windows_batch[i, :nw])  # (nw, 5, 7)
        window_counts.append(nw)
    all_windows_cat = torch.cat(all_windows, dim=0).to(device)  # (total, 5, 7)

    # Encode → quantize → decode
    latents = vqvae.encode(all_windows_cat).latents  # (total, 128)
    state_rep = latents.view(latents.size(0), -1, latents.size(1))
    quantized, vq_codes, vq_losses = vqvae.vq_layer(state_rep)
    quantized_flat = quantized.view(latents.size(0), -1)  # (total, 128)

    decoded = vqvae.decode(quantized_flat)  # (total, 5, 7)
    recon_loss = F.mse_loss(decoded, all_windows_cat.to(decoded.dtype))
    vq_loss = vq_losses.sum()

    # Mean pool quantized latents per trajectory
    traj_reprs = []
    offset = 0
    for nw in window_counts:
        traj_q = quantized_flat[offset:offset + nw]  # (nw, 128)
        traj_reprs.append(traj_q.mean(dim=0))
        offset += nw
    traj_reprs = torch.stack(traj_reprs)  # (B, 128)

    return {
        'recon_loss': recon_loss,
        'vq_loss': vq_loss,
        'traj_reprs': traj_reprs,
        'vq_codes': vq_codes,
        'total_windows': all_windows_cat.size(0),
    }


def train_epoch(vqvae, verb_head, loader, optimizer, verb_criterion,
                verb_loss_weight, device, max_grad_norm=1.0):
    vqvae.train()
    if verb_head is not None:
        verb_head.train()

    total_recon = total_vq = total_verb = 0.0
    correct = total = 0
    n_batches = 0

    for windows, verb_ids, n_windows in loader:
        verb_ids = verb_ids.to(device)

        result = forward_vqvla(vqvae, windows, n_windows, device)
        loss = result['recon_loss'] + 5 * result['vq_loss']

        if verb_head is not None and verb_loss_weight > 0:
            verb_logits = verb_head(result['traj_reprs'])
            verb_loss = verb_criterion(verb_logits, verb_ids)
            loss = loss + verb_loss_weight * verb_loss
            total_verb += verb_loss.item()
            preds = verb_logits.argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            all_params = list(vqvae.parameters())
            if verb_head is not None:
                all_params += list(verb_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        total_recon += result['recon_loss'].item()
        total_vq += result['vq_loss'].item()
        n_batches += 1

    acc = 100.0 * correct / max(total, 1)
    return {
        'recon': total_recon / n_batches,
        'vq': total_vq / n_batches,
        'verb': total_verb / n_batches if verb_loss_weight > 0 else 0,
        'verb_acc': acc,
    }


@torch.no_grad()
def eval_epoch(vqvae, verb_head, loader, verb_criterion, verb_loss_weight,
               device):
    vqvae.eval()
    if verb_head is not None:
        verb_head.eval()

    total_recon = total_vq = total_verb = 0.0
    correct = total = 0
    all_codes = []
    n_batches = 0

    for windows, verb_ids, n_windows in loader:
        verb_ids = verb_ids.to(device)

        result = forward_vqvla(vqvae, windows, n_windows, device)

        if verb_head is not None and verb_loss_weight > 0:
            verb_logits = verb_head(result['traj_reprs'])
            verb_loss = verb_criterion(verb_logits, verb_ids)
            total_verb += verb_loss.item()
            preds = verb_logits.argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)

        total_recon += result['recon_loss'].item()
        total_vq += result['vq_loss'].item()
        all_codes.append(result['vq_codes'].cpu())
        n_batches += 1

    # Codebook utilization — vq_codes shape is (total_windows, 1, 4)
    all_codes = torch.cat(all_codes, dim=0)  # (N, 1, 4)
    all_codes = all_codes.squeeze(1)         # (N, 4)
    used_codes = set()
    for q in range(min(VQVLA_NUM_TOKENS, all_codes.size(-1))):
        used_codes.update(all_codes[:, q].unique().tolist())
    utilization = len(used_codes)

    acc = 100.0 * correct / max(total, 1)
    return {
        'recon': total_recon / n_batches,
        'vq': total_vq / n_batches,
        'verb': total_verb / n_batches if verb_loss_weight > 0 else 0,
        'verb_acc': acc,
        'codebook_used': utilization,
        'codebook_total': VQVLA_VOCAB_SIZE,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VQ-VLA tokenizer")
    parser.add_argument("--tag", type=str, required=True,
                        help="Experiment tag for checkpoint naming")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--config_dir", type=str, default=VQVLA_CONFIG_DIR)
    parser.add_argument("--pretrained_path", type=str,
                        default=VQVLA_CHECKPOINT_PATH)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_windows", type=int, default=16,
                        help="Max 5-step windows per trajectory")
    # Verb loss
    parser.add_argument("--verb_loss_weight", type=float, default=0.0,
                        help="Lambda for verb CE loss (0 = vanilla fine-tuning)")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--weighted_verb_loss", action="store_true",
                        default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience in epochs (0 = disabled)")
    # Output
    parser.add_argument("--save_dir", type=str,
                        default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading training data...")
    train_df = load_calvin_to_dataframe(args.data_dir)
    print("Loading validation data...")
    val_df = load_calvin_to_dataframe(args.val_dir)

    # Filter sparse verb classes
    if args.min_class_count > 0:
        verb_counts = train_df['primary_verb'].value_counts()
        keep_verbs = set(verb_counts[verb_counts >= args.min_class_count].index)
        train_df = train_df[train_df['primary_verb'].isin(keep_verbs)].reset_index(drop=True)
        val_df = val_df[val_df['primary_verb'].isin(keep_verbs)].reset_index(drop=True)
        print("Filtered to {} verb classes, {} train / {} val".format(
            len(keep_verbs), len(train_df), len(val_df)))

    unique_verbs = sorted(train_df['primary_verb'].unique())
    verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
    id_to_verb = {i: v for v, i in verb_to_id.items()}
    num_verbs = len(verb_to_id)
    print("Verb classes ({}): {}".format(num_verbs, unique_verbs))

    train_ds = CalvinVQVLADataset(train_df, args.data_dir,
                                   verb_to_id=verb_to_id,
                                   max_windows=args.max_windows)
    val_ds = CalvinVQVLADataset(val_df, args.val_dir,
                                 verb_to_id=verb_to_id,
                                 max_windows=args.max_windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # ── Load pretrained VQ-VLA ───────────────────────────────────────────────
    print("Loading pretrained VQ-VLA from {}...".format(args.pretrained_path))
    from tokenization.vqvla import ActionVQVAELossWrapper

    wrapper = ActionVQVAELossWrapper(
        model_path=args.config_dir,
        checkpoint_path=args.pretrained_path,
        is_eval=False,   # Not eval — we want to train
        freeze=False,    # Not frozen — we fine-tune everything
        use_action_type_pe=True,
        use_time_pe=True,
    )
    vqvae = wrapper.vqvae.to(device)
    print("VQ-VLA loaded ({:.1f}M params)".format(
        sum(p.numel() for p in vqvae.parameters()) / 1e6))

    # ── Verb head (only if lambda > 0) ───────────────────────────────────────
    verb_head = None
    verb_criterion = None
    if args.verb_loss_weight > 0:
        latent_dim = 128  # Fixed by VQ-VLA architecture
        verb_head = VerbHead(latent_dim, num_verbs).to(device)
        print("Verb head: {} -> {} classes".format(latent_dim, num_verbs))

        if args.weighted_verb_loss:
            class_counts = train_df['primary_verb'].value_counts()
            weights = torch.zeros(num_verbs)
            for verb, cid in verb_to_id.items():
                weights[cid] = 1.0 / class_counts.get(verb, 1)
            weights = weights / weights.sum() * num_verbs
            verb_criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            verb_criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    params = list(vqvae.parameters())
    if verb_head is not None:
        params += list(verb_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, "vqvla_ft_{}".format(args.tag))
    os.makedirs(save_dir, exist_ok=True)
    best_val_metric = float('inf')  # Track best val recon (or verb loss)
    best_verb_acc = 0.0

    # CSV logging
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = ["epoch", "train_recon", "train_vq", "train_verb", "train_verb_acc",
                  "val_recon", "val_vq", "val_verb", "val_verb_acc", "codebook_used", "lr"]
    csv_writer.writerow(csv_header)

    patience_counter = 0

    print("\nTraining for {} epochs (lambda={})".format(
        args.epochs, args.verb_loss_weight))
    print("Save dir: {}".format(save_dir))
    print("=" * 70)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            vqvae, verb_head, train_loader, optimizer, verb_criterion,
            args.verb_loss_weight, device, max_grad_norm=args.max_grad_norm)
        scheduler.step()

        val_metrics = eval_epoch(
            vqvae, verb_head, val_loader, verb_criterion,
            args.verb_loss_weight, device)

        # Print progress
        line = "Epoch {:3d}/{}: train recon={:.5f} vq={:.5f}".format(
            epoch + 1, args.epochs, train_metrics['recon'], train_metrics['vq'])
        if args.verb_loss_weight > 0:
            line += " verb={:.4f} acc={:.1f}%".format(
                train_metrics['verb'], train_metrics['verb_acc'])
        line += " | val recon={:.5f} vq={:.5f}".format(
            val_metrics['recon'], val_metrics['vq'])
        if args.verb_loss_weight > 0:
            line += " verb={:.4f} acc={:.1f}%".format(
                val_metrics['verb'], val_metrics['verb_acc'])
        line += " | codes={}/{}".format(
            val_metrics['codebook_used'], val_metrics['codebook_total'])
        print(line)

        # Save best checkpoint (by val recon loss for vanilla, verb acc for verb)
        if args.verb_loss_weight > 0:
            is_best = val_metrics['verb_acc'] > best_verb_acc
            if is_best:
                best_verb_acc = val_metrics['verb_acc']
        else:
            is_best = val_metrics['recon'] < best_val_metric
            if is_best:
                best_val_metric = val_metrics['recon']

        if is_best:
            patience_counter = 0
            # Save VQ-VLA inner model weights (compatible with get_code / decode)
            torch.save(vqvae.state_dict(),
                       os.path.join(save_dir, "vqvla_weights.pth"))
            # Save full checkpoint for resuming
            full_ckpt = {
                'epoch': epoch + 1,
                'vqvae_state_dict': vqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args),
                'verb_to_id': verb_to_id,
                'id_to_verb': id_to_verb,
                'num_verbs': num_verbs,
            }
            if verb_head is not None:
                full_ckpt['verb_head_state_dict'] = verb_head.state_dict()
            torch.save(full_ckpt, os.path.join(save_dir, "full.pth"))
            print("  -> Saved best checkpoint (epoch {})".format(epoch + 1))
        else:
            patience_counter += 1

        # CSV logging
        current_lr = optimizer.param_groups[0]['lr']
        csv_writer.writerow([
            epoch + 1,
            "{:.6f}".format(train_metrics['recon']),
            "{:.6f}".format(train_metrics['vq']),
            "{:.6f}".format(train_metrics['verb']),
            "{:.2f}".format(train_metrics['verb_acc']),
            "{:.6f}".format(val_metrics['recon']),
            "{:.6f}".format(val_metrics['vq']),
            "{:.6f}".format(val_metrics['verb']),
            "{:.2f}".format(val_metrics['verb_acc']),
            val_metrics['codebook_used'],
            "{:.8f}".format(current_lr),
        ])
        csv_file.flush()

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print("Early stopping at epoch {} ({} epochs without improvement)".format(
                epoch + 1, patience_counter))
            break

    csv_file.close()

    # Save config
    config = {
        'tag': args.tag,
        'verb_loss_weight': args.verb_loss_weight,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'max_windows': args.max_windows,
        'num_verbs': num_verbs,
        'verb_to_id': verb_to_id,
        'pretrained_path': args.pretrained_path,
        'best_val_recon': float(best_val_metric) if args.verb_loss_weight == 0 else None,
        'best_verb_acc': float(best_verb_acc) if args.verb_loss_weight > 0 else None,
        'final_codebook_used': val_metrics['codebook_used'],
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("\nDone. Best checkpoint saved to {}".format(save_dir))


if __name__ == "__main__":
    main()
