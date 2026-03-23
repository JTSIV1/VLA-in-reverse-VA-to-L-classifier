"""Fine-tune VQ-VLA action tokenizer with CLIP contrastive loss.

Same as finetune_tokenizer.py but replaces the verb classification head with
contrastive action-language alignment (InfoNCE). The entire VQ-VLA tokenizer
is fine-tuned with: recon + 5*vq + lambda * contrastive.

Uses ALL instructions (no verb filtering) since the contrastive objective
handles full sentences, not verb classes.

Usage:
    python -m openvla_experiment.scripts.finetune_tokenizer_clip \
        --tag clip_frozen --clip_lambda 1.0 \
        --text_model laion/CLIP-ViT-B-32-laion2B-s34B-b79K --text_type clip

    python -m openvla_experiment.scripts.finetune_tokenizer_clip \
        --tag gpt2_frozen --clip_lambda 1.0 \
        --text_model gpt2 --text_type gpt2
"""

import csv
import math
import os
import sys
import json
import argparse
import numpy as np
import time as time_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Block tensorflow (numpy 2.x crash in mmml env)
os.environ.setdefault('USE_TF', '0')

# Project root imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DATA_DIR, VAL_DIR, DATA_ROOT, ACTION_KEY, EPISODE_TEMPLATE
from tokenization.vqvae_tokenizer import (
    VQVLA_WINDOW_SIZE, VQVLA_NUM_TOKENS, VQVLA_VOCAB_SIZE,
    VQVLA_CONFIG_DIR, VQVLA_CHECKPOINT_PATH,
)
# Reuse text encoder and contrastive components from clip_action_language
from tokenization.clip_action_language import (
    TextEncoderWrapper, LoRALayer, LoRAWrappedLinear,
    ActionTransformer, load_calvin_raw,
)


# ─── Dataset ────────────────────────────────────────────────────────────────

class CalvinCLIPVQVLADataset(Dataset):
    """CALVIN trajectories with raw instructions for contrastive training.

    No verb filtering — returns full instruction strings.
    Preloads all actions into RAM via cached .npz file.
    """

    def __init__(self, df, data_dir, window_size=VQVLA_WINDOW_SIZE,
                 max_windows=16):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.window_size = window_size
        self.max_windows = max_windows

        # Preload via cache
        cache_path = os.path.join(data_dir, '_action_cache.npz')
        all_starts = self.df['start_idx'].values.astype(int)
        all_ends = self.df['end_idx'].values.astype(int)

        if os.path.exists(cache_path):
            print("  Loading action cache from {}...".format(cache_path))
            cache = np.load(cache_path)
            offset = int(cache['offset'])
            all_actions = cache['actions']
        else:
            needed = set()
            for s, e in zip(all_starts, all_ends):
                needed.update(range(s, e + 1))
            needed = sorted(needed)
            offset = needed[0]
            size = needed[-1] - offset + 1
            print("  Building action cache: {} timesteps ({}-{})...".format(
                len(needed), offset, needed[-1]))
            all_actions = np.zeros((size, 7), dtype=np.float32)
            for j in needed:
                path = os.path.join(data_dir, EPISODE_TEMPLATE.format(j))
                data = np.load(path, mmap_mode='r')
                all_actions[j - offset] = data[ACTION_KEY]
            np.savez_compressed(cache_path, actions=all_actions,
                                offset=np.array(offset))
            print("  Saved cache to {}".format(cache_path))

        self.trajectories = []
        for i in range(len(self.df)):
            s = all_starts[i] - offset
            e = all_ends[i] - offset + 1
            self.trajectories.append(all_actions[s:e].copy())
        del all_actions
        print("  Done. Loaded {} trajectories.".format(len(self.trajectories)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        actions = self.trajectories[idx]
        instruction = self.df.iloc[idx]['instruction']

        T = actions.shape[0]
        ws = self.window_size
        n_windows = T // ws
        if n_windows == 0:
            padded = np.pad(actions, ((0, ws - T), (0, 0)), mode='edge')
            windows = padded[np.newaxis]
            n_windows = 1
        else:
            windows = actions[:n_windows * ws].reshape(n_windows, ws, 7)

        if n_windows > self.max_windows:
            windows = windows[:self.max_windows]
            n_windows = self.max_windows

        padded_windows = np.zeros(
            (self.max_windows, ws, 7), dtype=np.float32)
        padded_windows[:n_windows] = windows

        return (torch.from_numpy(padded_windows),
                instruction,
                torch.tensor(n_windows, dtype=torch.long))


# ─── Contrastive head ──────────────────────────────────────────────────────

class ContrastiveHead(nn.Module):
    """Action transformer + projection for contrastive alignment.

    Takes per-window quantized latents (128-d from VQ-VLA), pools via
    transformer, projects to shared CLIP space.
    """

    def __init__(self, latent_dim=128, d_model=128, nhead=4,
                 transformer_layers=2, proj_dim=128, dropout=0.1,
                 max_windows=16):
        super().__init__()
        self.action_transformer = ActionTransformer(
            input_dim=latent_dim, d_model=d_model, nhead=nhead,
            num_layers=transformer_layers, dropout=dropout,
            max_len=max_windows)
        self.action_proj = nn.Linear(d_model, proj_dim)
        # Learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=20.0)

    def forward(self, window_latents, n_windows):
        """
        Args:
            window_latents: (B, max_windows, 128) quantized latents per window
            n_windows: (B,) real window counts
        Returns:
            action_emb: (B, proj_dim) L2-normalized
        """
        B, max_w, D = window_latents.shape
        device = window_latents.device
        pad_mask = torch.arange(max_w, device=device).unsqueeze(0) >= \
            n_windows.unsqueeze(1)  # True = pad
        cls_out = self.action_transformer(window_latents, mask=pad_mask)
        action_emb = self.action_proj(cls_out)
        return F.normalize(action_emb, dim=-1)


# ─── Forward / Loss ────────────────────────────────────────────────────────

def forward_vqvla(vqvae, windows_batch, n_windows_batch, device):
    """Forward pass through VQ-VLA encoder/quantizer/decoder."""
    B = windows_batch.size(0)

    all_windows = []
    window_counts = []
    for i in range(B):
        nw = n_windows_batch[i].item()
        all_windows.append(windows_batch[i, :nw])
        window_counts.append(nw)
    all_windows_cat = torch.cat(all_windows, dim=0).to(device)

    latents = vqvae.encode(all_windows_cat).latents
    state_rep = latents.view(latents.size(0), -1, latents.size(1))
    quantized, vq_codes, vq_losses = vqvae.vq_layer(state_rep)
    quantized_flat = quantized.view(latents.size(0), -1)

    decoded = vqvae.decode(quantized_flat)
    recon_loss = F.mse_loss(decoded, all_windows_cat.to(decoded.dtype))
    vq_loss = vq_losses.sum()

    # Reshape quantized latents back to per-trajectory
    # Pad to (B, max_windows, 128)
    max_w = max(window_counts)
    traj_latents = torch.zeros(B, max_w, quantized_flat.size(-1),
                               device=device, dtype=quantized_flat.dtype)
    offset = 0
    for i, nw in enumerate(window_counts):
        traj_latents[i, :nw] = quantized_flat[offset:offset + nw]
        offset += nw

    return {
        'recon_loss': recon_loss,
        'vq_loss': vq_loss,
        'traj_latents': traj_latents,  # (B, max_w, 128)
        'vq_codes': vq_codes,
    }


def contrastive_loss(action_emb, text_emb, text_list, temperature):
    """Supervised contrastive loss with false-negative masking."""
    B = len(action_emb)
    device = action_emb.device

    logits = (action_emb @ text_emb.T) * temperature

    # Positive mask: same instruction = positive pair
    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if text_list[i] == text_list[j]:
                pos_mask[i, j] = True

    # Action → text
    logits_stable = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits_stable - logits_stable.exp().sum(dim=1, keepdim=True).log()
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_a2t = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
    loss_a2t = loss_a2t.mean()

    # Text → action
    logits_t = (text_emb @ action_emb.T) * temperature
    logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values.detach()
    log_prob_t = logits_t - logits_t.exp().sum(dim=1, keepdim=True).log()
    n_pos_t = pos_mask.T.float().sum(dim=1).clamp(min=1)
    loss_t2a = -(log_prob_t * pos_mask.T.float()).sum(dim=1) / n_pos_t
    loss_t2a = loss_t2a.mean()

    return (loss_a2t + loss_t2a) / 2


def train_epoch(vqvae, clip_head, text_encoder, text_proj, loader,
                optimizer, clip_lambda, device, max_grad_norm=1.0):
    vqvae.train()
    clip_head.train()

    total_recon = total_vq = total_clip = 0.0
    n_batches = 0

    for windows, instructions, n_windows in loader:
        result = forward_vqvla(vqvae, windows, n_windows, device)
        loss = result['recon_loss'] + 5 * result['vq_loss']

        if clip_lambda > 0:
            action_emb = clip_head(result['traj_latents'], n_windows.to(device))
            with torch.set_grad_enabled(text_encoder.lora_r > 0):
                text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)

            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions),
                clip_head.temperature)
            loss = loss + clip_lambda * clip_loss
            total_clip += clip_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            trainable = [p for p in vqvae.parameters() if p.requires_grad]
            trainable += list(clip_head.parameters())
            trainable += [p for p in text_proj.parameters()]
            if text_encoder.lora_r > 0:
                trainable += [p for p in text_encoder.parameters()
                              if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()

        total_recon += result['recon_loss'].item()
        total_vq += result['vq_loss'].item()
        n_batches += 1

    return {
        'recon': total_recon / n_batches,
        'vq': total_vq / n_batches,
        'clip': total_clip / n_batches if clip_lambda > 0 else 0,
        'temp': clip_head.temperature.item(),
    }


@torch.no_grad()
def eval_epoch(vqvae, clip_head, text_encoder, text_proj, loader,
               clip_lambda, device):
    vqvae.eval()
    clip_head.eval()

    total_recon = total_vq = total_clip = 0.0
    all_codes = []
    n_batches = 0

    for windows, instructions, n_windows in loader:
        result = forward_vqvla(vqvae, windows, n_windows, device)

        if clip_lambda > 0:
            action_emb = clip_head(result['traj_latents'], n_windows.to(device))
            text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)

            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions),
                clip_head.temperature)
            total_clip += clip_loss.item()

        total_recon += result['recon_loss'].item()
        total_vq += result['vq_loss'].item()
        all_codes.append(result['vq_codes'].cpu())
        n_batches += 1

    # Codebook utilization
    all_codes = torch.cat(all_codes, dim=0).squeeze(1)
    used_codes = set()
    for q in range(min(VQVLA_NUM_TOKENS, all_codes.size(-1))):
        used_codes.update(all_codes[:, q].unique().tolist())

    return {
        'recon': total_recon / n_batches,
        'vq': total_vq / n_batches,
        'clip': total_clip / n_batches if clip_lambda > 0 else 0,
        'temp': clip_head.temperature.item(),
        'codebook_used': len(used_codes),
        'codebook_total': VQVLA_VOCAB_SIZE,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VQ-VLA tokenizer with CLIP contrastive loss")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--config_dir", type=str, default=VQVLA_CONFIG_DIR)
    parser.add_argument("--pretrained_path", type=str,
                        default=VQVLA_CHECKPOINT_PATH)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_windows", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--vqvla_lora_r", type=int, default=0,
                        help="LoRA rank for VQ-VLA conv layers (0=full finetune)")
    # CLIP contrastive
    parser.add_argument("--clip_lambda", type=float, default=1.0)
    parser.add_argument("--text_model", type=str,
                        default='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    parser.add_argument("--text_type", type=str, default='clip',
                        choices=['clip', 'gpt2'])
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--transformer_layers", type=int, default=2)
    # Output
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # ── Load data (no verb filtering) ────────────────────────────────────
    print("Loading CALVIN data (no verb filtering)...")
    train_df = load_calvin_raw(args.data_dir)
    val_df = load_calvin_raw(args.val_dir)
    print("Train: {} episodes, {} unique instructions".format(
        len(train_df), train_df['instruction'].nunique()))
    print("Val: {} episodes, {} unique instructions".format(
        len(val_df), val_df['instruction'].nunique()))

    train_ds = CalvinCLIPVQVLADataset(train_df, args.data_dir,
                                       max_windows=args.max_windows)
    val_ds = CalvinCLIPVQVLADataset(val_df, args.val_dir,
                                     max_windows=args.max_windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Load pretrained VQ-VLA ──────────────────────────────────────────
    print("Loading pretrained VQ-VLA from {}...".format(args.pretrained_path))
    from tokenization.vqvla import ActionVQVAELossWrapper

    wrapper = ActionVQVAELossWrapper(
        model_path=args.config_dir,
        checkpoint_path=args.pretrained_path,
        is_eval=False,
        freeze=False,
        use_action_type_pe=True,
        use_time_pe=True,
    )
    vqvae = wrapper.vqvae.to(device)
    vqvae_total = sum(p.numel() for p in vqvae.parameters())

    if args.vqvla_lora_r > 0:
        # Apply LoRA to VQ-VLA conv layers: freeze base weights, add
        # low-rank adapters. Uses peft's LoraConfig targeting Conv2d.
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=args.vqvla_lora_r,
            lora_alpha=args.vqvla_lora_r,
            target_modules=["conv"],  # targets all CausalConv2d inner Conv2d
            lora_dropout=0.05,
        )
        vqvae = get_peft_model(vqvae, lora_config)
        vqvae_trainable = sum(p.numel() for p in vqvae.parameters()
                              if p.requires_grad)
        print("VQ-VLA loaded ({:.1f}M total, LoRA r={}: {:.1f}K trainable)".format(
            vqvae_total / 1e6, args.vqvla_lora_r, vqvae_trainable / 1e3))
    else:
        vqvae_trainable = vqvae_total
        print("VQ-VLA loaded ({:.1f}M params, all trainable)".format(
            vqvae_total / 1e6))

    # ── Contrastive head ────────────────────────────────────────────────
    latent_dim = 128  # Fixed by VQ-VLA architecture
    clip_head = ContrastiveHead(
        latent_dim=latent_dim, d_model=args.d_model,
        nhead=4, transformer_layers=args.transformer_layers,
        proj_dim=args.proj_dim, max_windows=args.max_windows,
    ).to(device)

    # ── Text encoder ────────────────────────────────────────────────────
    text_encoder = TextEncoderWrapper(
        model_name=args.text_model,
        model_type=args.text_type,
        freeze=(args.lora_r == 0),
        lora_r=args.lora_r,
    ).to(device)
    text_proj = nn.Linear(text_encoder.output_dim, args.proj_dim).to(device)

    # Print param counts
    clip_params = sum(p.numel() for p in clip_head.parameters())
    text_proj_params = sum(p.numel() for p in text_proj.parameters())
    lora_params = sum(p.numel() for p in text_encoder.parameters()
                      if p.requires_grad)
    trainable_total = vqvae_trainable + clip_params + text_proj_params + lora_params
    print("Contrastive head: {:.2f}M params".format(clip_params / 1e6))
    print("Text proj: {:.1f}K params".format(text_proj_params / 1e3))
    if args.lora_r > 0:
        print("LoRA params: {:.1f}K".format(lora_params / 1e3))
    print("Total trainable: {:.2f}M".format(trainable_total / 1e6))

    # ── Optimizer ───────────────────────────────────────────────────────
    params = [p for p in vqvae.parameters() if p.requires_grad] + \
        list(clip_head.parameters()) + list(text_proj.parameters())
    if args.lora_r > 0:
        params += [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── Training loop ───────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, "vqvla_clip_{}".format(args.tag))
    os.makedirs(save_dir, exist_ok=True)
    best_val_total = float('inf')
    best_epoch = 0
    patience_counter = 0

    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_recon", "train_vq", "train_clip",
                         "val_recon", "val_vq", "val_clip", "temp",
                         "codebook_used", "lr", "time"])

    print("\nTraining for {} epochs (clip_lambda={}, text={})".format(
        args.epochs, args.clip_lambda,
        "{} {}".format(args.text_type,
                       "LoRA r={}".format(args.lora_r)
                       if args.lora_r > 0 else "frozen")))
    print("Save dir: {}".format(save_dir))
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time_mod.time()
        train_m = train_epoch(vqvae, clip_head, text_encoder, text_proj,
                              train_loader, optimizer, args.clip_lambda,
                              device, args.max_grad_norm)
        val_m = eval_epoch(vqvae, clip_head, text_encoder, text_proj,
                           val_loader, args.clip_lambda, device)
        scheduler.step()
        dt = time_mod.time() - t0

        val_total = val_m['recon'] + 5 * val_m['vq'] + \
            args.clip_lambda * val_m['clip']

        print("Epoch {:3d}/{} ({:.1f}s) | "
              "Train: recon={:.5f} vq={:.5f} clip={:.4f} | "
              "Val: recon={:.5f} vq={:.5f} clip={:.4f} | "
              "temp={:.2f} codes={}/{}".format(
                  epoch, args.epochs, dt,
                  train_m['recon'], train_m['vq'], train_m['clip'],
                  val_m['recon'], val_m['vq'], val_m['clip'],
                  val_m['temp'], val_m['codebook_used'],
                  val_m['codebook_total']))

        # Save best
        if val_total < best_val_total:
            best_val_total = val_total
            best_epoch = epoch
            patience_counter = 0

            # Save merged VQ-VLA weights (compatible with downstream loading)
            if args.vqvla_lora_r > 0:
                # Merge LoRA adapters into base weights, save clean state dict
                import copy
                vqvae_copy = copy.deepcopy(vqvae)
                merged = vqvae_copy.merge_and_unload()
                torch.save(merged.state_dict(),
                           os.path.join(save_dir, "vqvla_weights.pth"))
                del vqvae_copy, merged
            else:
                torch.save(vqvae.state_dict(),
                           os.path.join(save_dir, "vqvla_weights.pth"))
            full_ckpt = {
                'epoch': epoch,
                'vqvae_state_dict': vqvae.state_dict(),
                'clip_head_state_dict': clip_head.state_dict(),
                'text_proj_state_dict': text_proj.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_m,
                'val_metrics': val_m,
                'args': vars(args),
            }
            if args.lora_r > 0:
                lora_state = {k: v for k, v in text_encoder.state_dict().items()
                              if 'lora' in k}
                full_ckpt['lora_state_dict'] = lora_state
            torch.save(full_ckpt, os.path.join(save_dir, "full.pth"))
            print("  -> Saved best (val_total={:.4f})".format(val_total))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping at epoch {} "
                      "(no improvement for {} epochs, best=epoch {})".format(
                          epoch, args.patience, best_epoch))
                break

        # CSV
        csv_writer.writerow([
            epoch,
            "{:.6f}".format(train_m['recon']),
            "{:.6f}".format(train_m['vq']),
            "{:.6f}".format(train_m['clip']),
            "{:.6f}".format(val_m['recon']),
            "{:.6f}".format(val_m['vq']),
            "{:.6f}".format(val_m['clip']),
            "{:.3f}".format(val_m['temp']),
            val_m['codebook_used'],
            "{:.8f}".format(optimizer.param_groups[0]['lr']),
            "{:.1f}".format(dt),
        ])
        csv_file.flush()

    csv_file.close()

    config = {
        'tag': args.tag,
        'clip_lambda': args.clip_lambda,
        'text_model': args.text_model,
        'text_type': args.text_type,
        'lora_r': args.lora_r,
        'proj_dim': args.proj_dim,
        'epochs_run': epoch,
        'best_epoch': best_epoch,
        'best_val_total': float(best_val_total),
        'pretrained_path': args.pretrained_path,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("\nDone. Best val total: {:.4f} at epoch {}".format(
        best_val_total, best_epoch))


if __name__ == "__main__":
    main()
