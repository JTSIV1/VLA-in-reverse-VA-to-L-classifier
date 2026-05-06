"""Shared utilities for tokenizer training and evaluation.

Includes:
  - extract_episode_batch: encode a CalvinTokenizerDataset batch through a
    frozen tokenizer, returning latents, codes, and losses.
  - Checkpoint saving, resume, and logging helpers for train_tokenizer.py.
"""

import csv
import json
import os
import numpy as np
import torch


def resume_checkpoint(args, model, optimizer, verb_head, clip_head, text_proj, device):
    """Load checkpoint and restore training state.

    Returns (start_epoch, best_metric, best_verb_acc).
    """
    start_epoch = 0
    best_metric = float('inf')
    best_verb_acc = 0.0

    if not (args.resume and os.path.isfile(args.resume)):
        return start_epoch, best_metric, best_verb_acc

    print(f"Loading checkpoint from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict',
                                   ckpt.get('vqvae_state_dict', {})))
    if args.freeze_tokenizer:
        print("Probe mode: tokenizer loaded & frozen, aux heads initialized fresh")
    else:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if verb_head is not None and 'verb_head_state_dict' in ckpt:
            verb_head.load_state_dict(ckpt['verb_head_state_dict'])
        if clip_head is not None and 'clip_head_state_dict' in ckpt:
            clip_head.load_state_dict(ckpt['clip_head_state_dict'])
        if text_proj is not None and 'text_proj_state_dict' in ckpt:
            text_proj.load_state_dict(ckpt['text_proj_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_metric = ckpt.get('best_metric', float('inf'))
        best_verb_acc = ckpt.get('best_verb_acc', 0.0)
        print(f"Resumed at epoch {start_epoch}")

    return start_epoch, best_metric, best_verb_acc


def setup_output_dir(args):
    """Create output directory and return (save_dir, run_name).

    Naming convention:
        <tokenizer>_<tag>[_<text_type>_<aux_head><aux_lambda>][_pfsq]
    where <text_type> appears only for aux_head=clip (e.g. `_vlm_clip0.1`,
    plain `_clip0.1` for legacy generic CLIP) so VLM-aligned and generic-CLIP
    tokenizers get distinct directories.
    """
    run_name = args.tokenizer
    if args.tag:
        run_name += f"_{args.tag}"
    if args.aux_head != 'none':
        prefix = ""
        if args.aux_head == 'clip' and getattr(args, 'text_type', 'clip') == 'vlm':
            prefix = "vlm_"
        aux_suffix = f"_{prefix}{args.aux_head}{args.aux_lambda}"
        if getattr(args, 'aux_target', 'latent') == 'post_fsq':
            aux_suffix += "_pfsq"
        run_name += aux_suffix
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, run_name


CSV_HEADER = [
    "epoch", "train_recon", "train_vq",
    "train_verb", "train_verb_acc", "train_verb_macro_f1",
    "train_clip",
    "val_recon", "val_vq",
    "val_verb", "val_verb_acc", "val_verb_macro_f1",
    "val_clip", "val_r1", "val_r5", "val_r10",
    "val_codebook_util",
    "lr", "time",
]


def open_csv_logger(save_dir, resume=False):
    """Open metrics CSV file and return (csv_writer, csv_file)."""
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "a" if resume else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not resume:
        csv_writer.writerow(CSV_HEADER)
    return csv_writer, csv_file


def log_epoch(epoch, n_epochs, dt, train_m, val_m, aux_head, retrieval=None):
    """Print one-line epoch summary to stdout."""
    line = f"Epoch {epoch+1:3d}/{n_epochs} ({dt:.1f}s)"
    line += f" | train: recon={train_m['recon']:.5f} vq={train_m['vq']:.5f}"
    if aux_head == 'verb':
        line += (f" verb={train_m['verb']:.4f}"
                 f" acc={train_m['verb_acc']:.1f}%"
                 f" mF1={train_m['verb_macro_f1']:.1f}%")
    if aux_head == 'clip':
        line += f" clip={train_m['clip']:.4f}"
    line += f" | val: recon={val_m['recon']:.5f} vq={val_m['vq']:.5f}"
    if aux_head == 'verb':
        line += (f" verb={val_m['verb']:.4f}"
                 f" acc={val_m['verb_acc']:.1f}%"
                 f" mF1={val_m['verb_macro_f1']:.1f}%")
    if aux_head == 'clip':
        line += f" clip={val_m['clip']:.4f}"
        if retrieval:
            line += (f" R@1={retrieval.get('r@1', 0):.1f}%"
                     f" R@5={retrieval.get('r@5', 0):.1f}%")
    if val_m.get('codebook_util') is not None:
        line += f" | codes={val_m['codebook_util']}"
    print(line)


def write_csv_row(csv_writer, csv_file, epoch, train_m, val_m, retrieval, lr, dt):
    """Append one row to the metrics CSV."""
    cb_util = val_m.get('codebook_util')
    csv_writer.writerow([
        epoch + 1,
        f"{train_m['recon']:.6f}", f"{train_m['vq']:.6f}",
        f"{train_m['verb']:.6f}", f"{train_m['verb_acc']:.2f}",
        f"{train_m['verb_macro_f1']:.2f}",
        f"{train_m['clip']:.6f}",
        f"{val_m['recon']:.6f}", f"{val_m['vq']:.6f}",
        f"{val_m['verb']:.6f}", f"{val_m['verb_acc']:.2f}",
        f"{val_m['verb_macro_f1']:.2f}",
        f"{val_m['clip']:.6f}",
        f"{retrieval.get('r@1', 0.0):.2f}",
        f"{retrieval.get('r@5', 0.0):.2f}",
        f"{retrieval.get('r@10', 0.0):.2f}",
        cb_util if cb_util is not None else "",
        f"{lr:.8f}", f"{dt:.1f}",
    ])
    csv_file.flush()


def save_best_checkpoint(save_dir, epoch, model, optimizer, train_m, val_m,
                         args, best_metric, best_verb_acc,
                         verb_head=None, train_ds=None,
                         clip_head=None, text_proj=None):
    """Save model weights and full checkpoint."""
    ckpt = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_m,
        'val_metrics': val_m,
        'args': vars(args),
        'best_metric': best_metric,
        'best_verb_acc': best_verb_acc,
    }
    if args.aux_head == 'verb' and verb_head is not None:
        ckpt['verb_head_state_dict'] = verb_head.state_dict()
        ckpt['verb_to_id'] = train_ds.verb_to_id
        ckpt['id_to_verb'] = train_ds.id_to_verb
    if args.aux_head == 'clip' and clip_head is not None:
        ckpt['clip_head_state_dict'] = clip_head.state_dict()
        if text_proj is not None:
            ckpt['text_proj_state_dict'] = text_proj.state_dict()

    torch.save(model.state_dict(),
               os.path.join(save_dir, "tokenizer_weights.pth"))
    torch.save(ckpt, os.path.join(save_dir, "full.pth"))
    print(f"  -> Saved best checkpoint (epoch {epoch+1})")


def save_final_config(save_dir, args, run_name, last_epoch, best_metric, best_verb_acc):
    """Save run config JSON at end of training."""
    config = {
        'tokenizer': args.tokenizer,
        'run_name': run_name,
        'aux_head': args.aux_head,
        'aux_lambda': args.aux_lambda,
        'epochs_run': last_epoch + 1,
        'best_metric': float(best_metric),
        'best_verb_acc': float(best_verb_acc),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone. Checkpoints saved to {save_dir}")


# ======================================================================
# Batch encoding for CalvinTokenizerDataset
# ======================================================================

def extract_episode_batch(model, ep_batch, device, tok_type):
    """Encode K chunks per episode through a tokenizer model.

    Takes a collated batch from CalvinTokenizerDataset and runs each valid
    chunk through the tokenizer's forward().  Handles masking padded chunks
    and reshaping outputs back to episode format.

    Each tokenizer's forward() returns a dict with:
        recon_loss, vq_loss, latents, codes

    Latent / code shapes per tokenizer:
    - VQ-BeT: (N, latent_dim) / (N, groups) per chunk
              → (B, K, latent_dim) / (B, K, groups)
    - OAT/QueST: (N, T', 256) / (N, T') per chunk
              → (B, K*T', 256) / (B, K*T')
              with positions expanded and n_valid scaled by T'.

    Args:
        model: frozen tokenizer (VqBetTokenizer, OATTok, or QueSTTok).
        ep_batch: dict from DataLoader with keys:
            'chunks'     (B, K, chunk_size, action_dim)
            'positions'  (B, K)
            'n_valid'    (B,)
            'verb_label' (B,)
            'instruction' (list of str)
        device: torch device.
        tok_type: 'vq_bet', 'oat', or 'quest'.

    Returns:
        dict with keys: recon_loss, vq_loss, latents, codes, positions,
        n_valid, verb_ids, instructions.  All tensors on ``device``.
    """
    chunks = ep_batch['chunks'].to(device)
    positions = ep_batch['positions'].to(device)
    n_valid = ep_batch['n_valid'].to(device)
    verb_ids = ep_batch['verb_label'].to(device)
    instructions = ep_batch.get('instruction', [''] * chunks.size(0))
    B, K = chunks.shape[0], chunks.shape[1]

    # Action real lens per chunk (if available from dataset)
    raw_real_lens = ep_batch.get('action_real_lens')

    # Mask out padded chunks
    valid_mask = torch.arange(K, device=device).unsqueeze(0) < n_valid.unsqueeze(1)
    mask_flat = valid_mask.view(B * K)

    # Flatten (B, K, ...) -> (B*K, ...) and select valid chunks
    if tok_type == 'vq_bet':
        valid_chunks = chunks.reshape(B * K, -1)[mask_flat]
        result = model(valid_chunks)
    else:
        flat = chunks.view(B * K, chunks.shape[2], chunks.shape[3])
        valid_chunks = flat[mask_flat]
        batch_dict = {"action": valid_chunks}
        if raw_real_lens is not None:
            real_lens_flat = raw_real_lens.to(device).view(B * K)[mask_flat]
            batch_dict["action_real_lens"] = real_lens_flat
        result = model(batch_dict)

    lat = result['latents']
    raw_codes = result.get('codes')
    raw_fsq = result.get('fsq_codes')  # (N, T', fsq_dim) or None

    if lat.ndim == 2:
        # VQ-BeT: (N, latent_dim) -> (B, K, latent_dim)
        latents = torch.zeros(B, K, lat.size(-1), device=device)
        latents.view(B * K, -1)[mask_flat] = lat

        # Codes: (N, groups) -> (B, K, groups)
        if raw_codes is not None:
            codes = torch.zeros(B, K, raw_codes.size(-1),
                                device=device, dtype=raw_codes.dtype)
            codes.view(B * K, -1)[mask_flat] = raw_codes
        else:
            codes = None
        fsq_codes = None  # VQ-BeT has no FSQ codes
    else:
        # OAT/QueST: (N, T', 256) -> (B, K*T', 256)
        T_prime = lat.size(1)
        latent_dim = lat.size(2)

        lat_full = torch.zeros(B * K, T_prime, latent_dim, device=device)
        lat_full[mask_flat] = lat
        latents = lat_full.view(B, K * T_prime, latent_dim)

        # Codes: (N, T') -> (B, K*T')
        if raw_codes is not None:
            codes_full = torch.zeros(B * K, T_prime,
                                     device=device, dtype=raw_codes.dtype)
            codes_full[mask_flat] = raw_codes
            codes = codes_full.view(B, K * T_prime)
        else:
            codes = None

        # FSQ codes: (N, T', fsq_dim) -> (B, K*T', fsq_dim)
        if raw_fsq is not None:
            fsq_dim = raw_fsq.size(-1)
            fsq_full = torch.zeros(B * K, T_prime, fsq_dim, device=device)
            fsq_full[mask_flat] = raw_fsq
            fsq_codes = fsq_full.view(B, K * T_prime, fsq_dim)
        else:
            fsq_codes = None

        # Expand positions with within-chunk offsets for PE
        pos_expanded = positions.unsqueeze(2).expand(B, K, T_prime)
        offsets = torch.arange(T_prime, device=device).float() / (T_prime * K)
        positions = (pos_expanded + offsets.view(1, 1, T_prime)).reshape(B, K * T_prime)

        n_valid = n_valid * T_prime

    return {
        'recon_loss': result['recon_loss'],
        'vq_loss': result['vq_loss'],
        'latents': latents,
        'codes': codes,
        'fsq_codes': fsq_codes,
        'positions': positions,
        'n_valid': n_valid,
        'verb_ids': verb_ids,
        'instructions': instructions,
    }
