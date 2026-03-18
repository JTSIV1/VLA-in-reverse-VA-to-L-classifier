"""
Zero-shot retrieval evaluation for CLIP-trained action tokenizers.

Two evaluation modes:
  1. Verb retrieval: Given an action trajectory, retrieve the correct verb
     from a set of 21+ verb templates.
  2. Instruction retrieval: Given an action trajectory, retrieve the correct
     instruction from all unique instructions in the val set.

Metrics: R@1, R@5, R@10, median rank, per-verb accuracy.

Usage:
  python -m openvla_experiment.scripts.eval_clip_retrieval --tag clip_full
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ['USE_TF'] = '0'

from config import DATA_ROOT, TRAIN_DIR, VAL_DIR
from tokenization.clip_action_language import (
    ActionTransformer, TextEncoderWrapper, load_calvin_raw)
from tokenization.vqvla import ActionVQVAELossWrapper
from tokenization.vqvae_tokenizer import (
    VQVLA_CONFIG_DIR, VQVLA_CHECKPOINT_PATH, VQVLA_WINDOW_SIZE)
from utils import extract_verb

# Reuse dataset from finetuning script
from openvla_experiment.scripts.finetune_tokenizer_clip import (
    CalvinCLIPVQVLADataset, ContrastiveHead)


def load_clip_model(ckpt_dir, device):
    """Load a trained CLIP tokenizer checkpoint."""
    full_ckpt_path = os.path.join(ckpt_dir, 'full.pth')
    ckpt = torch.load(full_ckpt_path, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ckpt['args'])

    # Load VQ-VLA: first load architecture with pretrained weights,
    # then override with our finetuned weights
    wrapper = ActionVQVAELossWrapper(
        model_path=args.config_dir,
        checkpoint_path=args.pretrained_path,
        is_eval=True, freeze=True,
        use_action_type_pe=True, use_time_pe=True)
    vqvae = wrapper.vqvae
    # Load finetuned weights
    vqvla_weights_path = os.path.join(ckpt_dir, 'vqvla_weights.pth')
    ft_state = torch.load(vqvla_weights_path, map_location='cpu', weights_only=False)
    vqvae.load_state_dict(ft_state, strict=True)
    vqvae = vqvae.to(device).eval()

    # Load contrastive head
    clip_head = ContrastiveHead(
        latent_dim=128, d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        proj_dim=args.proj_dim, max_windows=args.max_windows).to(device)
    clip_head.load_state_dict(ckpt['clip_head_state_dict'])
    clip_head.eval()

    # Load text encoder
    text_encoder = TextEncoderWrapper(
        model_name=args.text_model, model_type=args.text_type,
        freeze=True, lora_r=0).to(device)
    # If LoRA was used, load LoRA weights
    if args.lora_r > 0 and 'lora_state_dict' in ckpt:
        text_encoder._apply_lora(args.lora_r)
        lora_sd = ckpt['lora_state_dict']
        text_encoder.load_state_dict(lora_sd, strict=False)
    text_encoder.eval()

    # Load text projection
    text_proj = torch.nn.Linear(text_encoder.output_dim, args.proj_dim).to(device)
    text_proj.load_state_dict(ckpt['text_proj_state_dict'])
    text_proj.eval()

    return vqvae, clip_head, text_encoder, text_proj, args


@torch.no_grad()
def encode_all_actions(vqvae, clip_head, loader, device):
    """Encode all trajectories to action embeddings."""
    all_embs = []
    all_instructions = []

    for windows, instructions, n_windows in tqdm(loader, desc="Encoding actions"):
        windows = windows.to(device)
        n_windows = n_windows.to(device)
        B, max_w, T, D = windows.shape

        # Use the same forward_vqvla logic as training
        all_windows = []
        window_counts = []
        for i in range(B):
            nw = n_windows[i].item()
            all_windows.append(windows[i, :nw])
            window_counts.append(nw)
        all_windows_cat = torch.cat(all_windows, dim=0).to(device)

        # VQ-VLA encode (handles PE internally)
        latents = vqvae.encode(all_windows_cat).latents
        state_rep = latents.view(latents.size(0), -1, latents.size(1))
        quantized, _, _ = vqvae.vq_layer(state_rep)
        quantized_flat = quantized.view(latents.size(0), -1)

        # Reshape back to per-trajectory (B, max_w, 128)
        max_w_actual = max(window_counts)
        traj_latents = torch.zeros(B, max_w_actual, quantized_flat.size(-1),
                                   device=device, dtype=quantized_flat.dtype)
        offset = 0
        for i, nw in enumerate(window_counts):
            traj_latents[i, :nw] = quantized_flat[offset:offset + nw]
            offset += nw

        # Pad to loader's max_w if needed
        if max_w_actual < max_w:
            pad = torch.zeros(B, max_w - max_w_actual, traj_latents.size(-1),
                              device=device, dtype=traj_latents.dtype)
            traj_latents = torch.cat([traj_latents, pad], dim=1)

        action_emb = clip_head(traj_latents, n_windows)
        all_embs.append(action_emb.cpu())
        all_instructions.extend(list(instructions))

    return torch.cat(all_embs, dim=0), all_instructions


@torch.no_grad()
def encode_texts(text_encoder, text_proj, texts, device, batch_size=64):
    """Encode a list of text strings to embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        features = text_encoder(batch)
        emb = text_proj(features)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def retrieval_metrics(action_embs, text_embs, labels_action, labels_text):
    """Compute retrieval metrics.

    Args:
        action_embs: (N, D) action embeddings for N val episodes
        text_embs: (M, D) text embeddings for M candidate texts
        labels_action: list of N labels (one per val episode)
        labels_text: list of M labels (one per candidate text)
    Returns:
        dict with R@1, R@5, R@10, median_rank, mean_rank
    """
    sims = action_embs @ text_embs.T

    ranks = []
    for i in range(len(action_embs)):
        target_label = labels_action[i]
        correct_indices = [j for j, l in enumerate(labels_text) if l == target_label]
        if not correct_indices:
            continue

        sorted_indices = sims[i].argsort(descending=True).tolist()
        for rank, idx in enumerate(sorted_indices):
            if idx in correct_indices:
                ranks.append(rank + 1)
                break

    ranks = np.array(ranks)

    return {
        'R@1': float((ranks <= 1).mean() * 100),
        'R@5': float((ranks <= 5).mean() * 100),
        'R@10': float((ranks <= 10).mean() * 100),
        'median_rank': float(np.median(ranks)),
        'mean_rank': float(np.mean(ranks)),
        'N': int(len(ranks)),
    }


def per_verb_accuracy(action_embs, text_embs, action_verbs, text_verbs):
    """Compute per-verb R@1 accuracy."""
    sims = action_embs @ text_embs.T

    verb_correct = {}
    verb_total = {}

    for i in range(len(action_embs)):
        verb = action_verbs[i]
        top_idx = sims[i].argmax().item()
        predicted_verb = text_verbs[top_idx]

        verb_total[verb] = verb_total.get(verb, 0) + 1
        if predicted_verb == verb:
            verb_correct[verb] = verb_correct.get(verb, 0) + 1

    results = {}
    for verb in sorted(verb_total.keys()):
        correct = verb_correct.get(verb, 0)
        total = verb_total[verb]
        results[verb] = {'correct': correct, 'total': total,
                         'acc': correct / total * 100}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, required=True,
                        help='Tokenizer tag (e.g. clip_full)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Override checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, default='validation',
                        choices=['training', 'validation'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = args.ckpt_dir or os.path.join(
        PROJECT_ROOT, 'checkpoints', 'vqvla_clip_{}'.format(args.tag))
    print("Loading model from: {}".format(ckpt_dir))

    vqvae, clip_head, text_encoder, text_proj, train_args = \
        load_clip_model(ckpt_dir, device)

    data_dir = VAL_DIR if args.split == 'validation' else TRAIN_DIR
    df = load_calvin_raw(data_dir)
    print("{} split: {} episodes, {} unique instructions".format(
        args.split, len(df), df['instruction'].nunique()))

    dataset = CalvinCLIPVQVLADataset(df, data_dir, max_windows=train_args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    # Extract primary verb for each episode
    episode_verbs = []
    for inst in df['instruction']:
        verbs = extract_verb(inst)
        episode_verbs.append(verbs[0] if verbs else 'unknown')

    # Encode all actions
    print("\nEncoding action trajectories...")
    action_embs, action_instructions = encode_all_actions(
        vqvae, clip_head, loader, device)

    unique_instructions = sorted(df['instruction'].unique())
    unique_verbs = sorted(set(episode_verbs))
    print("Unique instructions: {}, unique verbs: {}".format(
        len(unique_instructions), len(unique_verbs)))

    # Encode text candidates
    print("Encoding text candidates...")
    instr_embs = encode_texts(text_encoder, text_proj, unique_instructions, device)
    verb_embs = encode_texts(text_encoder, text_proj, unique_verbs, device)

    # Instruction retrieval
    instr_labels = list(df['instruction'])
    instr_metrics = retrieval_metrics(
        action_embs, instr_embs, instr_labels, unique_instructions)

    # Verb retrieval
    verb_metrics = retrieval_metrics(
        action_embs, verb_embs, episode_verbs, unique_verbs)

    # Per-verb breakdown
    per_verb = per_verb_accuracy(action_embs, verb_embs, episode_verbs, unique_verbs)

    # Print results
    print("\n" + "=" * 60)
    print("Results for: {} ({})".format(args.tag, ckpt_dir))
    print("=" * 60)

    print("\n--- Instruction Retrieval ({} candidates) ---".format(
        len(unique_instructions)))
    for k, v in instr_metrics.items():
        print("  {}: {:.2f}".format(k, v) if isinstance(v, float)
              else "  {}: {}".format(k, v))

    print("\n--- Verb Retrieval ({} candidates) ---".format(len(unique_verbs)))
    for k, v in verb_metrics.items():
        print("  {}: {:.2f}".format(k, v) if isinstance(v, float)
              else "  {}: {}".format(k, v))

    print("\n--- Per-Verb R@1 ---")
    for verb, stats in sorted(per_verb.items(), key=lambda x: -x[1]['acc']):
        print("  {:<15s}: {:.1f}% ({}/{})".format(
            verb, stats['acc'], stats['correct'], stats['total']))

    # Save results
    results = {
        'tag': args.tag,
        'ckpt_dir': ckpt_dir,
        'split': args.split,
        'instruction_retrieval': instr_metrics,
        'verb_retrieval': verb_metrics,
        'per_verb': per_verb,
    }
    out_path = os.path.join(ckpt_dir, 'retrieval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to {}".format(out_path))


if __name__ == '__main__':
    main()
