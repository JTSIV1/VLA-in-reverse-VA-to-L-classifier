"""VQ-VAE chunk-based action tokenizer for CALVIN action trajectories.

Each trajectory (T, 7) is split into non-overlapping chunks of size K.
Each chunk (K*7-dim) is encoded by a small MLP into a learned codebook index.
Sequence length after tokenization: T // K (last partial chunk discarded).

Comparison to FAST:
  FAST: DCT + BPE, sequence-level compression, ~25 tokens regardless of K
  VQ-VAE: learned, per-chunk, ~T/K tokens (30/15/7 for K=2/4/8 on CALVIN ~61-step trajectories)

Usage (standalone, to fit):
    python -m tokenization.vqvae_tokenizer --save_path ./checkpoints/vqvae_k4_c512 --chunk_size 4 --num_codes 512

Usage (as module):
    from tokenization.vqvae_tokenizer import load_vqvae_tokenizer, tokenize_trajectory_vqvae
    tok = load_vqvae_tokenizer("./checkpoints/vqvae_k4_c512")
    token_ids = tokenize_trajectory_vqvae(tok, actions_np)  # (T, 7) -> (T//K,) np.int64
"""

import os
import sys
import json
import argparse

# Ensure project root is on path for standalone execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from utils import load_calvin_to_dataframe
from config import DATA_DIR, ACTION_KEY, EPISODE_TEMPLATE, VQVAE_TOKENIZER_PATH


class ActionVQVAE(nn.Module):
    """Chunk-based VQ-VAE for action sequences.

    Encoder: Linear(chunk_size*action_dim -> 128) -> ReLU -> Linear(128 -> latent_dim)
    VQ:      nearest-neighbor lookup in codebook, straight-through gradient estimator
    Decoder: Linear(latent_dim -> 128) -> ReLU -> Linear(128 -> chunk_size*action_dim)
    """

    def __init__(self, action_dim=7, chunk_size=4, latent_dim=64,
                 num_codes=512, commitment_cost=0.25):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost

        input_dim = chunk_size * action_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Codebook: (num_codes, latent_dim)
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def _nearest_codes(self, z):
        """Find nearest codebook entry for each row in z.
        Args:
            z: (B, latent_dim)
        Returns:
            indices: (B,) long
            quantized: (B, latent_dim) — the corresponding codebook vectors
        """
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
        z2 = (z ** 2).sum(dim=1, keepdim=True)           # (B, 1)
        e2 = (self.codebook.weight ** 2).sum(dim=1)       # (num_codes,)
        ze = z @ self.codebook.weight.T                    # (B, num_codes)
        distances = z2 + e2 - 2 * ze                      # (B, num_codes)
        indices = distances.argmin(dim=1)                  # (B,)
        quantized = self.codebook(indices)                 # (B, latent_dim)
        return indices, quantized

    def encode(self, x):
        """Encode chunks to codebook indices.
        Args:
            x: (B, chunk_size * action_dim) float tensor
        Returns:
            indices: (B,) long tensor
        """
        z = self.encoder(x)
        indices, _ = self._nearest_codes(z)
        return indices

    def forward(self, x):
        """Forward pass for training.
        Args:
            x: (B, chunk_size * action_dim)
        Returns:
            recon: (B, chunk_size * action_dim) reconstructed chunk
            recon_loss: scalar
            vq_loss: scalar (codebook + commitment)
        """
        z = self.encoder(x)                               # (B, latent_dim)
        indices, quantized = self._nearest_codes(z)

        # VQ losses
        codebook_loss = F.mse_loss(z.detach(), quantized)
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: gradient bypasses VQ
        z_q = z + (quantized - z).detach()

        recon = self.decoder(z_q)                         # (B, chunk_size * action_dim)
        recon_loss = F.mse_loss(recon, x)

        return recon, recon_loss, vq_loss


class VerbDecodableVQVAE(nn.Module):
    """VQ-VAE with joint verb classification head.

    Wraps ActionVQVAE and adds a transformer-based verb classifier that
    mirrors the downstream ActionToVerbTransformer architecture:
    [CLS] + codebook embeddings + positional encoding → transformer → classify CLS.

    Training loss: recon_loss + vq_loss + verb_loss_weight * verb_CE
    Saves inner ActionVQVAE weights as vqvae.pth for downstream compatibility.
    """

    def __init__(self, num_verbs, action_dim=7, chunk_size=4,
                 latent_dim=64, num_codes=512, commitment_cost=0.25,
                 cls_d_model=128, cls_nhead=4, cls_layers=2,
                 max_chunks=32, cls_dropout=0.1):
        super().__init__()
        self.vqvae = ActionVQVAE(
            action_dim=action_dim, chunk_size=chunk_size,
            latent_dim=latent_dim, num_codes=num_codes,
            commitment_cost=commitment_cost)
        self.chunk_size = chunk_size
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.num_verbs = num_verbs

        # Transformer classifier over quantized token sequence
        # Project codebook vectors (latent_dim) to classifier's d_model
        self.cls_proj = nn.Linear(latent_dim, cls_d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cls_d_model))
        self.cls_pos = nn.Parameter(torch.zeros(1, max_chunks + 1, cls_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

        self.cls_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cls_d_model, nhead=cls_nhead,
                batch_first=True, activation='gelu',
                dropout=cls_dropout)
            for _ in range(cls_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(cls_d_model, cls_d_model // 2),
            nn.LayerNorm(cls_d_model // 2),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_d_model // 2, num_verbs))

    def forward(self, chunk_lists):
        """Trajectory-level forward pass.

        Args:
            chunk_lists: list of tensors, each (n_chunks_i, chunk_dim)
        Returns:
            dict with recon_loss, vq_loss, verb_logits
        """
        all_chunks = torch.cat(chunk_lists, dim=0)
        z = self.vqvae.encoder(all_chunks)
        indices, quantized = self.vqvae._nearest_codes(z)
        codebook_loss = F.mse_loss(z.detach(), quantized)
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.vqvae.commitment_cost * commitment_loss
        z_q = z + (quantized - z).detach()
        recon = self.vqvae.decoder(z_q)
        recon_loss = F.mse_loss(recon, all_chunks)

        # Split z_q back per trajectory, build padded batch for transformer
        B = len(chunk_lists)
        device = z_q.device
        lengths = [cl.size(0) for cl in chunk_lists]
        max_len = max(lengths)
        # (B, max_len, latent_dim)
        z_q_padded = torch.zeros(B, max_len, self.latent_dim, device=device)
        offset = 0
        for i, n in enumerate(lengths):
            z_q_padded[i, :n] = z_q[offset:offset + n]
            offset += n

        # Project to classifier d_model and prepend CLS
        tok_emb = self.cls_proj(z_q_padded)  # (B, max_len, cls_d_model)
        cls_expand = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_expand, tok_emb], dim=1)  # (B, 1+max_len, cls_d_model)
        seq = seq + self.cls_pos[:, :seq.size(1)]

        # Padding mask: True = padded (ignored)
        total_len = 1 + max_len  # CLS + tokens
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        # real length = 1 (CLS) + n_chunks_i
        real_lens = torch.tensor([1 + n for n in lengths], device=device).unsqueeze(1)
        padding_mask = positions >= real_lens

        for layer in self.cls_layers:
            seq = layer(seq, src_key_padding_mask=padding_mask)

        cls_out = seq[:, 0]  # CLS token output
        verb_logits = self.classifier(cls_out)

        return {
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'verb_logits': verb_logits,
        }


class CalvinTrajectoryDataset(Dataset):
    """Dataset returning chunked trajectories with verb labels for VQ-VAE training."""

    def __init__(self, df, data_dir, chunk_size, verb_to_id, max_chunks):
        self.df = df
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.verb_to_id = verb_to_id
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        actions = []
        for i in range(row['start_idx'], row['end_idx'] + 1):
            path = os.path.join(self.data_dir, EPISODE_TEMPLATE.format(i))
            data = np.load(path, mmap_mode='r')
            actions.append(np.array(data[ACTION_KEY]))
        actions = np.array(actions, dtype=np.float32)

        T = actions.shape[0]
        action_dim = actions.shape[1]
        chunk_dim = self.chunk_size * action_dim
        T_eff = (T // self.chunk_size) * self.chunk_size

        if T_eff == 0:
            chunks = np.zeros((1, chunk_dim), dtype=np.float32)
            n_chunks = 1
        else:
            chunks = actions[:T_eff].reshape(-1, chunk_dim)
            n_chunks = chunks.shape[0]

        if n_chunks < self.max_chunks:
            pad = np.zeros((self.max_chunks - n_chunks, chunk_dim), dtype=np.float32)
            chunks = np.concatenate([chunks, pad], axis=0)
        else:
            chunks = chunks[:self.max_chunks]
            n_chunks = self.max_chunks

        verb_id = self.verb_to_id.get(row['primary_verb'], 0)
        return (torch.from_numpy(chunks),
                torch.tensor(verb_id, dtype=torch.long),
                torch.tensor(n_chunks, dtype=torch.long))


def collect_chunks(df, data_dir, chunk_size):
    """Collect all non-overlapping action chunks from training trajectories.

    Args:
        chunk_size: number of timesteps per chunk
    Returns:
        np.ndarray of shape (N_chunks, chunk_size * action_dim)
    """
    all_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting chunks"):
        actions = []
        for i in range(row['start_idx'], row['end_idx'] + 1):
            path = os.path.join(data_dir, EPISODE_TEMPLATE.format(i))
            data = np.load(path, mmap_mode='r')
            actions.append(np.array(data[ACTION_KEY]))
        actions = np.array(actions, dtype=np.float32)   # (T, action_dim)
        T = actions.shape[0]
        T_eff = (T // chunk_size) * chunk_size
        if T_eff == 0:
            continue
        chunks = actions[:T_eff].reshape(T_eff // chunk_size, chunk_size * actions.shape[1])
        all_chunks.append(chunks)
    return np.concatenate(all_chunks, axis=0)            # (N_total, chunk_size * action_dim)


def fit_vqvae_tokenizer(df, data_dir, save_path,
                        chunk_size=4, num_codes=512, latent_dim=64,
                        epochs=100, batch_size=2048, lr=1e-3,
                        commitment_cost=0.25):
    """Fit a VQ-VAE tokenizer on CALVIN training trajectories.

    Saves vqvae.pth and vqvae_config.json to save_path.
    """
    chunks = collect_chunks(df, data_dir, chunk_size)
    action_dim = chunks.shape[1] // chunk_size
    input_dim = chunks.shape[1]

    print(f"Fitting VQ-VAE on {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, input_dim={input_dim}, "
          f"num_codes={num_codes}, latent_dim={latent_dim})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = ActionVQVAE(action_dim=action_dim, chunk_size=chunk_size,
                        latent_dim=latent_dim, num_codes=num_codes,
                        commitment_cost=commitment_cost).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.from_numpy(chunks))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        total_recon = total_vq = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            _, recon_loss, vq_loss = model(batch)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            n = len(loader)
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"recon={total_recon/n:.5f}, vq={total_vq/n:.5f}")

    # Codebook utilization
    model.eval()
    all_indices = []
    with torch.no_grad():
        chunks_t = torch.from_numpy(chunks).to(device)
        for i in range(0, len(chunks_t), batch_size):
            chunk_batch = chunks_t[i:i + batch_size]
            all_indices.append(model.encode(chunk_batch).cpu())
    all_indices = torch.cat(all_indices)
    used = all_indices.unique().numel()
    print(f"Codebook utilization: {used}/{num_codes} codes used "
          f"({100 * used / num_codes:.1f}%)")

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "vqvae.pth"))
    meta = {
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "latent_dim": latent_dim,
        "num_codes": num_codes,
        "commitment_cost": commitment_cost,
    }
    with open(os.path.join(save_path, "vqvae_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"VQ-VAE tokenizer saved to {save_path}")
    return model


def fit_verb_decodable_vqvae(df, data_dir, save_path,
                              chunk_size=4, num_codes=512, latent_dim=64,
                              epochs=200, batch_size=64, lr=1e-3,
                              commitment_cost=0.25, verb_loss_weight=1.0,
                              weighted_verb_loss=True, min_class_count=30,
                              cls_d_model=128, cls_nhead=4, cls_layers=2,
                              cls_dropout=0.1):
    """Fit a verb-decodable VQ-VAE tokenizer on CALVIN training trajectories.

    Joint training: total_loss = recon_loss + vq_loss + lambda * verb_CE.
    The transformer classifier over quantized token sequences shapes the
    codebook to preserve verb-discriminative information.

    Saves vqvae.pth (inner ActionVQVAE weights, downstream-compatible),
    vqvae_verb_full.pth (full model), and vqvae_config.json to save_path.
    """
    # Filter sparse classes
    if min_class_count > 0:
        verb_counts = df['primary_verb'].value_counts()
        keep_verbs = set(verb_counts[verb_counts >= min_class_count].index)
        n_before = len(df)
        df = df[df['primary_verb'].isin(keep_verbs)].reset_index(drop=True)
        print("Filtered classes with <{} samples: {}→{} classes, {}→{} samples".format(
            min_class_count, len(verb_counts), len(keep_verbs), n_before, len(df)))

    unique_verbs = sorted(df['primary_verb'].unique())
    verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
    num_verbs = len(verb_to_id)
    print("Verb classes ({}): {}".format(num_verbs, unique_verbs))

    # Compute max_chunks across dataset
    traj_lengths = df['end_idx'] - df['start_idx'] + 1
    max_chunks = int(traj_lengths.max() // chunk_size) + 1

    dataset = CalvinTrajectoryDataset(df, data_dir, chunk_size, verb_to_id, max_chunks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training verb-decodable VQ-VAE on {} ({} trajectories, max_chunks={}, "
          "num_codes={}, latent_dim={}, verb_loss_weight={})".format(
              device, len(dataset), max_chunks, num_codes, latent_dim, verb_loss_weight))

    model = VerbDecodableVQVAE(
        num_verbs=num_verbs, chunk_size=chunk_size,
        latent_dim=latent_dim, num_codes=num_codes,
        commitment_cost=commitment_cost,
        cls_d_model=cls_d_model, cls_nhead=cls_nhead,
        cls_layers=cls_layers, max_chunks=max_chunks,
        cls_dropout=cls_dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Optionally weight verb loss by inverse frequency
    if weighted_verb_loss:
        class_counts = df['primary_verb'].value_counts()
        weights = torch.zeros(num_verbs)
        for verb, cid in verb_to_id.items():
            weights[cid] = 1.0 / class_counts.get(verb, 1)
        weights = weights / weights.sum() * num_verbs
        verb_criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        verb_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_recon = total_vq = total_verb = 0.0
        correct = total = 0

        for chunks_padded, verb_ids, n_chunks in loader:
            chunks_padded = chunks_padded.to(device)
            verb_ids = verb_ids.to(device)

            # Build list of real (unpadded) chunk tensors per trajectory
            chunk_lists = []
            for i in range(chunks_padded.size(0)):
                nc = n_chunks[i].item()
                chunk_lists.append(chunks_padded[i, :nc])

            result = model(chunk_lists)
            verb_loss = verb_criterion(result['verb_logits'], verb_ids)
            loss = result['recon_loss'] + result['vq_loss'] + verb_loss_weight * verb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += result['recon_loss'].item()
            total_vq += result['vq_loss'].item()
            total_verb += verb_loss.item()
            preds = result['verb_logits'].argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)

        scheduler.step()
        n = len(loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = 100.0 * correct / max(total, 1)
            print("  Epoch {:3d}/{}: recon={:.5f}, vq={:.5f}, "
                  "verb={:.5f}, verb_acc={:.1f}%".format(
                      epoch + 1, epochs, total_recon / n, total_vq / n,
                      total_verb / n, acc))

    # Codebook utilization
    model.eval()
    all_indices = []
    with torch.no_grad():
        for chunks_padded, _, n_chunks in loader:
            chunks_padded = chunks_padded.to(device)
            for i in range(chunks_padded.size(0)):
                nc = n_chunks[i].item()
                real_chunks = chunks_padded[i, :nc]
                all_indices.append(model.vqvae.encode(real_chunks).cpu())
    all_indices = torch.cat(all_indices)
    used = all_indices.unique().numel()
    print("Codebook utilization: {}/{} codes used ({:.1f}%)".format(
        used, num_codes, 100.0 * used / num_codes))

    # Save
    os.makedirs(save_path, exist_ok=True)
    # Inner ActionVQVAE weights — downstream load_vqvae_tokenizer() compatible
    torch.save(model.vqvae.state_dict(), os.path.join(save_path, "vqvae.pth"))
    # Full model (for analysis / resume)
    torch.save(model.state_dict(), os.path.join(save_path, "vqvae_verb_full.pth"))
    meta = {
        "action_dim": 7,
        "chunk_size": chunk_size,
        "latent_dim": latent_dim,
        "num_codes": num_codes,
        "commitment_cost": commitment_cost,
        "verb_loss_weight": verb_loss_weight,
        "num_verbs": num_verbs,
        "verb_to_id": verb_to_id,
    }
    with open(os.path.join(save_path, "vqvae_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Verb-decodable VQ-VAE saved to {}".format(save_path))
    return model


def load_vqvae_tokenizer(path=VQVAE_TOKENIZER_PATH):
    """Load a fitted VQ-VAE tokenizer (on CPU, in eval mode)."""
    with open(os.path.join(path, "vqvae_config.json")) as f:
        meta = json.load(f)
    # Filter to only ActionVQVAE constructor args (verb-decodable saves extra keys)
    _vqvae_keys = {"action_dim", "chunk_size", "latent_dim", "num_codes", "commitment_cost"}
    model = ActionVQVAE(**{k: v for k, v in meta.items() if k in _vqvae_keys})
    state = torch.load(os.path.join(path, "vqvae.pth"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def tokenize_trajectory_vqvae(model, actions_np, batch_size=1024):
    """Tokenize a single trajectory using VQ-VAE.

    Args:
        model: ActionVQVAE loaded via load_vqvae_tokenizer (CPU, eval)
        actions_np: (T, action_dim) numpy array
    Returns:
        np.ndarray of shape (T // chunk_size,) with int64 code indices
        (last partial chunk is discarded)
    """
    chunk_size = model.chunk_size
    T = actions_np.shape[0]
    T_eff = (T // chunk_size) * chunk_size
    if T_eff == 0:
        return np.array([], dtype=np.int64)

    chunks = actions_np[:T_eff].astype(np.float32)
    chunks = chunks.reshape(T_eff // chunk_size, chunk_size * actions_np.shape[1])
    chunks_t = torch.from_numpy(chunks)

    all_indices = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks_t), batch_size):
            batch = chunks_t[i:i + batch_size]
            all_indices.append(model.encode(batch))
    return torch.cat(all_indices).numpy().astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# VQ-VLA: published pretrained causal-VAE + ResidualVQ tokenizer
#   Architecture: causal conv VAE encoder, ResidualVQ (4 quantizers, codebook=256)
#   Output: 4 integer codes per trajectory (whole trajectory → 4 tokens)
#   Pretrained on Open X-Embodiment + LIBERO + RH20T + ManiSkill + RLBench
#   Paper: "Improving VLA Models via Scaling Vector-Quantized Action Tokenizers"
#   Repo:  https://github.com/xiaoxiao0406/VQ-VLA
#   Weights: HuggingFace VQ-VLA/vq-vla-weight (action_tokenizer_weight/all_data_vq.pth)
#
# The vendored module lives in vqvla/ (copied from prismatic/action_vqvae/).
# ─────────────────────────────────────────────────────────────────────────────

# Number of tokens and vocab size are fixed by the VQ-VLA architecture:
# The pretrained checkpoint uses ActionVQVAEPE with use_action_type_pe=True, use_time_pe=True.
# The PE time embedding has temporal_compression_ratio=5, so exactly T=5 steps per window
# produce encoder output (B,128,1,1) → flat (B,128) matching vq_embed_dim=128.
VQVLA_WINDOW_SIZE = 5   # temporal_compression_ratio in ActionVQVAEPE; T=5 → enc (B,128,1,1)
VQVLA_NUM_TOKENS = 4    # vqvae_groups = 4  (tokens per window, fixed by ResidualVQ)
VQVLA_VOCAB_SIZE = 256  # vqvae_n_embed = 256 (hardcoded in modeling_causal_vae.py)
# Tokens per trajectory: (T // VQVLA_WINDOW_SIZE) * VQVLA_NUM_TOKENS ≈ 12*4=48 for T=61

# Default paths
VQVLA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "vqvla", "config")
VQVLA_CHECKPOINT_PATH = "./checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth"


def download_pretrained_vqvla(save_dir="./checkpoints/vqvla_pretrained"):
    """Download VQ-VLA pretrained action tokenizer weights from HuggingFace.

    Downloads action_tokenizer_weight/all_data_vq.pth (~1.4 GB) from
    VQ-VLA/vq-vla-weight into save_dir, preserving the subdirectory structure.

    Returns the path to the downloaded .pth file.
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading VQ-VLA pretrained weights to {save_dir} ...")
    path = hf_hub_download(
        repo_id="VQ-VLA/vq-vla-weight",
        filename="action_tokenizer_weight/all_data_vq.pth",
        local_dir=save_dir,
    )
    print(f"Downloaded to {path}")
    return path


def load_vqvla_tokenizer(config_dir=VQVLA_CONFIG_DIR, checkpoint_path=VQVLA_CHECKPOINT_PATH):
    """Load VQ-VLA ActionVQVAELossWrapper on CPU in eval/frozen mode.

    Args:
        config_dir:       Directory containing config.json (default: ./vqvla_config/)
        checkpoint_path:  Path to all_data_vq.pth pretrained weights.
                          Pass None to get an untrained model (for smoke-testing).
    Returns:
        ActionVQVAELossWrapper in eval mode, parameters frozen, on CPU.
    """
    # Ensure the vendored vqvla package is importable from this project
    import sys
    _pkg_dir = os.path.dirname(__file__)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)

    from tokenization.vqvla import ActionVQVAELossWrapper

    ckpt = checkpoint_path if (checkpoint_path and os.path.isfile(checkpoint_path)) else None
    if checkpoint_path and not ckpt:
        print(f"[WARNING] VQ-VLA checkpoint not found at {checkpoint_path}. "
              f"Run download_pretrained_vqvla() first.")

    # The published checkpoint uses ActionVQVAEPE (action-type + time PE).
    # Must pass these flags so the wrapper picks config_action_type_pe_time_pe.json
    # and builds the right encoder (in_channels=21, time_emb for T=5 windows).
    wrapper = ActionVQVAELossWrapper(
        model_path=config_dir,
        checkpoint_path=ckpt,
        is_eval=True,
        freeze=True,
        use_action_type_pe=True,
        use_time_pe=True,
    )
    wrapper.eval()
    # Move to CPU explicitly (preprocess() uses self.device which follows parameters)
    wrapper = wrapper.cpu()
    print(f"Loaded VQ-VLA tokenizer from {config_dir} "
          f"(checkpoint={'pretrained' if ckpt else 'random'}, "
          f"tokens={VQVLA_NUM_TOKENS}, vocab={VQVLA_VOCAB_SIZE})")
    return wrapper


def tokenize_trajectory_vqvla(wrapper, actions_np):
    """Tokenize a single trajectory with the VQ-VLA tokenizer.

    The pretrained checkpoint uses ActionVQVAEPE with temporal_compression_ratio=5,
    so the time embedding requires exactly T=VQVLA_WINDOW_SIZE=5 steps per window.
    We split the trajectory into non-overlapping 5-step windows, encode each
    independently (→ 4 codes), and concatenate all codes.

    Args:
        wrapper:     ActionVQVAELossWrapper (from load_vqvla_tokenizer, CPU, eval)
        actions_np:  (T, action_dim) numpy float32 array
    Returns:
        np.ndarray of shape (n_windows * VQVLA_NUM_TOKENS,) with int64 code indices,
        each in {0 .. VQVLA_VOCAB_SIZE-1} = {0..255}.
        For T=61 and window_size=5: shape = (12*4,) = (48,).
        If T < VQVLA_WINDOW_SIZE, the trajectory is edge-padded and encoded once.
    """
    T = actions_np.shape[0]
    n_windows = T // VQVLA_WINDOW_SIZE
    if n_windows == 0:
        # Trajectory too short; pad to VQVLA_WINDOW_SIZE and encode once
        padded = np.pad(actions_np, ((0, VQVLA_WINDOW_SIZE - T), (0, 0)), mode='edge')
        actions_t = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            vq_codes = wrapper.get_code(actions_t)   # (1, 4)
        return vq_codes[0].cpu().numpy().astype(np.int64)

    all_codes = []
    for i in range(n_windows):
        window = actions_np[i * VQVLA_WINDOW_SIZE : (i + 1) * VQVLA_WINDOW_SIZE]
        actions_t = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1, 8, 7)
        with torch.no_grad():
            vq_codes = wrapper.get_code(actions_t)   # (1, 4)
        all_codes.append(vq_codes[0].cpu().numpy().astype(np.int64))          # (4,)
    return np.concatenate(all_codes, axis=0)  # (n_windows * 4,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a VQ-VAE chunk tokenizer on CALVIN training trajectories")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--save_path", type=str, default=VQVAE_TOKENIZER_PATH)
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Number of timesteps per chunk (K)")
    parser.add_argument("--num_codes", type=int, default=512,
                        help="VQ-VAE codebook size (vocabulary)")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent dimension of VQ-VAE encoder output")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for VQ-VAE fitting")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Use only N samples for quick testing")
    # Verb-decodable VQ-VAE args
    parser.add_argument("--verb_decodable", action="store_true",
                        help="Train verb-decodable VQ-VAE (joint recon + classification)")
    parser.add_argument("--verb_loss_weight", type=float, default=1.0,
                        help="Weight lambda for verb classification loss")
    parser.add_argument("--min_class_count", type=int, default=30,
                        help="Drop verb classes with fewer than N training samples")
    parser.add_argument("--weighted_verb_loss", action="store_true", default=True,
                        help="Use inverse-frequency weighted CE for verb loss")
    # Classifier architecture (verb-decodable only)
    parser.add_argument("--cls_d_model", type=int, default=128,
                        help="Classifier transformer d_model")
    parser.add_argument("--cls_layers", type=int, default=2,
                        help="Classifier transformer layers")
    parser.add_argument("--cls_nhead", type=int, default=4,
                        help="Classifier transformer attention heads")
    parser.add_argument("--cls_dropout", type=float, default=0.1,
                        help="Classifier dropout rate")
    args = parser.parse_args()

    df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        df = df.head(min(args.debug, len(df))).copy()
        print(f"[DEBUG] Using {len(df)} samples")

    if args.verb_decodable:
        fit_verb_decodable_vqvae(
            df, args.data_dir, args.save_path,
            chunk_size=args.chunk_size, num_codes=args.num_codes,
            latent_dim=args.latent_dim, epochs=args.epochs,
            batch_size=min(args.batch_size, 64),
            lr=args.lr, commitment_cost=args.commitment_cost,
            verb_loss_weight=args.verb_loss_weight,
            weighted_verb_loss=args.weighted_verb_loss,
            min_class_count=args.min_class_count,
            cls_d_model=args.cls_d_model, cls_nhead=args.cls_nhead,
            cls_layers=args.cls_layers, cls_dropout=args.cls_dropout,
        )
    else:
        fit_vqvae_tokenizer(
            df, args.data_dir, args.save_path,
            chunk_size=args.chunk_size, num_codes=args.num_codes,
            latent_dim=args.latent_dim, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            commitment_cost=args.commitment_cost,
        )
