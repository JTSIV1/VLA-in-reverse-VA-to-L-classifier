"""VQ-VAE chunk-based action tokenizer for CALVIN action trajectories.

Each trajectory (T, 7) is split into non-overlapping chunks of size K.
Each chunk (K*7-dim) is encoded by a small MLP into a learned codebook index.
Sequence length after tokenization: T // K (last partial chunk discarded).

Comparison to FAST:
  FAST: DCT + BPE, sequence-level compression, ~25 tokens regardless of K
  VQ-VAE: learned, per-chunk, ~T/K tokens (30/15/7 for K=2/4/8 on CALVIN ~61-step trajectories)

Usage (standalone, to fit):
    python vqvae_tokenizer.py --save_path ./checkpoints/vqvae_k4_c512 --chunk_size 4 --num_codes 512

Usage (as module):
    from vqvae_tokenizer import load_vqvae_tokenizer, tokenize_trajectory_vqvae
    tok = load_vqvae_tokenizer("./checkpoints/vqvae_k4_c512")
    token_ids = tokenize_trajectory_vqvae(tok, actions_np)  # (T, 7) -> (T//K,) np.int64
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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


def load_vqvae_tokenizer(path=VQVAE_TOKENIZER_PATH):
    """Load a fitted VQ-VAE tokenizer (on CPU, in eval mode)."""
    with open(os.path.join(path, "vqvae_config.json")) as f:
        meta = json.load(f)
    model = ActionVQVAE(**meta)
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
VQVLA_NUM_TOKENS = 4    # vqvae_groups = 4  (tokens per trajectory)
VQVLA_VOCAB_SIZE = 256  # vqvae_n_embed = 256 (hardcoded in modeling_causal_vae.py)

# Default paths
VQVLA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "vqvla_config")
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

    from vqvla import ActionVQVAELossWrapper

    ckpt = checkpoint_path if (checkpoint_path and os.path.isfile(checkpoint_path)) else None
    if checkpoint_path and not ckpt:
        print(f"[WARNING] VQ-VLA checkpoint not found at {checkpoint_path}. "
              f"Run download_pretrained_vqvla() first.")

    wrapper = ActionVQVAELossWrapper(
        model_path=config_dir,
        checkpoint_path=ckpt,
        is_eval=True,
        freeze=True,
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

    Args:
        wrapper:     ActionVQVAELossWrapper (from load_vqvla_tokenizer, CPU, eval)
        actions_np:  (T, action_dim) numpy float32 array
    Returns:
        np.ndarray of shape (VQVLA_NUM_TOKENS,) = (4,) with int64 code indices,
        each in {0 .. VQVLA_VOCAB_SIZE-1} = {0..255}.
    """
    actions_t = torch.from_numpy(actions_np.astype(np.float32)).unsqueeze(0)  # (1, T, 7)
    vq_codes = wrapper.get_code(actions_t)   # (1, 4)
    return vq_codes[0].cpu().numpy().astype(np.int64)                          # (4,)


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
    args = parser.parse_args()

    df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        df = df.head(min(args.debug, len(df))).copy()
        print(f"[DEBUG] Using {len(df)} samples")

    fit_vqvae_tokenizer(
        df, args.data_dir, args.save_path,
        chunk_size=args.chunk_size, num_codes=args.num_codes,
        latent_dim=args.latent_dim, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        commitment_cost=args.commitment_cost,
    )
