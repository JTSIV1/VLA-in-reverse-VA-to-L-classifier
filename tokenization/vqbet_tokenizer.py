"""VQ-BeT action tokenizer.

MLP encoder/decoder + ResidualVQ, adapted from:
  https://github.com/jayLEE0301/vq_bet_official/blob/main/vqvae/vqvae.py

Much smaller than VQ-VLA (~250K vs ~3.5M params) while using residual
quantization (multiple codebook groups) for better expressiveness than
a single-codebook VQ-VAE.

Default architecture:
  Encoder: Linear(chunk_dim -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> latent_dim)
  VQ:      ResidualVQ(dim=latent_dim, num_quantizers=groups, codebook_size=n_embed)
  Decoder: Linear(latent_dim -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> chunk_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ
from oat.model.common.normalizer import LinearNormalizer


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EncoderMLP(nn.Module):
    """Simple MLP encoder/decoder (shared architecture)."""

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(_init_weights)

    def forward(self, x):
        return self.fc(self.encoder(x))


class VQBeTTokenizer(nn.Module):
    """VQ-BeT action tokenizer: MLP encoder + ResidualVQ + MLP decoder.

    Processes action chunks of shape (chunk_size * action_dim,) through
    a simple MLP encoder, quantizes via residual VQ, and reconstructs.

    Args:
        action_dim: per-timestep action dimension (default: 7)
        chunk_size: number of timesteps per chunk (default: 10)
        latent_dim: encoder output / codebook dimension (default: 512)
        n_embed: codebook size per quantizer group (default: 32)
        groups: number of residual quantizer groups (default: 4)
        hidden_dim: MLP hidden dimension (default: 128)
        num_layers: number of hidden layers in encoder/decoder (default: 1)
    """

    def __init__(self, action_dim=7, chunk_size=10, latent_dim=512,
                 n_embed=32, groups=4, hidden_dim=128, num_layers=1):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.groups = groups

        input_dim = action_dim * chunk_size

        self.encoder = EncoderMLP(input_dim, latent_dim,
                                  hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = EncoderMLP(latent_dim, input_dim,
                                  hidden_dim=hidden_dim, num_layers=num_layers)
        self.vq_layer = ResidualVQ(
            dim=latent_dim,
            num_quantizers=groups,
            codebook_size=n_embed,
        )

        # Vocab size = n_embed (each group has same codebook size)
        self.vocab_size = n_embed
        # Total discrete states = n_embed ^ groups
        self.total_codes = n_embed ** groups

        # Action normalizer (fitted externally via set_normalizer)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_flat(self, x):
        """Normalize flat chunks (B, chunk_size * action_dim) via the fitted normalizer."""
        B = x.size(0)
        # Reshape to (B, chunk_size, action_dim) for normalizer
        x_3d = x.view(B, self.chunk_size, self.action_dim)
        x_3d = self.normalizer['action'].normalize(x_3d)
        return x_3d.view(B, -1)

    def _unnormalize_flat(self, x):
        """Unnormalize flat chunks back to original action space."""
        B = x.size(0)
        x_3d = x.view(B, self.chunk_size, self.action_dim)
        x_3d = self.normalizer['action'].unnormalize(x_3d)
        return x_3d.view(B, -1)

    def encode(self, x):
        """Encode flattened chunks to quantized latents.

        Args:
            x: (B, chunk_size * action_dim) flat chunks (raw actions)
        Returns:
            quantized: (B, latent_dim)
            indices: (B, groups) codebook indices per group
            commit_loss: scalar commitment loss
        """
        x_norm = self._normalize_flat(x)
        z = self.encoder(x_norm)           # (B, latent_dim)
        z = z.unsqueeze(1)                 # (B, 1, latent_dim) for ResidualVQ
        quantized, indices, commit_loss = self.vq_layer(z)
        quantized = quantized.squeeze(1)   # (B, latent_dim)
        indices = indices.squeeze(1)       # (B, groups)
        commit_loss = commit_loss.sum()
        return quantized, indices, commit_loss

    def decode(self, quantized):
        """Decode quantized latents to reconstructed chunks (normalized space).

        Args:
            quantized: (B, latent_dim)
        Returns:
            recon: (B, chunk_size * action_dim) in normalized space
        """
        return self.decoder(quantized)

    def forward(self, x):
        """Training forward pass. MSE computed in normalized [-1,1] space.

        Args:
            x: (B, chunk_size * action_dim) flat chunks (raw actions)
        Returns:
            recon: (B, chunk_size * action_dim) in normalized space
            recon_loss: scalar (MSE in normalized space)
            vq_loss: scalar (commitment loss from ResidualVQ)
        """
        x_norm = self._normalize_flat(x)
        z = self.encoder(x_norm)
        z = z.unsqueeze(1)
        quantized, indices, commit_loss = self.vq_layer(z)
        quantized = quantized.squeeze(1)
        indices = indices.squeeze(1)
        commit_loss = commit_loss.sum()
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x_norm)
        return recon, recon_loss, commit_loss

    def get_code(self, x):
        """Get codebook indices for a batch of chunks.

        Args:
            x: (B, chunk_size * action_dim)
        Returns:
            indices: (B, groups) int64
        """
        with torch.no_grad():
            _, indices, _ = self.encode(x)
        return indices

    def decode_from_indices(self, indices):
        """Decode from codebook indices.

        Args:
            indices: (B, groups) int64
        Returns:
            recon: (B, chunk_size * action_dim)
        """
        # Look up codebook vectors and sum (residual)
        indices = indices.unsqueeze(1)  # (B, 1, groups)
        quantized = self.vq_layer.get_output_from_indices(indices)
        quantized = quantized.squeeze(1)  # (B, latent_dim)
        return self.decode(quantized)
