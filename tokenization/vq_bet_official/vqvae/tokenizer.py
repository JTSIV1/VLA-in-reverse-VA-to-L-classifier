"""VQ-BeT action tokenizer (nn.Module wrapper).

MLP encoder/decoder + ResidualVQ, adapted from vqvae.py in this directory.
This is the version used by train_tokenizer.py — a clean nn.Module with
forward(), encode(), decode() matching the OAT/QueST tokenizer interface.

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

        self.vocab_size = n_embed
        self.total_codes = n_embed ** groups

        # Action normalizer (fitted externally via set_normalizer)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_flat(self, x):
        """Normalize flat chunks (B, chunk_size * action_dim) via the fitted normalizer."""
        B = x.size(0)
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
        z = self.encoder(x_norm)
        z = z.unsqueeze(1)
        quantized, indices, commit_loss = self.vq_layer(z)
        quantized = quantized.squeeze(1)
        indices = indices.squeeze(1)
        commit_loss = commit_loss.sum()
        return quantized, indices, commit_loss

    def decode(self, quantized):
        """Decode quantized latents to reconstructed chunks (normalized space)."""
        return self.decoder(quantized)

    def forward(self, x):
        """Training forward pass. MSE computed in normalized [-1,1] space.

        Args:
            x: (B, chunk_size * action_dim) flat chunks (raw actions)
        Returns:
            dict with recon_loss, vq_loss, latents (B, latent_dim),
            codes (B, groups).
        """
        x_norm = self._normalize_flat(x)
        z = self.encoder(x_norm)
        z = z.unsqueeze(1)
        quantized, indices, commit_loss = self.vq_layer(z)
        quantized = quantized.squeeze(1)  # (B, latent_dim)
        indices = indices.squeeze(1)      # (B, groups)
        commit_loss = commit_loss.sum()
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x_norm)
        return {
            'recon_loss': recon_loss,
            'vq_loss': commit_loss,
            'latents': quantized,   # post-VQ, gradient via STE
            'codes': indices.detach(),
        }

    def get_code(self, x):
        """Get codebook indices for a batch of chunks."""
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
        indices = indices.unsqueeze(1)
        quantized = self.vq_layer.get_output_from_indices(indices)
        quantized = quantized.squeeze(1)
        return self.decode(quantized)
