"""
CLIP-style contrastive action-language alignment for VQ-VAE action tokenizers.

Architecture:
  Action branch:  actions → Tiny VQ-VLA encoder → quantize (STE) → Transformer → project → L2-norm
  Language branch: instruction → Text Encoder (CLIP/GPT-2) → project → L2-norm
  Loss: L_recon + L_vq + lambda * L_contrastive (symmetric InfoNCE)

Usage:
  python tokenization/clip_action_language.py --data_root /path/to/calvin --save_dir checkpoints/clip_tokenizer
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Workaround: prevent transformers from importing tensorflow (numpy 2.x crash
# in mmml env). USE_TF=0 is the official transformers env var to disable TF.
os.environ['USE_TF'] = '0'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (DATA_ROOT, TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE,
                     ACTION_KEY)


# ============================================================================
# Causal Convolution Building Blocks (reimplemented without diffusers)
# ============================================================================

class CausalConv2d(nn.Module):
    """2D convolution with causal (left-only) padding on the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        time_ks, action_ks = kernel_size
        self.time_pad = dilation * (time_ks - 1)
        self.action_pad = action_ks // 2
        self.causal_padding = (self.action_pad, self.action_pad,
                               self.time_pad, 0)  # (left, right, top, bottom)

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.pad(x, self.causal_padding, mode='constant', value=0)
        return self.conv(x)


class CausalGroupNorm(nn.GroupNorm):
    """GroupNorm that handles 4D tensors (B, C, T, D) by folding time into batch."""

    def forward(self, x):
        if x.dim() == 4:
            b, c, t, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * t, c, d)
            x = super().forward(x)
            x = x.reshape(b, t, c, d).permute(0, 2, 1, 3)
            return x
        return super().forward(x)


class CausalResnetBlock2D(nn.Module):
    """Residual block with causal convolutions, GroupNorm, and SiLU."""

    def __init__(self, in_channels, out_channels=None, groups=32, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        # Clamp groups to not exceed channel count
        groups_1 = min(groups, in_channels)
        groups_2 = min(groups, out_channels)

        self.norm1 = CausalGroupNorm(groups_1, in_channels, eps=1e-6)
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)

        self.norm2 = CausalGroupNorm(groups_2, out_channels, eps=1e-6)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = CausalConv2d(in_channels, out_channels,
                                         kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


# ============================================================================
# Vector Quantization (simple implementation, no external dependencies)
# ============================================================================

class VectorQuantize(nn.Module):
    """Single-codebook vector quantization with straight-through estimator."""

    def __init__(self, dim, codebook_size=256, commitment_weight=0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, dim)
        # Initialize uniformly
        limit = 1.0 / codebook_size
        nn.init.uniform_(self.codebook.weight, -limit, limit)

    def forward(self, x):
        """
        Args:
            x: (B, D) latent vectors
        Returns:
            quantized: (B, D) quantized vectors (with STE gradients)
            indices: (B,) codebook indices
            loss: scalar VQ loss (codebook + commitment)
        """
        # Compute distances: ||x - e||^2
        dists = (torch.sum(x ** 2, dim=-1, keepdim=True)
                 + torch.sum(self.codebook.weight ** 2, dim=-1)
                 - 2 * x @ self.codebook.weight.T)
        indices = dists.argmin(dim=-1)
        quantized = self.codebook(indices)

        # Losses
        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, indices, loss


class ResidualVQ(nn.Module):
    """Residual vector quantization with multiple sequential codebooks."""

    def __init__(self, dim, num_quantizers=2, codebook_size=256,
                 commitment_weight=0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantize(dim, codebook_size, commitment_weight)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, D) latent vectors
        Returns:
            quantized: (B, D) sum of all quantized residuals
            all_indices: (B, num_quantizers) codebook indices per stage
            total_loss: scalar sum of VQ losses
        """
        quantized_out = torch.zeros_like(x)
        residual = x
        all_indices = []
        total_loss = 0.0

        for quantizer in self.quantizers:
            quantized, indices, loss = quantizer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        all_indices = torch.stack(all_indices, dim=-1)  # (B, num_quantizers)
        return quantized_out, all_indices, total_loss


# ============================================================================
# Tiny VQ-VLA Encoder / Decoder
# ============================================================================

class TinyVQVLAEncoder(nn.Module):
    """
    Causal convolutional encoder: (B, 1, T=5, 7) → (B, latent_dim).
    3 stages: channels 32→64→128, 2 ResNet blocks per stage.
    """

    def __init__(self, in_channels=1, latent_dim=64,
                 block_channels=(32, 64, 128), blocks_per_stage=2,
                 num_mid_blocks=2, norm_groups=32):
        super().__init__()
        self.conv_in = CausalConv2d(in_channels, block_channels[0], kernel_size=3)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = block_channels[0]
        for i, ch in enumerate(block_channels):
            blocks = nn.ModuleList()
            for j in range(blocks_per_stage):
                in_ch = prev_ch if j == 0 else ch
                blocks.append(CausalResnetBlock2D(in_ch, ch, groups=norm_groups))
            self.down_blocks.append(blocks)
            prev_ch = ch
            # Downsample between stages (not after last)
            if i < len(block_channels) - 1:
                self.down_convs.append(
                    CausalConv2d(ch, ch, kernel_size=3, stride=2))
            else:
                self.down_convs.append(nn.Identity())

        # Mid blocks
        self.mid_blocks = nn.ModuleList([
            CausalResnetBlock2D(block_channels[-1], block_channels[-1],
                                groups=norm_groups)
            for _ in range(num_mid_blocks)
        ])

        # Output
        out_groups = min(norm_groups, block_channels[-1])
        self.norm_out = CausalGroupNorm(out_groups, block_channels[-1], eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv2d(block_channels[-1], latent_dim, kernel_size=3)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T, 7) raw action window
        Returns:
            z: (B, latent_dim) encoded latent
        """
        x = self.conv_in(x)

        for blocks, down_conv in zip(self.down_blocks, self.down_convs):
            for block in blocks:
                x = block(x)
            x = down_conv(x)

        for block in self.mid_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        # Global pool to (B, latent_dim)
        x = x.flatten(2).mean(dim=-1)  # (B, C, T*D) -> (B, C)
        return x


class TinyVQVLADecoder(nn.Module):
    """
    MLP decoder: (B, latent_dim) → (B, T=5, 7).
    Uses an MLP instead of transposed convolutions because 7-d action vectors
    have no spatial structure for convolutions to exploit. The conv decoder
    bottlenecked at Linear(1, 7) and could only predict the mean.
    """

    def __init__(self, latent_dim=64, action_window_size=5, action_dim=7,
                 hidden_dim=256, num_layers=3, **kwargs):
        super().__init__()
        self.action_window_size = action_window_size
        self.action_dim = action_dim
        out_dim = action_window_size * action_dim

        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, latent_dim) quantized latent
        Returns:
            actions: (B, action_window_size, action_dim)
        """
        x = self.mlp(x)
        return x.reshape(-1, self.action_window_size, self.action_dim)


class TinyVQVLA(nn.Module):
    """
    Tiny VQ-VLA: causal convolutional VQ-VAE for action sequences.
    Input: 5-step action windows (B, 5, 7)
    Output: quantized latents + reconstructed actions
    ~3.7M parameters.
    """

    def __init__(self, action_dim=7, action_window_size=5,
                 latent_dim=64, block_channels=(32, 64, 128),
                 blocks_per_stage=2, num_mid_blocks=2,
                 num_quantizers=2, codebook_size=256,
                 commitment_weight=0.25, norm_groups=32):
        super().__init__()
        self.action_dim = action_dim
        self.action_window_size = action_window_size
        self.latent_dim = latent_dim

        self.encoder = TinyVQVLAEncoder(
            in_channels=1, latent_dim=latent_dim,
            block_channels=block_channels,
            blocks_per_stage=blocks_per_stage,
            num_mid_blocks=num_mid_blocks,
            norm_groups=norm_groups)

        self.vq = ResidualVQ(
            dim=latent_dim, num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight)

        self.decoder = TinyVQVLADecoder(
            latent_dim=latent_dim,
            action_window_size=action_window_size,
            action_dim=action_dim)

    def encode(self, actions):
        """Encode action window to latent.
        Args:
            actions: (B, T, 7) action window
        Returns:
            z: (B, latent_dim) pre-quantization latent
        """
        x = actions.unsqueeze(1)  # (B, 1, T, 7)
        return self.encoder(x)

    def quantize(self, z):
        """Quantize latent vectors.
        Args:
            z: (B, latent_dim)
        Returns:
            quantized, indices, vq_loss
        """
        return self.vq(z)

    def decode(self, quantized):
        """Decode quantized latent to actions.
        Args:
            quantized: (B, latent_dim)
        Returns:
            actions: (B, T, 7)
        """
        return self.decoder(quantized)

    def forward(self, actions):
        """Full forward pass.
        Args:
            actions: (B, T, 7) action window
        Returns:
            recon: (B, T, 7) reconstructed actions
            vq_loss: scalar
            z_q: (B, latent_dim) quantized latent
            indices: (B, num_quantizers) codebook indices
        """
        z = self.encode(actions)
        z_q, indices, vq_loss = self.quantize(z)
        recon = self.decode(z_q)
        return recon, vq_loss, z_q, indices


# ============================================================================
# Action Transformer (sequence of quantized tokens → single embedding)
# ============================================================================

class ActionTransformer(nn.Module):
    """Transformer that pools a sequence of VQ tokens into a single embedding."""

    def __init__(self, input_dim=64, d_model=128, nhead=4, num_layers=2,
                 dropout=0.1, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: (B, S, input_dim) sequence of quantized token vectors
            mask: (B, S) bool mask, True = padded position
        Returns:
            cls_out: (B, d_model) CLS token output
        """
        B, S, _ = tokens.shape
        x = self.input_proj(tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, S+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed[:, :S + 1, :]

        # Build attention mask (True = ignore in PyTorch transformer)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_out = self.norm(x[:, 0, :])
        return cls_out


# ============================================================================
# Text Encoder Wrapper
# ============================================================================

class TextEncoderWrapper(nn.Module):
    """Wraps a pretrained text encoder (CLIP or GPT-2) with optional LoRA."""

    def __init__(self, model_name='laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
                 model_type='clip', freeze=True, lora_r=0):
        super().__init__()
        self.model_type = model_type
        self.freeze = freeze
        self.lora_r = lora_r

        if model_type == 'clip':
            from transformers import CLIPModel, CLIPTokenizerFast
            clip_model = CLIPModel.from_pretrained(model_name)
            self.text_model = clip_model.text_model
            self.text_projection = clip_model.text_projection
            self.tokenizer = CLIPTokenizerFast.from_pretrained(
                'openai/clip-vit-base-patch32')
            self.output_dim = clip_model.config.projection_dim  # 512
            del clip_model.vision_model  # Free memory
        elif model_type == 'gpt2':
            from transformers import GPT2Model, GPT2Tokenizer
            self.text_model = GPT2Model.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.text_projection = None
            self.output_dim = self.text_model.config.hidden_size  # 768
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if freeze:
            for p in self.text_model.parameters():
                p.requires_grad = False
            if self.text_projection is not None:
                for p in self.text_projection.parameters():
                    p.requires_grad = False

        # Apply LoRA if requested
        if lora_r > 0:
            self._apply_lora(lora_r)

    def _apply_lora(self, r):
        """Apply manual LoRA to attention Q and V projections."""
        self.lora_layers = nn.ModuleList()

        if self.model_type == 'clip':
            layers = self.text_model.encoder.layers
            for layer in layers:
                attn = layer.self_attn
                dim = attn.q_proj.in_features
                lora_q = LoRALayer(dim, dim, r)
                lora_v = LoRALayer(dim, dim, r)
                self.lora_layers.append(lora_q)
                self.lora_layers.append(lora_v)
                # Wrap the original projections
                attn._original_q_proj = attn.q_proj
                attn._original_v_proj = attn.v_proj
                attn.q_proj = LoRAWrappedLinear(attn.q_proj, lora_q)
                attn.v_proj = LoRAWrappedLinear(attn.v_proj, lora_v)
        elif self.model_type == 'gpt2':
            for block in self.text_model.h:
                attn = block.attn
                # GPT-2 uses c_attn (combined QKV projection)
                # We apply LoRA to the full c_attn and let it learn
                dim_in = attn.c_attn.weight.shape[0]  # 768
                dim_out = attn.c_attn.weight.shape[1]  # 2304 (3*768)
                lora = LoRALayer(dim_in, dim_out, r)
                self.lora_layers.append(lora)
                attn._original_c_attn = attn.c_attn
                attn.c_attn = LoRAWrappedLinear(attn.c_attn, lora)

    def forward(self, text_list):
        """
        Args:
            text_list: list of strings
        Returns:
            text_embeds: (B, output_dim) text embeddings
        """
        device = next(self.text_model.parameters()).device
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                return_tensors='pt').to(device)

        if self.model_type == 'clip':
            outputs = self.text_model(**inputs)
            pooled = outputs.pooler_output  # (B, 512) EOS token repr
            if self.text_projection is not None:
                pooled = self.text_projection(pooled)
            return pooled
        elif self.model_type == 'gpt2':
            outputs = self.text_model(**inputs)
            # Use last non-padding token (GPT-2 is causal)
            seq_lens = inputs['attention_mask'].sum(dim=-1) - 1
            batch_idx = torch.arange(len(text_list), device=device)
            pooled = outputs.last_hidden_state[batch_idx, seq_lens]
            return pooled


class LoRALayer(nn.Module):
    """Low-rank adaptation: W' = W + BA where B:(d_out, r), A:(r, d_in)."""

    def __init__(self, in_features, out_features, r=8, alpha=None):
        super().__init__()
        alpha = alpha or r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        return (x @ self.A.T @ self.B.T) * self.scale


class LoRAWrappedLinear(nn.Module):
    """Wraps a frozen linear layer with a LoRA adapter."""

    def __init__(self, original, lora):
        super().__init__()
        self.original = original
        self.lora = lora
        # Keep original frozen
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora(x)

    # Forward attribute access to original for compatibility
    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features if hasattr(self.original, 'in_features') else self.original.weight.shape[0]


# ============================================================================
# ActionLanguageCLIP: full model
# ============================================================================

class ActionLanguageCLIP(nn.Module):
    """
    CLIP-style contrastive model for action-language alignment.

    Action branch: actions → TinyVQVLA → quantized tokens → ActionTransformer → project
    Text branch: instruction → TextEncoder → project
    Loss: recon + vq + lambda * InfoNCE
    """

    def __init__(self, vqvla, action_transformer, text_encoder,
                 proj_dim=128, clip_lambda=1.0):
        super().__init__()
        self.vqvla = vqvla
        self.action_transformer = action_transformer
        self.text_encoder = text_encoder
        self.clip_lambda = clip_lambda

        # Projection heads
        action_input_dim = action_transformer.norm.normalized_shape[0]
        self.action_proj = nn.Linear(action_input_dim, proj_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, proj_dim)

        # Learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=20.0)

    def encode_actions(self, windows, n_windows):
        """Encode action trajectory to a single embedding.

        Args:
            windows: (B, max_windows, T, 7) padded action windows
            n_windows: (B,) number of real windows per trajectory
        Returns:
            action_emb: (B, proj_dim) L2-normalized action embedding
            recon_loss: scalar reconstruction loss
            vq_loss: scalar VQ loss
        """
        B, max_w, T, D = windows.shape
        device = windows.device

        # Flatten all windows, encode, quantize
        all_windows = windows.reshape(B * max_w, T, D)
        recon, vq_loss, z_q, indices = self.vqvla(all_windows)

        # Reconstruction loss (only on real windows)
        real_mask = torch.arange(max_w, device=device).unsqueeze(0) < n_windows.unsqueeze(1)
        real_mask_flat = real_mask.reshape(B * max_w)

        recon_loss = F.mse_loss(
            recon[real_mask_flat], all_windows[real_mask_flat],
            reduction='mean')

        # VQ loss is already averaged over all windows; scale by real fraction
        real_frac = real_mask_flat.float().mean()
        vq_loss = vq_loss * real_frac

        # Reshape z_q back to (B, max_windows, latent_dim)
        z_q = z_q.reshape(B, max_w, -1)

        # Build padding mask for transformer
        pad_mask = ~real_mask  # True = padded

        # Action transformer: (B, max_w, latent_dim) → (B, d_model)
        cls_out = self.action_transformer(z_q, mask=pad_mask)

        # Project and normalize
        action_emb = self.action_proj(cls_out)
        action_emb = F.normalize(action_emb, dim=-1)

        return action_emb, recon_loss, vq_loss

    def encode_text(self, text_list):
        """Encode instructions to embeddings.

        Args:
            text_list: list of strings
        Returns:
            text_emb: (B, proj_dim) L2-normalized text embedding
        """
        with torch.set_grad_enabled(self.text_encoder.lora_r > 0):
            text_features = self.text_encoder(text_list)
        text_emb = self.text_proj(text_features)
        text_emb = F.normalize(text_emb, dim=-1)
        return text_emb

    def contrastive_loss(self, action_emb, text_emb, text_list):
        """Supervised contrastive loss with false-negative masking.

        Episodes sharing the same instruction are treated as positives,
        not negatives. Without this, ~13 episodes per instruction means
        frequent false negatives in each batch.

        Args:
            action_emb: (B, D) L2-normalized
            text_emb: (B, D) L2-normalized
            text_list: list of B instruction strings
        Returns:
            loss: scalar
        """
        B = len(action_emb)
        device = action_emb.device

        # Cosine similarity matrix scaled by temperature
        logits = (action_emb @ text_emb.T) * self.temperature  # (B, B)

        # Build positive mask: pos_mask[i, j] = True if instruction i == j
        # For action→text: each row i should have high similarity to all
        # columns j where text_list[j] == text_list[i]
        pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
        for i in range(B):
            for j in range(B):
                if text_list[i] == text_list[j]:
                    pos_mask[i, j] = True

        # Supervised contrastive: for each anchor, average over all positives
        # log(sum_pos exp(s) / sum_all exp(s))
        # Mask out self from negatives but keep all positives
        neg_mask = ~pos_mask  # (B, B)

        # For numerical stability
        logits_max = logits.max(dim=1, keepdim=True).values.detach()
        logits = logits - logits_max

        exp_logits = logits.exp()
        # Denominator: sum over all (positives + negatives)
        denom = exp_logits.sum(dim=1, keepdim=True)  # (B, 1)

        # Log prob for each positive pair
        log_prob = logits - denom.log()  # (B, B)

        # Average log prob over positive pairs for each anchor
        # action → text direction
        n_pos = pos_mask.float().sum(dim=1)  # (B,)
        loss_a2t = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1)
        loss_a2t = loss_a2t.mean()

        # text → action direction (transpose)
        logits_t = (text_emb @ action_emb.T) * self.temperature
        logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values.detach()
        exp_logits_t = logits_t.exp()
        denom_t = exp_logits_t.sum(dim=1, keepdim=True)
        log_prob_t = logits_t - denom_t.log()
        # pos_mask.T has same structure (symmetric for same instructions)
        n_pos_t = pos_mask.T.float().sum(dim=1)
        loss_t2a = -(log_prob_t * pos_mask.T.float()).sum(dim=1) / n_pos_t.clamp(min=1)
        loss_t2a = loss_t2a.mean()

        return (loss_a2t + loss_t2a) / 2

    def forward(self, windows, n_windows, text_list):
        """Full forward pass with combined loss.

        Returns:
            total_loss, loss_dict
        """
        action_emb, recon_loss, vq_loss = self.encode_actions(windows, n_windows)
        text_emb = self.encode_text(text_list)
        clip_loss = self.contrastive_loss(action_emb, text_emb, text_list)

        total_loss = recon_loss + vq_loss + self.clip_lambda * clip_loss

        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'vq': vq_loss.item(),
            'clip': clip_loss.item(),
            'temp': self.temperature.item(),
        }
        return total_loss, loss_dict


# ============================================================================
# Dataset
# ============================================================================

def load_calvin_raw(data_dir):
    """Load CALVIN annotations WITHOUT verb filtering."""
    lang_path = os.path.join(data_dir, 'lang_annotations', 'auto_lang_ann.npy')
    if not os.path.exists(lang_path):
        raise FileNotFoundError(f"Annotations not found at {lang_path}")

    lang_data = np.load(lang_path, allow_pickle=True).item()
    instructions = lang_data['language']['ann']
    indices = lang_data['info']['indx']

    df = pd.DataFrame({
        'start_idx': [idx[0] for idx in indices],
        'end_idx': [idx[1] for idx in indices],
        'instruction': instructions
    })
    return df


class CalvinCLIPDataset(Dataset):
    """Dataset returning (action_windows, instruction, n_windows) tuples.

    Preloads all action trajectories into RAM at init to avoid per-sample
    disk I/O bottleneck (each episode spans ~60 .npz files).
    """

    def __init__(self, df, data_dir, action_window_size=5, max_windows=16):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.action_window_size = action_window_size
        self.max_windows = max_windows

        # Preload trajectories using a cached .npy file for fast startup.
        # First run builds the cache (~10 min); subsequent runs load in <1s.
        cache_path = os.path.join(data_dir, '_action_cache.npz')
        all_starts = self.df['start_idx'].values.astype(int)
        all_ends = self.df['end_idx'].values.astype(int)

        if os.path.exists(cache_path):
            print(f"  Loading action cache from {cache_path}...")
            cache = np.load(cache_path)
            offset = int(cache['offset'])
            all_actions = cache['actions']
        else:
            # Build cache: load all timesteps in the range
            needed = set()
            for s, e in zip(all_starts, all_ends):
                needed.update(range(s, e + 1))
            needed = sorted(needed)
            offset = needed[0]
            size = needed[-1] - offset + 1
            print(f"  Building action cache: {len(needed)} timesteps "
                  f"({offset}-{needed[-1]}) from {data_dir}...")
            all_actions = np.zeros((size, 7), dtype=np.float32)
            for j in needed:
                path = os.path.join(data_dir, EPISODE_TEMPLATE.format(j))
                data = np.load(path, mmap_mode='r')
                all_actions[j - offset] = data[ACTION_KEY]
            np.savez_compressed(cache_path, actions=all_actions,
                                offset=np.array(offset))
            print(f"  Saved cache to {cache_path}")

        # Slice per episode
        self.trajectories = []
        for i in range(len(self.df)):
            s = all_starts[i] - offset
            e = all_ends[i] - offset + 1
            self.trajectories.append(all_actions[s:e].copy())
        del all_actions
        print(f"  Done. Loaded {len(self.trajectories)} trajectories.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        actions = self.trajectories[idx]
        instruction = self.df.iloc[idx]['instruction']

        # Split into non-overlapping windows
        T = actions.shape[0]
        ws = self.action_window_size
        n_windows = T // ws
        if n_windows == 0:
            padded = np.zeros((ws, actions.shape[1]), dtype=np.float32)
            padded[:T] = actions
            windows = padded[np.newaxis]
            n_windows = 1
        else:
            windows = actions[:n_windows * ws].reshape(n_windows, ws, -1)

        if n_windows > self.max_windows:
            windows = windows[:self.max_windows]
            n_windows = self.max_windows

        padded_windows = np.zeros(
            (self.max_windows, ws, actions.shape[1]), dtype=np.float32)
        padded_windows[:n_windows] = windows

        return (torch.from_numpy(padded_windows),
                instruction,
                torch.tensor(n_windows, dtype=torch.long))


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_losses = {}
    n_batches = 0

    for windows, instructions, n_windows in loader:
        windows = windows.to(device)
        n_windows = n_windows.to(device)

        loss, loss_dict = model(windows, n_windows, list(instructions))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_losses = {}
    n_batches = 0

    for windows, instructions, n_windows in loader:
        windows = windows.to(device)
        n_windows = n_windows.to(device)

        _, loss_dict = model(windows, n_windows, list(instructions))

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def fit_clip_tokenizer(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (no verb filtering)
    print("Loading CALVIN data (no verb filtering)...")
    train_df = load_calvin_raw(args.train_dir)
    val_df = load_calvin_raw(args.val_dir)
    print(f"Train: {len(train_df)} episodes, Val: {len(val_df)} episodes")
    print(f"Unique instructions — train: {train_df['instruction'].nunique()}, "
          f"val: {val_df['instruction'].nunique()}")

    train_dataset = CalvinCLIPDataset(
        train_df, args.train_dir,
        action_window_size=args.action_window_size,
        max_windows=args.max_windows)
    val_dataset = CalvinCLIPDataset(
        val_df, args.val_dir,
        action_window_size=args.action_window_size,
        max_windows=args.max_windows)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    print("Building model...")
    vqvla = TinyVQVLA(
        action_dim=7,
        action_window_size=args.action_window_size,
        latent_dim=args.latent_dim,
        block_channels=tuple(args.block_channels),
        blocks_per_stage=args.blocks_per_stage,
        num_mid_blocks=args.num_mid_blocks,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size)

    action_transformer = ActionTransformer(
        input_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.transformer_layers,
        dropout=args.dropout,
        max_len=args.max_windows)

    text_encoder = TextEncoderWrapper(
        model_name=args.text_model,
        model_type=args.text_type,
        freeze=(args.lora_r == 0),
        lora_r=args.lora_r)

    model = ActionLanguageCLIP(
        vqvla=vqvla,
        action_transformer=action_transformer,
        text_encoder=text_encoder,
        proj_dim=args.proj_dim,
        clip_lambda=args.clip_lambda).to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   grad_clip=args.grad_clip)
        val_losses = eval_epoch(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        if epoch % args.log_every == 0 or epoch == 1:
            print("Epoch {:3d}/{} ({:.1f}s) | "
                  "Train: total={:.4f} recon={:.4f} vq={:.4f} clip={:.4f} | "
                  "Val: total={:.4f} recon={:.4f} vq={:.4f} clip={:.4f} | "
                  "temp={:.3f}".format(
                      epoch, args.epochs, dt,
                      train_losses['total'], train_losses['recon'],
                      train_losses['vq'], train_losses['clip'],
                      val_losses['total'], val_losses['recon'],
                      val_losses['vq'], val_losses['clip'],
                      train_losses['temp']))

        # Save best + early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch
            patience_counter = 0
            save_path = os.path.join(args.save_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v for k, v in model.state_dict().items()
                                     if not k.startswith('text_encoder.text_model.')
                                     or 'lora' in k},
                'vqvla_state_dict': model.vqvla.state_dict(),
                'val_losses': val_losses,
                'train_losses': train_losses,
                'args': vars(args),
            }, save_path)
            print(f"  -> Saved best model (val_total={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs, "
                      f"best was epoch {best_epoch})")
                break

    print(f"Training complete. Best val total loss: {best_val_loss:.4f} "
          f"at epoch {best_epoch}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CLIP action-language tokenizer")

    # Data
    p.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    p.add_argument('--val_dir', type=str, default=VAL_DIR)
    p.add_argument('--save_dir', type=str, default='checkpoints/clip_tokenizer')

    # VQ-VAE
    p.add_argument('--action_window_size', type=int, default=5)
    p.add_argument('--max_windows', type=int, default=16)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--block_channels', type=int, nargs='+', default=[32, 64, 128])
    p.add_argument('--blocks_per_stage', type=int, default=2)
    p.add_argument('--num_mid_blocks', type=int, default=2)
    p.add_argument('--num_quantizers', type=int, default=2)
    p.add_argument('--codebook_size', type=int, default=256)

    # Action transformer
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--transformer_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)

    # Text encoder
    p.add_argument('--text_model', type=str,
                   default='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    p.add_argument('--text_type', type=str, default='clip',
                   choices=['clip', 'gpt2'])
    p.add_argument('--lora_r', type=int, default=0,
                   help='LoRA rank (0=frozen)')

    # Projection
    p.add_argument('--proj_dim', type=int, default=128)

    # Training
    p.add_argument('--clip_lambda', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--patience', type=int, default=30,
                   help='Early stopping patience (epochs without improvement)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fit_clip_tokenizer(args)
