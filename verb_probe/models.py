"""Verb classification models for the probe.

MotionVerbClassifier: wraps VerbHead from tokenization/aux_heads.py.
GoalVerbClassifier: vision-only verb classifier (frozen ViT patches).
"""

import torch
import torch.nn as nn

from config import (
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS,
    DROPOUT_RATE, PATCH_SIZE, IMAGE_SIZE,
)
from tokenization.aux_heads import VerbHead
from verb_probe.image_encoders import build_image_encoder


class MotionVerbClassifier(nn.Module):
    """Motion-only verb classifier wrapping VerbHead.

    Handles the verb probe batch format (frames, actions, scene_vec, label, seq_len)
    and different action representations before delegating to VerbHead.

    Supports three input types:
      - action_rep="native":  raw continuous actions (B, T, action_dim)
      - action_rep="token_id": discrete token IDs (B, T) -> Embedding -> d_model
      - action_rep="latent":  continuous latent vectors (B, T, latent_dim)
    """

    def __init__(self, num_verbs, action_rep="native", action_dim=ACTION_DIM,
                 action_vocab_size=None, latent_dim=None,
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                 dropout=DROPOUT_RATE):
        super().__init__()
        self.action_rep = action_rep

        if action_rep == "token_id":
            assert action_vocab_size is not None
            self.token_embed = nn.Embedding(action_vocab_size, d_model)
            input_dim = d_model
        elif action_rep == "latent":
            assert latent_dim is not None
            self.token_embed = None
            input_dim = latent_dim
        else:  # native
            self.token_embed = None
            input_dim = action_dim

        self.verb_head = VerbHead(
            latent_dim=input_dim, num_verbs=num_verbs,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dropout=dropout)

    def forward(self, frames, actions, seq_lengths=None, scene_vec=None):
        """Forward pass. Ignores frames and scene_vec (motion-only).

        Args:
            frames: unused (kept for batch-format compatibility)
            actions: (B, T, action_dim) for native/latent, (B, T) for token_id
            seq_lengths: (B,) real sequence lengths (for padding mask)
            scene_vec: unused
        """
        if self.token_embed is not None:
            tokens = self.token_embed(actions)
        else:
            tokens = actions

        n_valid = seq_lengths if seq_lengths is not None else \
            torch.full((tokens.size(0),), tokens.size(1),
                       device=tokens.device, dtype=torch.long)

        return self.verb_head(tokens, n_valid)


class GoalVerbClassifier(nn.Module):
    """Vision-only verb classifier: image patches -> Transformer -> classify.

    Uses a frozen ViT encoder (DINOv2, VC-1) to extract patch tokens,
    prepends [CLS], and classifies from the CLS output.
    """

    def __init__(self, num_verbs, image_encoder="dinov2_s",
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                 dropout=DROPOUT_RATE, img_size=IMAGE_SIZE[0],
                 patch_size=PATCH_SIZE, num_frames=2, delta_patches=0):
        super().__init__()
        self.num_frames = num_frames
        self.delta_patches = delta_patches

        # Vision encoder
        self.patch_embed = build_image_encoder(image_encoder, d_model, img_size, patch_size)
        full_num_patches = self.patch_embed.num_tokens
        self.num_patches = delta_patches if delta_patches > 0 else full_num_patches

        # Positional encodings
        self.patch_pos = nn.Parameter(torch.zeros(1, full_num_patches, d_model))
        n_temporal = max(num_frames - 1, 1) if delta_patches > 0 else num_frames
        self.frame_pos = nn.Parameter(torch.zeros(1, n_temporal, 1, d_model))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        for name, p in self.named_parameters():
            if 'pos' in name or 'cls_token' in name:
                nn.init.trunc_normal_(p, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            activation="gelu", dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs))

    def forward(self, frames, actions=None, seq_lengths=None, scene_vec=None):
        """Forward pass. Ignores actions and scene_vec (vision-only).

        Args:
            frames: (B, num_frames, C, H, W) image frames
            actions: unused
            seq_lengths: unused
            scene_vec: unused
        """
        B = frames.size(0)
        nf = frames.size(1)
        parts = [self.cls_token.expand(B, -1, -1)]

        if self.delta_patches > 0:
            K = self.delta_patches
            d = self.cls_token.size(-1)
            all_patches = [self.patch_embed(frames[:, fi]) for fi in range(nf)]
            for pi in range(nf - 1):
                diff = all_patches[pi + 1] - all_patches[pi]
                mag = diff.norm(dim=-1)
                topk_idx = mag.topk(K, dim=-1).indices
                idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, d)
                selected = torch.gather(diff, 1, idx_exp)
                pos = torch.gather(
                    self.patch_pos.expand(B, -1, -1), 1, idx_exp)
                selected = selected + pos + self.frame_pos[:, pi]
                parts.append(selected)
        else:
            for fi in range(nf):
                patches = self.patch_embed(frames[:, fi])
                patches = patches + self.patch_pos + self.frame_pos[:, fi]
                parts.append(patches)

        seq = torch.cat(parts, dim=1)
        x = self.encoder(seq)
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)
