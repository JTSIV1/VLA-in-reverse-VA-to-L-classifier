import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import load_calvin_to_dataframe
from config import (
    DATA_DIR, VAL_DIR, IMAGE_KEY, ACTION_KEY, EPISODE_TEMPLATE,
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS, DROPOUT_RATE, PATCH_SIZE,
    IMAGE_SIZE, IMG_MEAN, IMG_STD, R3M_IMG_SIZE, R3M_VARIANT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN, NUM_WORKERS,
    WARMUP_EPOCHS, GRAD_CLIP_NORM, FAST_VOCAB_SIZE, FAST_TOKENIZER_PATH,
    VQVAE_TOKENIZER_PATH, VQVAE_VOCAB_SIZE,
)


class PatchEmbed(nn.Module):
    """ViT-style patch embedding: split image into non-overlapping patches and
    linearly project each to d_model via Conv2d (equivalent to flatten + linear,
    but fused into a single GPU op)."""

    def __init__(self, img_size=200, patch_size=25, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class R3MEncoder(nn.Module):
    """R3M pretrained vision encoder: ResNet → global feature → project to d_model.
    Each image becomes a single token (not patch tokens)."""

    def __init__(self, d_model, variant=R3M_VARIANT, freeze=True):
        super().__init__()
        from r3m import load_r3m
        r3m_model = load_r3m(variant)
        self.r3m = r3m_model.module  # unwrap DataParallel
        self.freeze = freeze
        if freeze:
            self.r3m.eval()
            for p in self.r3m.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.r3m.outdim, d_model)
        self.num_patches = 1  # global feature, not patches

    def forward(self, x):
        # x: (B, 3, 224, 224) -> (B, 1, d_model)
        if self.freeze:
            with torch.no_grad():
                features = self.r3m(x)  # (B, outdim)
        else:
            features = self.r3m(x)  # (B, outdim) — gradients flow through
        return self.proj(features).unsqueeze(1)  # (B, 1, d_model)


class ViTEncoder(nn.Module):
    """Frozen ViT patch encoder using DINOv2 or VC-1 via timm.
    Extracts patch tokens, spatially pools to pool_size×pool_size,
    and projects to d_model."""

    # VC-1 state dict key remapping: vc1 uses different names than timm
    VC1_KEY_MAP = {
        'cls_token': 'cls_token',
        'pos_embed': 'pos_embed',
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    }

    def __init__(self, d_model, variant="dinov2_s", pool_size=7):
        super().__init__()
        import timm
        self.pool_size = pool_size

        if variant == "dinov2_s":
            self.vit = timm.create_model('vit_small_patch14_dinov2',
                                          pretrained=True, num_classes=0,
                                          img_size=224)
            embed_dim = 384
            self.grid_size = 16  # 224 / 14
        elif variant == "dinov2_b":
            self.vit = timm.create_model('vit_base_patch14_dinov2',
                                          pretrained=True, num_classes=0,
                                          img_size=224)
            embed_dim = 768
            self.grid_size = 16
        elif variant == "vc1":
            # VC-1 base is ViT-B/16 — load timm skeleton, fill with HF weights
            self.vit = timm.create_model('vit_base_patch16_224',
                                          pretrained=False, num_classes=0)
            embed_dim = 768
            self.grid_size = 14  # 224 / 16
            self._load_vc1_weights()
        else:
            raise ValueError(f"Unknown ViT variant: {variant}")

        # Freeze all backbone parameters
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        # Spatial pooling: native grid -> pool_size × pool_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # Project from ViT embed_dim to our d_model
        self.proj = nn.Linear(embed_dim, d_model)
        self.num_patches = pool_size * pool_size

    def _load_vc1_weights(self):
        """Load VC-1 pretrained weights from HuggingFace into timm ViT."""
        from huggingface_hub import hf_hub_download
        import torch as _torch
        path = hf_hub_download('facebook/vc1-base', 'pytorch_model.bin')
        vc1_state = _torch.load(path, map_location='cpu', weights_only=True)['model']

        # Remap VC-1 keys to timm convention
        timm_state = {}
        for k, v in vc1_state.items():
            new_key = k
            # VC-1 uses 'blocks.N.attn' -> timm uses 'blocks.N.attn'  (same)
            # VC-1 uses 'blocks.N.norm1' -> timm uses 'blocks.N.norm1' (same)
            # VC-1 uses 'blocks.N.mlp.fc1' -> timm uses 'blocks.N.mlp.fc1' (same)
            # Main difference: VC-1 may have 'norm.' for final norm
            if k == 'norm.weight':
                new_key = 'norm.weight'
            elif k == 'norm.bias':
                new_key = 'norm.bias'
            timm_state[new_key] = v

        # Load with strict=False to skip head/decoder keys
        missing, unexpected = self.vit.load_state_dict(timm_state, strict=False)
        n_loaded = len(timm_state) - len(unexpected)
        print(f"VC-1: loaded {n_loaded} params, {len(missing)} missing, "
              f"{len(unexpected)} unexpected")

    def forward(self, x):
        # x: (B, 3, 224, 224)
        with torch.no_grad():
            features = self.vit.forward_features(x)  # (B, 1+N, embed_dim)
        patches = features[:, 1:, :]  # drop CLS token -> (B, N, embed_dim)

        # Reshape to spatial grid, pool, reshape back
        B, N, D = patches.shape
        h = w = self.grid_size
        patches = patches.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)
        patches = self.pool(patches)  # (B, D, pool_size, pool_size)
        patches = patches.flatten(2).transpose(1, 2)  # (B, pool_size^2, D)

        return self.proj(patches)  # (B, num_patches, d_model)


class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, action_dim=ACTION_DIM,
                 dropout=DROPOUT_RATE, img_size=IMAGE_SIZE[0],
                 patch_size=PATCH_SIZE, max_action_len=MAX_SEQ_LEN,
                 modality="full", action_rep="native",
                 fast_vocab_size=FAST_VOCAB_SIZE,
                 cross_layers=CROSS_LAYERS,
                 vision_encoder="patch", freeze_vision=True,
                 num_frames=2, delta_patches=0,
                 modal_dropout=0.0, aux_loss_weight=0.0):
        super().__init__()
        self.modality = modality
        self.action_rep = action_rep
        self.num_layers = num_layers
        self.cross_layers = cross_layers
        self.vision_encoder_type = vision_encoder
        self.num_frames = num_frames
        self.delta_patches = delta_patches  # 0 = use all patches; >0 = top-K changed
        self.modal_dropout = modal_dropout
        self.aux_loss_weight = aux_loss_weight

        # -- Vision branch (skip for action_only) --
        if modality != "action_only":
            if vision_encoder == "r3m":
                self.vision_enc = R3MEncoder(d_model, freeze=freeze_vision)
                self.num_patches = self.vision_enc.num_patches  # 1 per image
            elif vision_encoder in ("dinov2_s", "dinov2_b", "vc1"):
                self.vision_enc = ViTEncoder(d_model, variant=vision_encoder)
                if delta_patches > 0:
                    # Delta mode: top-K changed patches per frame pair
                    self.num_patches = delta_patches
                else:
                    self.num_patches = self.vision_enc.num_patches  # 49 (7x7 pooled)
            else:
                self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
                self.num_patches = (img_size // patch_size) ** 2
            # Spatial position: need full grid size for delta gather
            full_num_patches = (self.vision_enc.num_patches
                                if hasattr(self, 'vision_enc') else self.num_patches)
            self.patch_pos = nn.Parameter(
                torch.zeros(1, full_num_patches, d_model))
            # Temporal position per frame (or per frame-pair for delta mode)
            n_temporal = max(num_frames - 1, 1) if delta_patches > 0 else num_frames
            self.frame_pos = nn.Parameter(torch.zeros(1, n_temporal, 1, d_model))
            # Token type for vision
            self.type_img = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.num_patches = (img_size // patch_size) ** 2

        # -- Action branch (skip for vision_only) --
        if modality != "vision_only":
            if action_rep == "native":
                self.action_proj = nn.Linear(action_dim, d_model)
            else:
                # FAST: discrete token IDs -> learned embeddings
                self.action_embed = nn.Embedding(fast_vocab_size, d_model)
            # Temporal position for action sequence
            self.action_pos = nn.Parameter(torch.zeros(1, max_action_len, d_model))
            self.type_action = nn.Parameter(torch.zeros(1, 1, d_model))

        # -- CLS token --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Separate positional embeddings per modality so spatial patch
        # positions and temporal action positions are independently learned,
        # rather than sharing a single flat 1D encoding
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        # Token type: like BERT segment embeddings
        self.type_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # Initialize all learned embeddings (ViT convention)
        for name, p in self.named_parameters():
            if 'type_' in name or '_pos' in name or 'cls_token' in name:
                nn.init.trunc_normal_(p, std=0.02)

        # -- Transformer encoder (ModuleList for per-layer attention masks) --
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
            )
            for _ in range(num_layers)
        ])

        # -- Classification head --
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs)
        )

        # -- Auxiliary unimodal heads (full modality only; used via forward_with_aux) --
        if aux_loss_weight > 0.0 and modality == "full":
            self.aux_vision_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_verbs)
            )
            self.aux_action_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_verbs)
            )
        else:
            self.aux_vision_head = None
            self.aux_action_head = None

    def _forward_core(self, frames, trajectories, seq_lengths=None, training=False):
        """Shared forward logic. Returns (x_final, v_end, x_transition).
        v_end: index of first action token (= 1 + n_vis_tokens).
        x_transition: x snapshot at the self→cross layer boundary (for aux heads),
                      or None if aux_loss_weight == 0 or modality != 'full'."""
        batch_size = trajectories.size(0)

        # CLS token + position + type
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos + self.type_cls
        parts = [cls]

        # Vision: encode each frame, add spatial + temporal position + type
        nf = 0
        if self.modality != "action_only":
            # frames: (B, num_frames, C, H, W)
            nf = frames.size(1)
            if self.delta_patches > 0:
                # Delta mode: extract patches for all frames, then compute diffs
                all_patches = []
                for fi in range(nf):
                    all_patches.append(self.vision_enc(frames[:, fi]))  # (B, P, d)
                K = self.delta_patches
                d = all_patches[0].size(-1)
                for pi in range(nf - 1):
                    diff = all_patches[pi + 1] - all_patches[pi]  # (B, P, d)
                    mag = diff.norm(dim=-1)  # (B, P)
                    topk_idx = mag.topk(K, dim=-1).indices  # (B, K)
                    # Gather top-K diff vectors
                    idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, K, d)
                    selected = torch.gather(diff, 1, idx_exp)  # (B, K, d)
                    # Gather spatial positions for the selected patches
                    pos = torch.gather(
                        self.patch_pos.expand(batch_size, -1, -1), 1, idx_exp)
                    selected = selected + pos + self.frame_pos[:, pi] + self.type_img
                    parts.append(selected)
            else:
                for fi in range(nf):
                    frame_i = frames[:, fi]  # (B, C, H, W)
                    if self.vision_encoder_type in ("r3m", "dinov2_s", "dinov2_b", "vc1"):
                        patches = self.vision_enc(frame_i)
                    else:
                        patches = self.patch_embed(frame_i)
                    # spatial pos + temporal pos for this frame + vision type
                    patches = patches + self.patch_pos + self.frame_pos[:, fi] + self.type_img
                    parts.append(patches)

        # Action embeddings + temporal position + type
        if self.modality != "vision_only":
            action_len = trajectories.size(1)
            if self.action_rep == "native":
                action_emb = self.action_proj(trajectories)
            else:
                action_emb = self.action_embed(trajectories)
            action_emb = action_emb + self.action_pos[:, :action_len, :] + self.type_action
            parts.append(action_emb)

        # Sequence layout: [CLS] [vision tokens] [action tokens]
        full_seq = torch.cat(parts, dim=1)
        total_len = full_seq.size(1)

        # Compute v_end: index of first action token (= 1 + n_vis_tokens)
        if self.modality == "action_only":
            n_vis_tokens = 0
        elif self.delta_patches > 0:
            n_vis_tokens = max(nf - 1, 1) * self.num_patches
        else:
            n_vis_tokens = nf * self.num_patches
        v_start, v_end = 1, 1 + n_vis_tokens

        # Modal dropout: batch-level, exclusive, training only
        if training and self.modal_dropout > 0.0 and self.modality == "full":
            r = torch.rand(1).item()
            if r < self.modal_dropout:
                # Zero out vision tokens → force model to use action only
                full_seq = full_seq.clone()
                full_seq[:, v_start:v_end] = 0.0
            elif r < 2 * self.modal_dropout:
                # Zero out action tokens → force model to use vision only
                full_seq = full_seq.clone()
                full_seq[:, v_end:] = 0.0
            # else: keep both (probability = 1 - 2*modal_dropout)

        # Padding mask: True = padded position to ignore
        src_key_padding_mask = None
        if seq_lengths is not None:
            positions = torch.arange(total_len, device=full_seq.device).unsqueeze(0)
            src_key_padding_mask = positions >= seq_lengths.unsqueeze(1)

        # Build block-diagonal self-only mask for early layers (full modality only)
        self_mask = None
        num_self_layers = self.num_layers - self.cross_layers
        if num_self_layers > 0 and self.modality == "full":
            # Additive mask: -inf blocks attention, 0.0 allows it
            self_mask = torch.full((total_len, total_len), float('-inf'),
                                   device=full_seq.device)
            self_mask[0, 0] = 0.0                                    # CLS → self only
            self_mask[v_start:v_end, v_start:v_end] = 0.0           # vision block
            self_mask[v_end:, v_end:] = 0.0                         # action block

        # Forward through layers; capture x at self→cross boundary for aux heads
        x_transition = None
        x = full_seq
        for i, layer in enumerate(self.layers):
            if (i == num_self_layers
                    and self.aux_loss_weight > 0.0
                    and self.modality == "full"):
                x_transition = x  # snapshot before first cross-modal layer
            mask = self_mask if i < num_self_layers else None
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # Edge case: all cross-modal (num_self_layers == 0) → transition at input
        if (self.aux_loss_weight > 0.0
                and self.modality == "full"
                and x_transition is None):
            x_transition = full_seq

        return x, v_end, x_transition

    def forward(self, frames, trajectories, seq_lengths=None):
        """Standard forward pass. Returns main logits only (eval-compatible)."""
        x, _v_end, _xt = self._forward_core(
            frames, trajectories, seq_lengths=seq_lengths,
            training=self.training)
        return self.classifier(x[:, 0, :])

    def forward_with_aux(self, frames, trajectories, seq_lengths=None):
        """Training forward pass. Returns (main_logits, aux_v_logits, aux_a_logits).
        Aux logits are None when aux_loss_weight == 0 or modality != 'full'."""
        x, v_end, x_transition = self._forward_core(
            frames, trajectories, seq_lengths=seq_lengths,
            training=self.training)
        main_logits = self.classifier(x[:, 0, :])
        aux_v_logits = None
        aux_a_logits = None
        if self.aux_vision_head is not None and x_transition is not None:
            if v_end > 1:
                aux_v_logits = self.aux_vision_head(
                    x_transition[:, 1:v_end].mean(dim=1))
            if v_end < x_transition.size(1):
                aux_a_logits = self.aux_action_head(
                    x_transition[:, v_end:].mean(dim=1))
        return main_logits, aux_v_logits, aux_a_logits


class CalvinVerbDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, max_seq_len=MAX_SEQ_LEN,
                 modality="full", fast_tokenizer=None,
                 vision_encoder="patch", img_size=IMAGE_SIZE[0],
                 num_frames=2, delta_patches=0,
                 vqvae_tokenizer=None, vqvae_chunk_size=4):
        """CALVIN dataset loader with modality ablation support.
        Args:
            modality: "full", "action_only", or "vision_only"
            fast_tokenizer: if provided, tokenize actions into FAST token IDs
            vision_encoder: "patch" or "r3m"
            img_size: image size for the vision encoder
            num_frames: number of frames to sample (2 = first+last, >2 = uniformly spaced)
            delta_patches: if >0, use top-K changed patches between frame pairs
            vqvae_tokenizer: if provided, tokenize actions into VQ-VAE chunk codes
            vqvae_chunk_size: number of timesteps per VQ-VAE chunk
        """
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.modality = modality
        self.fast_tokenizer = fast_tokenizer
        self.vqvae_tokenizer = vqvae_tokenizer
        self.vqvae_chunk_size = vqvae_chunk_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.delta_patches = delta_patches
        if vision_encoder == "r3m":
            self.num_patches = 1  # R3M: global feature per image
        elif vision_encoder in ("dinov2_s", "dinov2_b", "vc1"):
            self.num_patches = 49 if delta_patches == 0 else delta_patches  # 7x7 pooled
        else:
            self.num_patches = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2

        unique_verbs = sorted(list(set(df['primary_verb'].unique())))
        self.verb_to_id = {v: i for i, v in enumerate(unique_verbs)}
        self.id_to_verb = {i: v for v, i in self.verb_to_id.items()}
        print(f"Vocab mapped: {len(self.verb_to_id)} unique verbs found.")

    def __len__(self):
        return len(self.df)

    def _load_npz(self, idx):
        filename = EPISODE_TEMPLATE.format(idx)
        return np.load(os.path.join(self.data_dir, filename), mmap_mode='r')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_idx = row['start_idx']
        end_idx = row['end_idx']
        verb = row['primary_verb']

        # -- Load frames (skip for action_only to save I/O) --
        if self.modality != "action_only":
            # Sample frame indices: uniformly spaced including first and last
            total_steps = end_idx - start_idx + 1
            if self.num_frames == 2:
                frame_indices = [start_idx, end_idx]
            else:
                positions = np.linspace(0, total_steps - 1, self.num_frames, dtype=int)
                frame_indices = [start_idx + p for p in positions]
            frame_list = []
            for fi in frame_indices:
                data = self._load_npz(fi)
                img = Image.fromarray(np.array(data[IMAGE_KEY])).convert("RGB")
                if self.transform:
                    frame_list.append(self.transform(img))
                else:
                    frame_list.append(transforms.ToTensor()(img))
            frames = torch.stack(frame_list)  # (num_frames, C, H, W)
        else:
            # Dummy tensors — not used by model but keeps DataLoader shape consistent
            frames = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)

        # -- Load actions (skip for vision_only) --
        if self.modality != "vision_only":
            all_actions = []
            for i in range(start_idx, end_idx + 1):
                step_data = self._load_npz(i)
                all_actions.append(np.array(step_data[ACTION_KEY]))
            actions = np.array(all_actions)  # (T, 7)
            L = actions.shape[0]

            if self.fast_tokenizer is not None:
                # FAST tokenization: (T, 7) -> list[int] token IDs
                token_ids = self.fast_tokenizer(actions[np.newaxis])
                if isinstance(token_ids, list) and len(token_ids) > 0:
                    if isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                token_ids = list(token_ids)
                L_tok = len(token_ids)
                if L_tok < self.max_seq_len:
                    token_ids = token_ids + [0] * (self.max_seq_len - L_tok)
                else:
                    token_ids = token_ids[:self.max_seq_len]
                actions_tensor = torch.tensor(token_ids, dtype=torch.long)
                action_real_len = min(L_tok, self.max_seq_len)
            elif self.vqvae_tokenizer is not None:
                # VQ-VAE tokenization: (T, 7) -> (T//K,) int64 code indices
                from vqvae_tokenizer import tokenize_trajectory_vqvae
                token_ids = tokenize_trajectory_vqvae(
                    self.vqvae_tokenizer, actions).tolist()
                L_tok = len(token_ids)
                if L_tok < self.max_seq_len:
                    token_ids = token_ids + [0] * (self.max_seq_len - L_tok)
                else:
                    token_ids = token_ids[:self.max_seq_len]
                actions_tensor = torch.tensor(token_ids, dtype=torch.long)
                action_real_len = min(L_tok, self.max_seq_len)
            else:
                # Native continuous actions
                if L < self.max_seq_len:
                    actions_padded = np.pad(actions, ((0, self.max_seq_len - L), (0, 0)),
                                            mode='constant')
                else:
                    actions_padded = actions[:self.max_seq_len]
                actions_tensor = torch.tensor(actions_padded, dtype=torch.float32)
                action_real_len = min(L, self.max_seq_len)
        else:
            # Vision-only: dummy action tensor, no variable-length actions
            actions_tensor = torch.zeros(self.max_seq_len, ACTION_DIM)
            action_real_len = 0

        label_id = self.verb_to_id.get(verb, self.verb_to_id.get("unknown", 0))
        label = torch.tensor(label_id, dtype=torch.long)

        # Compute actual sequence length for padding mask
        seq_len = 1  # CLS
        if self.modality != "action_only":
            if self.delta_patches > 0:
                # Delta mode: (num_frames - 1) frame pairs × K patches each
                seq_len += max(self.num_frames - 1, 1) * self.delta_patches
            else:
                seq_len += self.num_frames * self.num_patches
        if self.modality != "vision_only":
            seq_len += action_real_len

        return frames, actions_tensor, label, seq_len


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Modality: {args.modality} | Action rep: {args.action_rep} | "
          f"Vision encoder: {args.vision_encoder} | "
          f"Cross layers: {args.cross_layers}/{NUM_LAYERS}")

    # Image size depends on vision encoder
    if args.vision_encoder in ("r3m", "dinov2_s", "dinov2_b", "vc1"):
        img_size = 224
    else:
        img_size = IMAGE_SIZE[0]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # --- Load FAST tokenizer if needed ---
    fast_tok = None
    vqvae_tok = None
    fast_vocab_size = FAST_VOCAB_SIZE
    if args.action_rep == "fast":
        from fast_tokenizer import load_fast_tokenizer
        fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
        fast_vocab_size = fast_tok.bpe_tokenizer.vocab_size
        print(f"Loaded FAST tokenizer from {args.fast_tokenizer_path} "
              f"(vocab_size={fast_vocab_size})")
    elif args.action_rep == "vq_vae":
        from vqvae_tokenizer import load_vqvae_tokenizer
        vqvae_tok = load_vqvae_tokenizer(args.vqvae_tokenizer_path)
        fast_vocab_size = vqvae_tok.num_codes
        print(f"Loaded VQ-VAE tokenizer from {args.vqvae_tokenizer_path} "
              f"(num_codes={fast_vocab_size}, chunk_size={vqvae_tok.chunk_size})")

    # --- Load dataframes ---
    print(f"Loading training data from {args.data_dir}...")
    df = load_calvin_to_dataframe(args.data_dir)

    print(f"Loading validation data from {args.val_dir}...")
    val_df = load_calvin_to_dataframe(args.val_dir)

    if args.debug:
        n = min(args.debug, len(df))
        df = df.head(n).copy()
        val_df = val_df.head(n).copy()
        args.epochs = min(args.epochs, 2)
        print(f"[DEBUG] Using {n} train samples, {len(val_df)} val samples, "
              f"{args.epochs} epochs")

    # --- Filter sparse classes ---
    if args.min_class_count > 0:
        verb_counts = df['primary_verb'].value_counts()
        keep_verbs = set(verb_counts[verb_counts >= args.min_class_count].index)
        n_before = len(df)
        df = df[df['primary_verb'].isin(keep_verbs)].reset_index(drop=True)
        val_df = val_df[val_df['primary_verb'].isin(keep_verbs)].reset_index(drop=True)
        dropped = verb_counts.index.difference(keep_verbs)
        print(f"Filtered classes with <{args.min_class_count} train samples: "
              f"{len(verb_counts)}→{len(keep_verbs)} classes, "
              f"train {n_before}→{len(df)}, val→{len(val_df)}")
        if len(dropped) > 0:
            print(f"  Dropped: {sorted(dropped.tolist())}")

    # --- Build datasets ---
    delta_patches = getattr(args, 'delta_patches', 0)
    vqvae_chunk_size = getattr(args, 'vqvae_chunk_size', 4)
    dataset = CalvinVerbDataset(df, args.data_dir, transform=transform,
                                max_seq_len=args.max_seq_len,
                                modality=args.modality,
                                fast_tokenizer=fast_tok,
                                vision_encoder=args.vision_encoder,
                                img_size=img_size,
                                num_frames=args.num_frames,
                                delta_patches=delta_patches,
                                vqvae_tokenizer=vqvae_tok,
                                vqvae_chunk_size=vqvae_chunk_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

    val_dataset = CalvinVerbDataset(val_df, args.val_dir, transform=transform,
                                    max_seq_len=args.max_seq_len,
                                    modality=args.modality,
                                    fast_tokenizer=fast_tok,
                                    vision_encoder=args.vision_encoder,
                                    img_size=img_size,
                                    num_frames=args.num_frames,
                                    delta_patches=delta_patches,
                                    vqvae_tokenizer=vqvae_tok,
                                    vqvae_chunk_size=vqvae_chunk_size)
    val_dataset.verb_to_id = dataset.verb_to_id
    val_dataset.id_to_verb = dataset.id_to_verb
    valid_mask = val_df['primary_verb'].isin(dataset.verb_to_id.keys())
    if (~valid_mask).sum() > 0:
        print(f"Dropping {(~valid_mask).sum()} val samples with unseen verbs")
        val_dataset.df = val_df[valid_mask].reset_index(drop=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    # --- Model, optimizer, scheduler ---
    num_verbs = len(dataset.verb_to_id)
    model = ActionToVerbTransformer(
        num_verbs=num_verbs, d_model=args.d_model,
        num_layers=args.num_layers,
        max_action_len=args.max_seq_len,
        img_size=img_size,
        modality=args.modality, action_rep=args.action_rep,
        fast_vocab_size=fast_vocab_size,
        cross_layers=args.cross_layers,
        vision_encoder=args.vision_encoder,
        freeze_vision=args.freeze_vision,
        num_frames=args.num_frames,
        delta_patches=delta_patches,
        modal_dropout=args.modal_dropout,
        aux_loss_weight=args.aux_loss_weight).to(device)

    # Optionally weight classes inversely by frequency
    if args.weighted_loss:
        class_counts = dataset.df['primary_verb'].value_counts()
        weights = torch.zeros(num_verbs)
        for verb, cid in dataset.verb_to_id.items():
            count = class_counts.get(verb, 1)
            weights[cid] = 1.0 / count
        weights = weights / weights.sum() * num_verbs  # normalize to mean=1
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"Using weighted CE loss (min weight={weights.min():.3f}, max={weights.max():.3f})")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    warmup_pct = min(args.warmup_epochs / args.epochs, 0.3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy='cos')

    # --- Training loop ---
    print("Starting training...")
    id_to_verb = dataset.id_to_verb
    training_log = []  # per-epoch metrics saved to JSON
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        # Per-class accumulators for train
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        class_loss_sum = defaultdict(float)

        for batch_idx, (frames, actions, labels, seq_lengths) in enumerate(dataloader):
            frames = frames.to(device)
            actions, labels = actions.to(device), labels.to(device)
            seq_lengths = seq_lengths.to(device)

            optimizer.zero_grad()
            main_logits, aux_v_logits, aux_a_logits = model.forward_with_aux(
                frames, actions, seq_lengths=seq_lengths)
            loss = criterion(main_logits, labels)
            if args.aux_loss_weight > 0.0:
                if aux_v_logits is not None:
                    loss = loss + args.aux_loss_weight * criterion(aux_v_logits, labels)
                if aux_a_logits is not None:
                    loss = loss + args.aux_loss_weight * criterion(aux_a_logits, labels)
            logits = main_logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Per-class stats (using per-sample loss)
            with torch.no_grad():
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, reduction='none')
                for lbl, pred, sl in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                    class_total[lbl] += 1
                    class_correct[lbl] += int(pred == lbl)
                    class_loss_sum[lbl] += sl

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}%")

        avg_loss = total_loss / len(dataloader)
        train_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"--- Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | LR: {current_lr:.2e} ---")

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = defaultdict(int)
        val_class_total = defaultdict(int)
        val_class_loss_sum = defaultdict(float)

        with torch.no_grad():
            for frames, actions, labels, seq_lengths in val_dataloader:
                frames = frames.to(device)
                actions, labels = actions.to(device), labels.to(device)
                seq_lengths = seq_lengths.to(device)

                logits = model(frames, actions, seq_lengths=seq_lengths)
                loss = criterion(logits, labels)
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, reduction='none')

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for lbl, pred, sl in zip(labels.cpu().tolist(),
                                         preds.cpu().tolist(),
                                         per_sample_loss.cpu().tolist()):
                    val_class_total[lbl] += 1
                    val_class_correct[lbl] += int(pred == lbl)
                    val_class_loss_sum[lbl] += sl

        val_avg = val_loss / len(val_dataloader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"    Val Loss: {val_avg:.4f} | Val Acc: {val_acc:.2f}%")

        # Build per-class metrics dict
        per_class_train = {}
        for cid in sorted(set(list(class_total.keys()) + list(val_class_total.keys()))):
            verb = id_to_verb.get(cid, str(cid))
            t = class_total.get(cid, 0)
            per_class_train[verb] = {
                "loss": class_loss_sum.get(cid, 0) / t if t > 0 else 0,
                "acc": 100 * class_correct.get(cid, 0) / t if t > 0 else 0,
                "count": t,
            }
        per_class_val = {}
        for cid in sorted(set(list(class_total.keys()) + list(val_class_total.keys()))):
            verb = id_to_verb.get(cid, str(cid))
            vt = val_class_total.get(cid, 0)
            per_class_val[verb] = {
                "loss": val_class_loss_sum.get(cid, 0) / vt if vt > 0 else 0,
                "acc": 100 * val_class_correct.get(cid, 0) / vt if vt > 0 else 0,
                "count": vt,
            }

        # Save best-val checkpoint
        if val_acc > best_val_acc and args.save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = args.save_path.replace(".pth", "_best.pth")
            best_ckpt = {
                'state_dict': model.state_dict(),
                'num_verbs': num_verbs,
                'verb_to_id': dataset.verb_to_id,
                'id_to_verb': dataset.id_to_verb,
                'd_model': args.d_model,
                'action_dim': ACTION_DIM,
                'nhead': NHEAD,
                'num_layers': args.num_layers,
                'patch_size': PATCH_SIZE,
                'img_size': img_size,
                'max_action_len': args.max_seq_len,
                'modality': args.modality,
                'action_rep': args.action_rep,
                'fast_vocab_size': fast_vocab_size,
                'cross_layers': args.cross_layers,
                'vision_encoder': args.vision_encoder,
                'freeze_vision': args.freeze_vision,
                'num_frames': args.num_frames,
                'delta_patches': delta_patches,
                'min_class_count': args.min_class_count,
                'vqvae_chunk_size': vqvae_chunk_size,
                'modal_dropout': args.modal_dropout,
                'aux_loss_weight': args.aux_loss_weight,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
            }
            save_dir = os.path.dirname(best_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(best_ckpt, best_path)
            print(f"    ★ New best val acc: {val_acc:.2f}% @ epoch {epoch+1} → {best_path}")

        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_avg,
            "val_acc": val_acc,
            "per_class_train": per_class_train,
            "per_class_val": per_class_val,
        }
        training_log.append(epoch_metrics)

        # Write log after each epoch so partial results are available
        if args.log_path:
            log_dir = os.path.dirname(args.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(args.log_path, "w") as f:
                json.dump({"config": {"modality": args.modality,
                                      "action_rep": args.action_rep,
                                      "cross_layers": args.cross_layers,
                                      "lr": args.lr, "batch_size": args.batch_size,
                                      "max_seq_len": args.max_seq_len},
                           "epochs": training_log}, f, indent=2)
            print(f"    Training log saved to {args.log_path}")

    # --- Save checkpoint ---
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'state_dict': model.state_dict(),
            'num_verbs': num_verbs,
            'verb_to_id': dataset.verb_to_id,
            'id_to_verb': dataset.id_to_verb,
            'd_model': args.d_model,
            'action_dim': ACTION_DIM,
            'nhead': NHEAD,
            'num_layers': args.num_layers,
            'patch_size': PATCH_SIZE,
            'img_size': img_size,
            'max_action_len': args.max_seq_len,
            'modality': args.modality,
            'action_rep': args.action_rep,
            'fast_vocab_size': fast_vocab_size,
            'cross_layers': args.cross_layers,
            'vision_encoder': args.vision_encoder,
            'freeze_vision': args.freeze_vision,
            'num_frames': args.num_frames,
            'delta_patches': delta_patches,
            'min_class_count': args.min_class_count,
            'vqvae_chunk_size': vqvae_chunk_size,
            'modal_dropout': args.modal_dropout,
            'aux_loss_weight': args.aux_loss_weight,
        }
        torch.save(checkpoint, args.save_path)
        print(f"\nCheckpoint saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to CALVIN training split")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR,
                        help="Path to CALVIN validation split")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS,
                        help="Number of warmup epochs for LR scheduler")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save checkpoint (e.g., model.pth)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to save training log JSON (e.g., training_log.json)")
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Debug mode: use only N samples for quick smoke testing")
    parser.add_argument("--modality", type=str, default="full",
                        choices=["full", "action_only", "vision_only"],
                        help="Which input modalities to use")
    parser.add_argument("--action_rep", type=str, default="native",
                        choices=["native", "fast", "vq_vae"],
                        help="Action representation: native continuous, FAST tokens, or VQ-VAE chunk codes")
    parser.add_argument("--fast_tokenizer_path", type=str, default=FAST_TOKENIZER_PATH,
                        help="Path to fitted FAST tokenizer")
    parser.add_argument("--vqvae_tokenizer_path", type=str, default=VQVAE_TOKENIZER_PATH,
                        help="Path to fitted VQ-VAE tokenizer")
    parser.add_argument("--vqvae_chunk_size", type=int, default=4,
                        help="Chunk size K used when fitting the VQ-VAE tokenizer")
    parser.add_argument("--cross_layers", type=int, default=CROSS_LAYERS,
                        help="Number of final layers with cross-modal attention "
                             "(default=NUM_LAYERS for early fusion)")
    parser.add_argument("--vision_encoder", type=str, default="patch",
                        choices=["patch", "r3m", "dinov2_s", "dinov2_b", "vc1"],
                        help="Vision encoder: patch, r3m, dinov2_s, dinov2_b, or vc1")
    parser.add_argument("--freeze_vision", action="store_true", default=True,
                        help="Freeze pretrained vision encoder weights (default: True)")
    parser.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false",
                        help="Fine-tune pretrained vision encoder")
    parser.add_argument("--num_frames", type=int, default=2,
                        help="Number of frames to sample (2=first+last, >2=uniformly spaced)")
    parser.add_argument("--delta_patches", type=int, default=0,
                        help="If >0, use only top-K changed patches between frame pairs "
                             "(requires ViT encoder: dinov2_s, dinov2_b, or vc1)")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use inverse-frequency weighted cross-entropy loss")
    parser.add_argument("--min_class_count", type=int, default=0,
                        help="Drop verb classes with fewer than N training samples (0=keep all)")
    parser.add_argument("--d_model", type=int, default=D_MODEL,
                        help="Transformer hidden dimension (default: 128)")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS,
                        help="Number of transformer layers (default: 4)")
    parser.add_argument("--modal_dropout", type=float, default=0.0,
                        help="Prob of zeroing each modality per batch (exclusive, full only). "
                             "e.g. 0.3 => 30%% vision-only batches, 30%% action-only, "
                             "40%% bimodal batches")
    parser.add_argument("--aux_loss_weight", type=float, default=0.0,
                        help="Weight lambda for auxiliary unimodal CE losses applied at "
                             "the self->cross layer transition (0 = disabled)")

    args = parser.parse_args()
    main(args)
