"""Verb classification models.

ActionToVerbTransformer: multimodal Transformer that fuses action trajectories
with vision (delta patches from VC-1 or DINOv2-S) via late cross-attention.
Classifies into N verb classes from [CLS] token output.

R3MEncoder: R3M pretrained vision encoder (ResNet -> global feature).
ViTEncoder: Frozen ViT patch encoder (DINOv2 or VC-1 via timm).
"""

import torch
import torch.nn as nn

from config import (
    ACTION_DIM, D_MODEL, NHEAD, NUM_LAYERS, CROSS_LAYERS,
    DROPOUT_RATE, PATCH_SIZE, IMAGE_SIZE, MAX_SEQ_LEN,
    R3M_VARIANT,
)
from verb_probe.image_encoders import build_image_encoder

# Modalities that fuse scene_obs with action trajectories (skip vision branch)
SCENE_FUSION_MODALITIES = ("scene_token", "scene_concat", "scene_film", "scene_mlp")


class R3MEncoder(nn.Module):
    """R3M pretrained vision encoder: ResNet -> global feature -> project to d_model.
    Each image becomes a single token (not patch tokens)."""

    def __init__(self, d_model, variant=R3M_VARIANT, freeze=True):
        super().__init__()
        from r3m import load_r3m
        r3m_model = load_r3m(variant)
        self.r3m = r3m_model.module
        self.freeze = freeze
        if freeze:
            self.r3m.eval()
            for p in self.r3m.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.r3m.outdim, d_model)
        self.num_patches = 1

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                features = self.r3m(x)
        else:
            features = self.r3m(x)
        return self.proj(features).unsqueeze(1)


class ViTEncoder(nn.Module):
    """Frozen ViT patch encoder using DINOv2 or VC-1 via timm.
    Extracts patch tokens, spatially pools to pool_size x pool_size,
    and projects to d_model."""

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
            self.grid_size = 16
        elif variant == "dinov2_b":
            self.vit = timm.create_model('vit_base_patch14_dinov2',
                                          pretrained=True, num_classes=0,
                                          img_size=224)
            embed_dim = 768
            self.grid_size = 16
        elif variant == "vc1":
            self.vit = timm.create_model('vit_base_patch16_224',
                                          pretrained=False, num_classes=0)
            embed_dim = 768
            self.grid_size = 14
            self._load_vc1_weights()
        else:
            raise ValueError(f"Unknown ViT variant: {variant}")

        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Linear(embed_dim, d_model)
        self.num_patches = pool_size * pool_size

    def _load_vc1_weights(self):
        from huggingface_hub import hf_hub_download
        import torch as _torch
        path = hf_hub_download('facebook/vc1-base', 'pytorch_model.bin')
        vc1_state = _torch.load(path, map_location='cpu', weights_only=True)['model']

        timm_state = {}
        for k, v in vc1_state.items():
            new_key = k
            if k == 'norm.weight':
                new_key = 'norm.weight'
            elif k == 'norm.bias':
                new_key = 'norm.bias'
            timm_state[new_key] = v

        missing, unexpected = self.vit.load_state_dict(timm_state, strict=False)
        n_loaded = len(timm_state) - len(unexpected)
        print(f"VC-1: loaded {n_loaded} params, {len(missing)} missing, "
              f"{len(unexpected)} unexpected")

    def forward(self, x):
        with torch.no_grad():
            features = self.vit.forward_features(x)
        patches = features[:, 1:, :]

        B, N, D = patches.shape
        h = w = self.grid_size
        patches = patches.transpose(1, 2).reshape(B, D, h, w)
        patches = self.pool(patches)
        patches = patches.flatten(2).transpose(1, 2)

        return self.proj(patches)


class ActionToVerbTransformer(nn.Module):
    def __init__(self, num_verbs, action_vocab_size=None, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=NUM_LAYERS, action_dim=ACTION_DIM,
                 dropout=DROPOUT_RATE, img_size=IMAGE_SIZE[0],
                 patch_size=PATCH_SIZE, max_action_len=MAX_SEQ_LEN,
                 modality="full", action_rep="native",
                 cross_layers=CROSS_LAYERS,
                 image_encoder="scratch", freeze_vision=True,
                 num_frames=2, delta_patches=0,
                 modal_dropout=0.0, aux_loss_weight=0.0,
                 scene_dim=0):
        super().__init__()
        self.modality = modality
        self.action_rep = action_rep
        self.num_layers = num_layers
        self.cross_layers = cross_layers
        self.image_encoder_type = image_encoder
        self.num_frames = num_frames
        self.delta_patches = delta_patches
        self.modal_dropout = modal_dropout
        self.aux_loss_weight = aux_loss_weight
        self.scene_dim = scene_dim

        # -- Scene fusion branches --
        if modality == "scene_token" and scene_dim > 0:
            self.scene_proj = nn.Sequential(
                nn.Linear(scene_dim, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
            self.type_scene = nn.Parameter(torch.zeros(1, 1, d_model))
        if modality == "scene_concat" and scene_dim > 0:
            self.scene_mlp = nn.Sequential(
                nn.Linear(scene_dim, 64), nn.ReLU(),
                nn.Linear(64, d_model))
        if modality == "scene_film" and scene_dim > 0:
            self.film_net = nn.Sequential(
                nn.Linear(scene_dim, 128), nn.ReLU(),
                nn.Linear(128, 2 * d_model))
        if modality == "scene_mlp" and scene_dim > 0:
            self.scene_only_mlp = nn.Sequential(
                nn.Linear(scene_dim, d_model), nn.LayerNorm(d_model),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

        # -- Vision branch --
        if modality not in ("action_only", "scene_obs") + SCENE_FUSION_MODALITIES:
            self.patch_embed = build_image_encoder(image_encoder, d_model, img_size, patch_size)
            full_num_patches = self.patch_embed.num_tokens
            self.num_patches = delta_patches if delta_patches > 0 else full_num_patches
            self.patch_pos = nn.Parameter(torch.zeros(1, full_num_patches, d_model))
            n_temporal = max(num_frames - 1, 1) if delta_patches > 0 else num_frames
            self.frame_pos = nn.Parameter(torch.zeros(1, n_temporal, 1, d_model))
            self.type_img = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.num_patches = (img_size // patch_size) ** 2

        # -- Action branch --
        if modality != "vision_only":
            if action_rep == "native":
                self.action_proj = nn.Linear(action_dim, d_model)
            else:
                assert action_vocab_size is not None
                self.action_embed = nn.Embedding(action_vocab_size, d_model)
            self.action_pos = nn.Parameter(torch.zeros(1, max_action_len, d_model))
            self.type_action = nn.Parameter(torch.zeros(1, 1, d_model))

        # -- CLS token --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        for name, p in self.named_parameters():
            if 'type_' in name or '_pos' in name or 'cls_token' in name:
                nn.init.trunc_normal_(p, std=0.02)

        # -- Transformer encoder --
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
            for _ in range(num_layers)
        ])

        # -- Classification head --
        cls_input_dim = 2 * d_model if (modality == "scene_concat" and scene_dim > 0) else d_model
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_verbs))

        # -- Auxiliary unimodal heads --
        if aux_loss_weight > 0.0 and modality == "full":
            self.aux_vision_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_verbs))
            self.aux_action_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_verbs))
        else:
            self.aux_vision_head = None
            self.aux_action_head = None

    def _forward_core(self, frames, trajectories, seq_lengths=None,
                      training=False, scene_vec=None):
        batch_size = trajectories.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos + self.type_cls
        parts = [cls]

        if self.modality == "scene_token" and self.scene_dim > 0 and scene_vec is not None:
            scene_tok = self.scene_proj(scene_vec).unsqueeze(1) + self.type_scene
            parts.append(scene_tok)

        nf = 0
        if self.modality not in ("action_only", "scene_obs") + SCENE_FUSION_MODALITIES:
            nf = frames.size(1)
            if self.delta_patches > 0:
                all_patches = [self.patch_embed(frames[:, fi]) for fi in range(nf)]
                K = self.delta_patches
                d = all_patches[0].size(-1)
                for pi in range(nf - 1):
                    diff = all_patches[pi + 1] - all_patches[pi]
                    mag = diff.norm(dim=-1)
                    topk_idx = mag.topk(K, dim=-1).indices
                    idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, d)
                    selected = torch.gather(diff, 1, idx_exp)
                    pos = torch.gather(
                        self.patch_pos.expand(batch_size, -1, -1), 1, idx_exp)
                    selected = selected + pos + self.frame_pos[:, pi] + self.type_img
                    parts.append(selected)
            else:
                for fi in range(nf):
                    patches = self.patch_embed(frames[:, fi])
                    patches = patches + self.patch_pos + self.frame_pos[:, fi] + self.type_img
                    parts.append(patches)

        if self.modality != "vision_only":
            action_len = trajectories.size(1)
            if self.action_rep == "native":
                action_emb = self.action_proj(trajectories)
            else:
                action_emb = self.action_embed(trajectories)
            if self.modality == "scene_film" and self.scene_dim > 0 and scene_vec is not None:
                film_out = self.film_net(scene_vec)
                gamma, beta = film_out.chunk(2, dim=-1)
                gamma = gamma + 1.0
                action_emb = gamma.unsqueeze(1) * action_emb + beta.unsqueeze(1)
            action_emb = action_emb + self.action_pos[:, :action_len, :] + self.type_action
            parts.append(action_emb)

        full_seq = torch.cat(parts, dim=1)
        total_len = full_seq.size(1)

        if self.modality == "scene_token" and self.scene_dim > 0:
            n_vis_tokens = 1
        elif self.modality in ("action_only", "scene_obs",
                               "scene_concat", "scene_film"):
            n_vis_tokens = 0
        elif self.delta_patches > 0:
            n_vis_tokens = max(nf - 1, 1) * self.num_patches
        else:
            n_vis_tokens = nf * self.num_patches
        v_start, v_end = 1, 1 + n_vis_tokens

        if training and self.modal_dropout > 0.0 and self.modality == "full":
            r = torch.rand(1).item()
            if r < self.modal_dropout:
                full_seq = full_seq.clone()
                full_seq[:, v_start:v_end] = 0.0
            elif r < 2 * self.modal_dropout:
                full_seq = full_seq.clone()
                full_seq[:, v_end:] = 0.0

        src_key_padding_mask = None
        if seq_lengths is not None:
            positions = torch.arange(total_len, device=full_seq.device).unsqueeze(0)
            src_key_padding_mask = positions >= seq_lengths.unsqueeze(1)

        self_mask = None
        num_self_layers = self.num_layers - self.cross_layers
        if num_self_layers > 0 and self.modality in ("full", "scene_token"):
            self_mask = torch.full((total_len, total_len), float('-inf'),
                                   device=full_seq.device)
            self_mask[0, 0] = 0.0
            self_mask[v_start:v_end, v_start:v_end] = 0.0
            self_mask[v_end:, v_end:] = 0.0

        x_transition = None
        x = full_seq
        for i, layer in enumerate(self.layers):
            if (i == num_self_layers
                    and self.aux_loss_weight > 0.0
                    and self.modality == "full"):
                x_transition = x
            mask = self_mask if i < num_self_layers else None
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if (self.aux_loss_weight > 0.0 and self.modality == "full"
                and x_transition is None):
            x_transition = full_seq

        return x, v_end, x_transition

    def forward(self, frames, trajectories, seq_lengths=None, scene_vec=None):
        if self.modality == "scene_mlp" and self.scene_dim > 0 and scene_vec is not None:
            cls_out = self.scene_only_mlp(scene_vec)
            return self.classifier(cls_out)
        x, _v_end, _xt = self._forward_core(
            frames, trajectories, seq_lengths=seq_lengths,
            training=self.training, scene_vec=scene_vec)
        cls_out = x[:, 0, :]
        if self.modality == "scene_concat" and self.scene_dim > 0 and scene_vec is not None:
            scene_emb = self.scene_mlp(scene_vec)
            cls_out = torch.cat([cls_out, scene_emb], dim=-1)
        return self.classifier(cls_out)

    def forward_with_aux(self, frames, trajectories, seq_lengths=None, scene_vec=None):
        if self.modality == "scene_mlp" and self.scene_dim > 0 and scene_vec is not None:
            cls_out = self.scene_only_mlp(scene_vec)
            return self.classifier(cls_out), None, None
        x, v_end, x_transition = self._forward_core(
            frames, trajectories, seq_lengths=seq_lengths,
            training=self.training, scene_vec=scene_vec)
        cls_out = x[:, 0, :]
        if self.modality == "scene_concat" and self.scene_dim > 0 and scene_vec is not None:
            scene_emb = self.scene_mlp(scene_vec)
            cls_out = torch.cat([cls_out, scene_emb], dim=-1)
        main_logits = self.classifier(cls_out)
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

    @torch.no_grad()
    def get_cls_attn_fracs(self, frames, trajectories, seq_lengths=None):
        num_self = self.num_layers - self.cross_layers
        v_start = 1
        input_cache = {}
        hooks = []
        for i in range(num_self, self.num_layers):
            def _make_hook(li):
                def _hook(_, args):
                    input_cache[li] = tuple(
                        a.detach() if isinstance(a, torch.Tensor) else a
                        for a in args[:3])
                return _hook
            hooks.append(
                self.layers[i].self_attn.register_forward_pre_hook(_make_hook(i)))
        _, v_end, _ = self._forward_core(
            frames, trajectories, seq_lengths=seq_lengths, training=False)
        for h in hooks:
            h.remove()
        fracs = {}
        for i in range(num_self, self.num_layers):
            if i not in input_cache:
                continue
            q, k, v = input_cache[i]
            _, attn_w = self.layers[i].self_attn(
                q, k, v, need_weights=True, average_attn_weights=True)
            if attn_w is None:
                continue
            cls_vis = attn_w[:, 0, v_start:v_end].sum(-1).mean().item()
            cls_act = attn_w[:, 0, v_end:].sum(-1).mean().item()
            total = cls_vis + cls_act + 1e-9
            fracs[i] = {"vision": cls_vis / total, "action": cls_act / total}
        return fracs, v_end
