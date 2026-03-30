"""Auxiliary heads for action tokenizer training.

ActionTransformer: shared CLS-token Transformer backbone for sequence pooling.
VerbHead: episode-level verb classifier (wraps ActionTransformer).
ContrastiveHead: action-language contrastive alignment (wraps ActionTransformer).
TextEncoderWrapper: pretrained text encoder (CLIP/GPT-2) with optional LoRA.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SemanticLoss


# ======================================================================
# LoRA
# ======================================================================

class LoRALayer(nn.Module):
    """Low-rank adaptation: W' = W + BA."""

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
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora(x)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features if hasattr(self.original, 'in_features') else self.original.weight.shape[0]


# ======================================================================
# Positional encoding
# ======================================================================

def sinusoidal_position_encoding(positions, d_model):
    """Generate sinusoidal PE from continuous normalized positions.

    Args:
        positions: (B, K) float in [0, 1] — normalized temporal position
        d_model: embedding dimension
    Returns:
        (B, K, d_model) positional encoding
    """
    B, K = positions.shape
    device = positions.device
    half_d = d_model // 2
    freq = torch.exp(torch.arange(half_d, device=device, dtype=torch.float32)
                     * -(math.log(10000.0) / half_d))
    # Scale positions to a range similar to standard integer PE
    # positions in [0,1] → scale by max_len equivalent (e.g. 100)
    angles = positions.unsqueeze(-1) * 100.0 * freq.unsqueeze(0).unsqueeze(0)
    pe = torch.zeros(B, K, d_model, device=device)
    pe[:, :, 0::2] = torch.sin(angles)
    pe[:, :, 1::2] = torch.cos(angles[:, :, :d_model - half_d] if d_model % 2 else angles)
    return pe


# ======================================================================
# Action Transformer (shared backbone)
# ======================================================================

class ActionTransformer(nn.Module):
    """CLS-token Transformer over token sequences with sinusoidal PE.

    Prepends a learnable [CLS] token, adds sinusoidal positional encoding
    from normalized temporal positions, runs a pre-norm TransformerEncoder,
    and returns the CLS output as the sequence-level embedding.

    Used by both VerbHead and ContrastiveHead to ensure identical action
    encoding.
    """

    def __init__(self, input_dim=64, d_model=128, nhead=4, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, n_valid, positions=None):
        """
        Args:
            tokens: (B, S, input_dim) — input token sequence
            n_valid: (B,) — number of real tokens per sequence
            positions: (B, S) float in [0,1] — normalized temporal positions.
                       If None, uses index-based positions normalized by n_valid.
        """
        S = tokens.size(1)
        device = tokens.device

        x = self.input_proj(tokens)

        # Sinusoidal PE from normalized positions
        if positions is None:
            idx = torch.arange(S, device=device).float().unsqueeze(0)
            positions = idx / n_valid.float().unsqueeze(1).clamp(min=1)
        x = x + sinusoidal_position_encoding(positions, self.d_model)

        # Prepend CLS
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Padding mask (CLS is always valid)
        pad_mask = torch.arange(S, device=device).unsqueeze(0) >= \
            n_valid.unsqueeze(1)
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.norm(x[:, 0, :])


# ======================================================================
# Auxiliary heads
# ======================================================================

class VerbHead(nn.Module):
    """Episode-level verb classifier wrapping ActionTransformer.

    Computes sinusoidal PE from normalized temporal positions and delegates
    the CLS + Transformer pooling to ActionTransformer. Classifies from
    the CLS output.
    """

    def __init__(self, latent_dim, num_verbs, d_model=128, nhead=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.action_transformer = ActionTransformer(
            input_dim=latent_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_verbs),
        )

    def forward(self, chunk_latents, n_valid, positions=None):
        """
        Args:
            chunk_latents: (B, max_K, latent_dim) — encoder output per chunk
            n_valid: (B,) — number of real chunks per episode
            positions: (B, max_K) float in [0,1] — normalized temporal position
                       of each chunk. If None, uses index-based position.
        Returns:
            logits: (B, num_verbs)
        """
        cls_out = self.action_transformer(
            chunk_latents, n_valid, positions=positions)
        return self.classifier(cls_out)


class ContrastiveHead(nn.Module):
    """Action transformer + projection for contrastive alignment."""

    def __init__(self, latent_dim=128, d_model=128, nhead=4,
                 transformer_layers=2, proj_dim=128, dropout=0.1):
        super().__init__()
        self.action_transformer = ActionTransformer(
            input_dim=latent_dim, d_model=d_model, nhead=nhead,
            num_layers=transformer_layers, dropout=dropout)
        self.action_proj = nn.Linear(d_model, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=20.0)

    def forward(self, window_latents, n_windows, positions=None):
        cls_out = self.action_transformer(window_latents, n_windows,
                                          positions=positions)
        action_emb = self.action_proj(cls_out)
        return F.normalize(action_emb, dim=-1)


def contrastive_loss(action_emb, text_emb, text_list, temperature):
    """Symmetric InfoNCE with false-negative masking."""
    B = len(action_emb)
    device = action_emb.device

    logits = (action_emb @ text_emb.T) * temperature

    # Same instruction = positive pair
    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if text_list[i] == text_list[j]:
                pos_mask[i, j] = True

    # Action -> text
    logits_stable = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits_stable - logits_stable.exp().sum(dim=1, keepdim=True).log()
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_a2t = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

    # Text -> action
    logits_t = (text_emb @ action_emb.T) * temperature
    logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values.detach()
    log_prob_t = logits_t - logits_t.exp().sum(dim=1, keepdim=True).log()
    n_pos_t = pos_mask.T.float().sum(dim=1).clamp(min=1)
    loss_t2a = -(log_prob_t * pos_mask.T.float()).sum(dim=1) / n_pos_t

    return (loss_a2t.mean() + loss_t2a.mean()) / 2


# ======================================================================
# Text encoder
# ======================================================================

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
            self.output_dim = clip_model.config.projection_dim
            del clip_model.vision_model
        elif model_type == 'gpt2':
            from transformers import GPT2Model, GPT2Tokenizer
            self.text_model = GPT2Model.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.text_projection = None
            self.output_dim = self.text_model.config.hidden_size
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if freeze:
            for p in self.text_model.parameters():
                p.requires_grad = False
            if self.text_projection is not None:
                for p in self.text_projection.parameters():
                    p.requires_grad = False

        if lora_r > 0:
            self._apply_lora(lora_r)

    def _apply_lora(self, r):
        self.lora_layers = nn.ModuleList()
        if self.model_type == 'clip':
            for layer in self.text_model.encoder.layers:
                attn = layer.self_attn
                dim = attn.q_proj.in_features
                lora_q = LoRALayer(dim, dim, r)
                lora_v = LoRALayer(dim, dim, r)
                self.lora_layers.append(lora_q)
                self.lora_layers.append(lora_v)
                attn.q_proj = LoRAWrappedLinear(attn.q_proj, lora_q)
                attn.v_proj = LoRAWrappedLinear(attn.v_proj, lora_v)
        elif self.model_type == 'gpt2':
            for block in self.text_model.h:
                attn = block.attn
                dim_in = attn.c_attn.weight.shape[0]
                dim_out = attn.c_attn.weight.shape[1]
                lora = LoRALayer(dim_in, dim_out, r)
                self.lora_layers.append(lora)
                attn.c_attn = LoRAWrappedLinear(attn.c_attn, lora)

    def precompute_cache(self, all_instructions):
        """Precompute and cache embeddings for all unique instruction strings.

        Call once before training when the text encoder is frozen (lora_r == 0).
        Subsequent forward() calls return cached embeddings instead of running
        the text model.
        """
        unique = list(set(all_instructions))
        if not unique:
            return
        device = next(self.text_model.parameters()).device
        self._embed_cache = {}
        # Encode in batches to avoid OOM
        bs = 256
        for i in range(0, len(unique), bs):
            batch = unique[i:i + bs]
            with torch.no_grad():
                emb = self._encode(batch, device)
            for text, vec in zip(batch, emb):
                self._embed_cache[text] = vec.cpu()
        print(f"Cached text embeddings for {len(self._embed_cache)} unique instructions")

    def _encode(self, text_list, device):
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                return_tensors='pt').to(device)
        if self.model_type == 'clip':
            outputs = self.text_model(**inputs)
            pooled = outputs.pooler_output
            if self.text_projection is not None:
                pooled = self.text_projection(pooled)
            return pooled
        elif self.model_type == 'gpt2':
            outputs = self.text_model(**inputs)
            seq_lens = inputs['attention_mask'].sum(dim=-1) - 1
            batch_idx = torch.arange(len(text_list), device=device)
            pooled = outputs.last_hidden_state[batch_idx, seq_lens]
            return pooled

    def forward(self, text_list):
        device = next(self.text_model.parameters()).device

        # Serve from cache if available (frozen encoder)
        if hasattr(self, '_embed_cache') and self._embed_cache:
            # Fall back to live encoding for any uncached strings
            missed = [t for t in text_list if t not in self._embed_cache]
            if missed:
                with torch.no_grad():
                    emb = self._encode(missed, device)
                for t, vec in zip(missed, emb):
                    self._embed_cache[t] = vec.cpu()
            return torch.stack([self._embed_cache[t] for t in text_list]).to(device)

        return self._encode(text_list, device)


# ======================================================================
# Builder
# ======================================================================

def get_head_latent_dim(tokenizer_type, latent_dim=64, aux_target='latent',
                        fsq_dim=None):
    """Determine the input dimension for aux heads given tokenizer config.

    OAT/QueST normally use the pre-FSQ 256-d encoder output.
    With aux_target='post_fsq', use the FSQ codebook dimension instead.
    """
    if aux_target == 'post_fsq' and fsq_dim is not None:
        return fsq_dim
    if tokenizer_type in ('vq_vae', 'vq_bet'):
        return latent_dim
    if tokenizer_type == 'vqvla':
        return 128
    if tokenizer_type in ('oat', 'quest'):
        return 256  # always pre-FSQ
    return 128


def build_aux_heads(tokenizer_type, device,
                    latent_dim=64,
                    num_verbs=0, verb_class_weights=None,
                    clip_config=None,
                    loss_function='ce',
                    semantic_temp=0.1,
                    id_to_verb=None,
                    aux_target='latent',
                    fsq_dim=None):
    """Build auxiliary heads for tokenizer training.

    Args:
        tokenizer_type: one of 'vq_vae', 'vq_bet', 'vqvla', 'oat', 'quest'
        device: torch device
        latent_dim: VQ-VAE/VQ-BeT latent dim
        num_verbs: number of verb classes (0 = no verb head)
        verb_class_weights: (num_verbs,) tensor of inverse-frequency weights, or None
        clip_config: dict with keys {d_model, transformer_layers, proj_dim,
                     text_model, text_type, text_lora_r}, or None to skip CLIP head
        aux_target: 'latent' (256d pre-FSQ) or 'post_fsq' (4d post-round)
        fsq_dim: FSQ codebook dimension (e.g. 4 for [8,5,5,5]), required when
                 aux_target='post_fsq'

    Returns:
        dict with keys: verb_head, verb_criterion, clip_head, text_encoder,
        text_proj, head_latent_dim
    """
    head_latent_dim = get_head_latent_dim(
        tokenizer_type, latent_dim=latent_dim,
        aux_target=aux_target, fsq_dim=fsq_dim)

    # Use small d_model when input is low-dimensional post-FSQ codes
    head_d_model = 16 if aux_target == 'post_fsq' else 128
    head_nhead = 2 if aux_target == 'post_fsq' else 4

    result = dict(verb_head=None, verb_criterion=None,
                  clip_head=None, text_encoder=None, text_proj=None,
                  head_latent_dim=head_latent_dim)

    # ── Verb head (always uses weighted CE when weights are provided) ──
    if num_verbs > 0:
        result['verb_head'] = VerbHead(
            head_latent_dim, num_verbs,
            d_model=head_d_model, nhead=head_nhead,
        ).to(device)
        print(f"Verb head: {head_latent_dim} -> d{head_d_model} -> CLS -> {num_verbs} classes")

        if loss_function == 'semantic':
            if id_to_verb is None:
                raise ValueError("id_to_verb mapping is required for SemanticLoss")
            
            result['verb_criterion'] = SemanticLoss(
                id_to_verb=id_to_verb,
                temperature=semantic_temp,
                class_weights=verb_class_weights 
            ).to(device)
            print(f"Using SemanticLoss with temperature {semantic_temp}")
        else:
            if verb_class_weights is not None:
                w = verb_class_weights.to(device)
                result['verb_criterion'] = nn.CrossEntropyLoss(
                    weight=w, ignore_index=-1)
            else:
                result['verb_criterion'] = nn.CrossEntropyLoss(ignore_index=-1)
            print("Using standard CrossEntropyLoss")

    # ── CLIP head ──────────────────────────────────────────────────────
    if clip_config is not None:
        cfg = clip_config
        clip_d = head_d_model if aux_target == 'post_fsq' else cfg.get('d_model', 128)
        clip_nhead = head_nhead if aux_target == 'post_fsq' else 4
        result['clip_head'] = ContrastiveHead(
            latent_dim=head_latent_dim,
            d_model=clip_d,
            nhead=clip_nhead,
            transformer_layers=cfg.get('transformer_layers', 2),
            proj_dim=cfg.get('proj_dim', 128),
        ).to(device)
        result['text_encoder'] = TextEncoderWrapper(
            model_name=cfg.get('text_model',
                               'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'),
            model_type=cfg.get('text_type', 'clip'),
            freeze=(cfg.get('text_lora_r', 0) == 0),
            lora_r=cfg.get('text_lora_r', 0),
        ).to(device)
        result['text_proj'] = nn.Linear(
            result['text_encoder'].output_dim,
            cfg.get('proj_dim', 128)).to(device)
        print(f"CLIP head: latent_dim={head_latent_dim}, "
              f"proj_dim={cfg.get('proj_dim', 128)}")

    return result
