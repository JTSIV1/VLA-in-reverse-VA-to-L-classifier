"""Unified action tokenizer training / fitting script.

Supports all tokenizer types:
  Gradient-based (iterative training):
    vq_bet   - chunk-based MLP + ResidualVQ (VQ-BeT paper)
    oat      - register encoder + FSQ (from oat/)
    quest    - causal conv + FSQ (from oat/)

  Fit-once (non-gradient):
    fast     - DCT + BPE fitting
    bin      - analytical binning (no training, eval only)

Optional auxiliary losses (for gradient-based tokenizers):
    --verb_cls_lambda L  - verb classification head on pooled latents
    --clip_lambda L      - contrastive action-language head

Usage:
    # VQ-BeT from scratch
    python tokenization/train_tokenizer.py --tokenizer vq_bet --epochs 100

    # OAT from scratch with verb head
    python tokenization/train_tokenizer.py --tokenizer oat --verb_cls_lambda 0.5 --epochs 50

    # FAST fit
    python tokenization/train_tokenizer.py --tokenizer fast --fast_vocab_size 1024

    # Resume from checkpoint
    python tokenization/train_tokenizer.py --tokenizer vq_bet --resume checkpoints/vq_bet_vanilla/full.pth
"""

import csv
import json
import math
import os
import sys
import time as time_mod
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_TOKENIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZATION_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZATION_DIR)

from utils import extract_verb, load_calvin_to_dataframe
from config import (
    DATA_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE, ACTION_DIM,
    TOKENIZER_HORIZON,
    TOKENIZER_DOWNSAMPLE_FACTOR, OAT_NUM_REGISTERS,
)
from datasets.calvin_dataset import (
    CalvinTokenizerDataset, CalvinActionCropDataset, CalvinDataset,
)

# ======================================================================
# Bridge data loading
# ======================================================================

import glob as _glob
from torch.utils.data import Dataset as _Dataset


DROID_ACTIONS_DIR = "/data/user_data/wenjiel2/datasets/droid_actions"
DROID_METADATA_CACHE = os.path.join(_PROJECT_ROOT, "data", "droid_tokenizer_metadata.csv")

DROID_VERB_MERGE_MAP = {
    'flip over': 'flip',
    'fold up': 'fold',
    'stack up': 'stack',
    'press down': 'press',
    'slide out': 'slide',
    'slide up': 'slide',
    'slide down': 'slide',
    'put down': 'put',
    'pull out': 'pull',
    'pull up': 'pull',
    'pull down': 'pull',
    'push down': 'push',
    'push up': 'push',
    'push in': 'push',
    'pour out': 'pour',
    'move up': 'move',
    'lift up': 'lift',
    'turn over': 'flip',
}


class BridgeActionChunkDataset(_Dataset):
    """Yields (horizon, 7) action chunks from BridgeV2 shards (dict format)."""

    def __init__(self, actions_list, horizon=32):
        self.horizon = horizon
        self.indices = []
        self.actions = actions_list
        for ep_idx, ep_actions in enumerate(actions_list):
            T = len(ep_actions)
            if T >= horizon:
                for start in range(T - horizon + 1):
                    self.indices.append((ep_idx, start))
            elif T >= 2:
                self.indices.append((ep_idx, 0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        actions = self.actions[ep_idx]
        T = len(actions)
        if T >= self.horizon:
            chunk = actions[start:start + self.horizon]
        else:
            chunk = np.pad(actions, ((0, self.horizon - T), (0, 0)), mode="edge")
        return {"action": torch.tensor(chunk, dtype=torch.float32)}


class BridgeFlatChunkDataset(_Dataset):
    """Yields (chunk_size * 7,) flat action chunks for VQ-BeT."""

    def __init__(self, actions_list, chunk_size=5):
        self.chunk_size = chunk_size
        self.indices = []
        self.actions = actions_list
        for ep_idx, ep_actions in enumerate(actions_list):
            T = len(ep_actions)
            if T >= chunk_size:
                for start in range(T - chunk_size + 1):
                    self.indices.append((ep_idx, start))
            elif T >= 2:
                self.indices.append((ep_idx, 0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        actions = self.actions[ep_idx]
        T = len(actions)
        if T >= self.chunk_size:
            chunk = actions[start:start + self.chunk_size]
        else:
            chunk = np.pad(actions, ((0, self.chunk_size - T), (0, 0)), mode="edge")
        return torch.tensor(chunk.flatten(), dtype=torch.float32)


def _np_scalar_to_str(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return str(value.tolist())
    return str(value)


@lru_cache(maxsize=16)
def _load_droid_shard_cached(shard_path):
    return np.load(shard_path, allow_pickle=True)


def _iter_droid_shard_records(shard_path):
    data = _load_droid_shard_cached(shard_path)
    action_keys = sorted(
        (k for k in data.files if k.startswith("actions_")),
        key=lambda key: int(key.split("_")[1]),
    )
    for action_key in action_keys:
        episode_idx = int(action_key.split("_")[1])
        actions = data[action_key]
        yield {
            'shard_path': shard_path,
            'episode_idx': episode_idx,
            'episode_path': _np_scalar_to_str(data[f'episode_path_{episode_idx}']),
            'instruction': _np_scalar_to_str(data[f'lang1_{episode_idx}']),
            'instruction2': _np_scalar_to_str(data[f'lang2_{episode_idx}']),
            'instruction3': _np_scalar_to_str(data[f'lang3_{episode_idx}']),
            'length': int(actions.shape[0]),
        }


def load_droid_metadata(actions_dir, metadata_cache=None, rebuild=False,
                        max_shards=None):
    metadata_cache = metadata_cache or DROID_METADATA_CACHE
    if os.path.exists(metadata_cache) and not rebuild:
        print(f"Loading DROID metadata from {metadata_cache}")
        return pd.read_csv(metadata_cache)

    shard_files = sorted(_glob.glob(os.path.join(actions_dir, "shard_*.npz")))
    if max_shards is not None:
        shard_files = shard_files[:max_shards]
    if not shard_files:
        raise FileNotFoundError(f"No DROID shard_*.npz files found under {actions_dir}")

    print(f"Building DROID metadata from {len(shard_files)} shards...")
    rows = []
    for shard_path in tqdm(shard_files, desc="Scanning DROID shards"):
        rows.extend(_iter_droid_shard_records(shard_path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No DROID episodes found under {actions_dir}")

    df['verbs'] = df['instruction'].apply(extract_verb)
    df = df[df['verbs'].apply(len) == 1].copy()
    df = df[~df['instruction'].str.contains(r'\bthen\b', case=False, na=False)].copy()
    and_mask = (
        df['instruction'].str.contains(r'\band\b', case=False, na=False)
        & ~df['instruction'].str.lower().str.startswith('go')
    )
    df = df[~and_mask].copy()
    df['primary_verb'] = df['verbs'].apply(lambda verbs: verbs[0]).replace(DROID_VERB_MERGE_MAP)
    df = df.drop(columns=['verbs']).reset_index(drop=True)

    os.makedirs(os.path.dirname(metadata_cache), exist_ok=True)
    df.to_csv(metadata_cache, index=False)
    print(f"Saved DROID metadata to {metadata_cache} ({len(df)} episodes)")
    return df


def split_episode_dataframe(df, val_fraction=0.1, seed=42,
                            max_train_episodes=None, max_val_episodes=None):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_fraction))
    val_df = df.iloc[perm[:n_val]].reset_index(drop=True)
    train_df = df.iloc[perm[n_val:]].reset_index(drop=True)
    if max_train_episodes is not None:
        train_df = train_df.head(max_train_episodes).reset_index(drop=True)
    if max_val_episodes is not None:
        val_df = val_df.head(max_val_episodes).reset_index(drop=True)
    return train_df, val_df


class DroidDatasetBase(_Dataset):
    def __init__(self, df, verb_to_id=None):
        self.df = df.reset_index(drop=True)
        self._verb_col = 'primary_verb' if 'primary_verb' in self.df.columns else 'verb'
        if verb_to_id is not None:
            self.verb_to_id = verb_to_id
        elif self._verb_col in self.df.columns:
            unique_verbs = sorted(self.df[self._verb_col].dropna().unique())
            self.verb_to_id = {verb: idx for idx, verb in enumerate(unique_verbs)}
        else:
            self.verb_to_id = {}
        self.id_to_verb = {idx: verb for verb, idx in self.verb_to_id.items()}

    def __len__(self):
        return len(self.df)

    def _get_row(self, idx):
        return self.df.iloc[idx]

    def _get_actions(self, idx):
        row = self._get_row(idx)
        shard = _load_droid_shard_cached(row['shard_path'])
        return shard[f"actions_{int(row['episode_idx'])}"].astype(np.float32)

    def _get_instruction(self, idx):
        return self._get_row(idx)['instruction']

    def _get_verb_id(self, idx):
        verb = self._get_row(idx).get(self._verb_col, None)
        return self.verb_to_id.get(verb, 0) if verb else 0

    def _chunk_actions(self, actions, window_size, max_windows, sample_windows=False):
        T, action_dim = actions.shape
        n_windows = T // window_size
        if n_windows == 0:
            padded = np.pad(actions, ((0, window_size - T), (0, 0)), mode='edge')
            windows = padded.reshape(1, window_size, action_dim)
            n_windows = 1
        else:
            usable = n_windows * window_size
            windows = actions[:usable].reshape(n_windows, window_size, action_dim)

        if n_windows > max_windows:
            if sample_windows:
                choose = np.sort(np.random.choice(n_windows, max_windows, replace=False))
                windows = windows[choose]
            else:
                windows = windows[:max_windows]
            n_windows = max_windows

        padded_windows = np.zeros((max_windows, window_size, action_dim), dtype=np.float32)
        padded_windows[:n_windows] = windows
        return padded_windows, n_windows


class DroidActionCropDataset(DroidDatasetBase):
    def __init__(self, df, horizon=32, random_crop=True, verb_to_id=None):
        super().__init__(df, verb_to_id=verb_to_id)
        self.horizon = horizon
        self.random_crop = random_crop

    def __getitem__(self, idx):
        actions = self._get_actions(idx)
        T = actions.shape[0]
        if T >= self.horizon:
            if self.random_crop:
                start = np.random.randint(0, T - self.horizon + 1)
            else:
                start = max((T - self.horizon) // 2, 0)
            chunk = actions[start:start + self.horizon]
        else:
            chunk = np.pad(actions, ((0, self.horizon - T), (0, 0)), mode='constant').astype(np.float32)
        return {"action": torch.tensor(chunk.tolist(), dtype=torch.float32)}


class DroidFlatCropDataset(DroidDatasetBase):
    def __init__(self, df, chunk_size=5, random_crop=True, verb_to_id=None):
        super().__init__(df, verb_to_id=verb_to_id)
        self.chunk_size = chunk_size
        self.random_crop = random_crop

    def __getitem__(self, idx):
        actions = self._get_actions(idx)
        T = actions.shape[0]
        if T >= self.chunk_size:
            if self.random_crop:
                start = np.random.randint(0, T - self.chunk_size + 1)
            else:
                start = max((T - self.chunk_size) // 2, 0)
            chunk = actions[start:start + self.chunk_size]
        else:
            chunk = np.pad(actions, ((0, self.chunk_size - T), (0, 0)), mode='edge')
        return torch.tensor(chunk.reshape(-1).tolist(), dtype=torch.float32)


class DroidTokenizerDataset(DroidDatasetBase):
    def __init__(self, df, window_size=5, max_windows=16,
                 include_instruction=False, verb_to_id=None,
                 return_format="tuple", sample_windows=False):
        super().__init__(df, verb_to_id=verb_to_id)
        self.window_size = window_size
        self.max_windows = max_windows
        self.include_instruction = include_instruction
        self.return_format = return_format
        self.sample_windows = sample_windows

    def __getitem__(self, idx):
        actions = self._get_actions(idx)
        padded_windows, n_windows = self._chunk_actions(
            actions, self.window_size, self.max_windows,
            sample_windows=self.sample_windows,
        )
        action_out = torch.tensor(padded_windows.tolist(), dtype=torch.float32)

        if self.return_format == "dict":
            out = {
                "action": action_out,
                "verb_label": torch.tensor(self._get_verb_id(idx), dtype=torch.long),
                "n_windows": torch.tensor(n_windows, dtype=torch.long),
            }
            if self.include_instruction:
                out["instruction"] = self._get_instruction(idx)
            return out

        result = (
            action_out,
            torch.tensor(self._get_verb_id(idx), dtype=torch.long),
        )
        if self.include_instruction:
            result = result + (self._get_instruction(idx),)
        return result + (torch.tensor(n_windows, dtype=torch.long),)


def load_bridge_actions(shard_dir):
    """Load all BridgeV2 action trajectories from shards."""
    shard_files = sorted(_glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    print(f"Loading {len(shard_files)} action shards...")
    actions_list = []
    for sf in tqdm(shard_files, desc="Loading shards"):
        data = np.load(sf, allow_pickle=True)
        for i in range(int(data["n_episodes"])):
            actions_list.append(data[f"actions_{i}"].astype(np.float32))
    print(f"Loaded {len(actions_list)} episodes")
    return actions_list


def fit_bridge_normalizer(actions_list):
    """Fit LinearNormalizer on BridgeV2 actions."""
    from oat.model.common.normalizer import LinearNormalizer
    all_actions = np.concatenate(actions_list, axis=0)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": all_actions}, mode="limits")
    return normalizer


def fit_droid_normalizer(df, max_episodes=2000):
    """Fit LinearNormalizer on sampled DROID actions."""
    from oat.model.common.normalizer import LinearNormalizer

    sample_df = df
    if max_episodes and len(df) > max_episodes:
        sample_df = df.sample(n=max_episodes, random_state=42).reset_index(drop=True)

    actions_list = []
    for row in tqdm(sample_df.itertuples(index=False), total=len(sample_df), desc="Fitting DROID normalizer"):
        shard = _load_droid_shard_cached(row.shard_path)
        actions = shard[f"actions_{int(row.episode_idx)}"].astype(np.float32)
        actions_list.append(actions)

    all_actions = np.concatenate(actions_list, axis=0)
    all_actions_t = torch.tensor(all_actions.tolist(), dtype=torch.float32)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": all_actions_t}, mode="limits")
    return normalizer


# ======================================================================
# Lazy imports (heavy deps)
# ======================================================================

def _import_vqbet():
    from tokenization.vqbet_tokenizer import VQBeTTokenizer
    return VQBeTTokenizer

def _import_oat():
    from oat.tokenizer.oat.tokenizer import OATTok
    from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
    from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
    from oat.tokenizer.oat.quantizer.fsq import FSQ
    return OATTok, RegisterEncoder, SinglePassDecoder, FSQ

def _import_quest():
    from oat.tokenizer.quest.tokenizer import QueSTTok
    return QueSTTok

def _import_fast():
    from tokenization.fast_tokenizer import FASTTokenizer, collect_trajectories
    return FASTTokenizer, collect_trajectories

# ======================================================================
# Text encoder and LoRA (for CLIP contrastive head)
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
        B, S, _ = tokens.shape
        x = self.input_proj(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :S + 1, :]
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x[:, 0, :])


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

    def forward(self, text_list):
        device = next(self.text_model.parameters()).device
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


# ======================================================================
# Auxiliary heads
# ======================================================================

class VerbHead(nn.Module):
    """Transformer + CLS token verb classifier on token sequences.

    Uses the same ActionTransformer (with CLS token) as ContrastiveHead,
    followed by an MLP classifier.  This preserves temporal ordering
    information that mean-pooling would discard.
    """

    def __init__(self, latent_dim, num_verbs, d_model=128, nhead=4,
                 num_layers=1, dropout=0.1, max_windows=16):
        super().__init__()
        self.action_transformer = ActionTransformer(
            input_dim=latent_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, max_len=max_windows)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_verbs),
        )

    def forward(self, window_latents, n_windows):
        """
        Args:
            window_latents: (B, max_w, latent_dim) pre-VQ encoder tokens
            n_windows: (B,) real token counts per trajectory
        Returns:
            logits: (B, num_verbs)
        """
        B, max_w, _ = window_latents.shape
        device = window_latents.device
        pad_mask = torch.arange(max_w, device=device).unsqueeze(0) >= \
            n_windows.unsqueeze(1)
        cls_out = self.action_transformer(window_latents, mask=pad_mask)
        return self.classifier(cls_out)


class ContrastiveHead(nn.Module):
    """Action transformer + projection for contrastive alignment."""

    def __init__(self, latent_dim=128, d_model=128, nhead=4,
                 transformer_layers=2, proj_dim=128, dropout=0.1,
                 max_windows=16):
        super().__init__()
        self.action_transformer = ActionTransformer(
            input_dim=latent_dim, d_model=d_model, nhead=nhead,
            num_layers=transformer_layers, dropout=dropout,
            max_len=max_windows)
        self.action_proj = nn.Linear(d_model, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=20.0)

    def forward(self, window_latents, n_windows):
        B, max_w, D = window_latents.shape
        device = window_latents.device
        pad_mask = torch.arange(max_w, device=device).unsqueeze(0) >= \
            n_windows.unsqueeze(1)
        cls_out = self.action_transformer(window_latents, mask=pad_mask)
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
# Tokenizer builders
# ======================================================================

def build_vqbet(args):
    VQBeTTokenizer = _import_vqbet()
    model = VQBeTTokenizer(
        action_dim=ACTION_DIM, chunk_size=args.chunk_size,
        latent_dim=args.latent_dim, n_embed=args.num_codes,
        groups=args.vq_groups, hidden_dim=args.hidden_dim,
        num_layers=args.num_mlp_layers)
    return model


def build_oat(args):
    OATTok, RegisterEncoder, SinglePassDecoder, FSQ = _import_oat()
    levels = getattr(args, 'fsq_levels', [8, 5, 5, 5])
    num_registers = getattr(args, 'num_registers', OAT_NUM_REGISTERS)
    latent_dim = len(levels)
    horizon = args.horizon
    enc = RegisterEncoder(
        sample_dim=ACTION_DIM, sample_horizon=horizon,
        emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
        latent_dim=latent_dim, num_registers=num_registers)
    dec = SinglePassDecoder(
        sample_dim=ACTION_DIM, sample_horizon=horizon,
        emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
        token_dropout_mode="pow2", latent_dim=latent_dim,
        latent_horizon=num_registers, use_causal_decoder=True)
    q = FSQ(levels=levels)
    tok = OATTok(encoder=enc, decoder=dec, quantizer=q)
    return tok


def build_quest(args):
    QueSTTok = _import_quest()
    levels = getattr(args, 'fsq_levels', [8, 5, 5, 5])
    ds = getattr(args, 'downsample_factor', TOKENIZER_DOWNSAMPLE_FACTOR)
    vq_type = getattr(args, 'vq_type', 'fsq')
    tok = QueSTTok(
        action_dim=ACTION_DIM, horizon=args.horizon,
        vq_type=vq_type, fsq_level=levels,
        vq_codebook_size=getattr(args, 'vq_codebook_size', 256),
        vq_codebook_dim=getattr(args, 'vq_codebook_dim', 256),
        downsample_factor=ds)
    return tok


# ======================================================================
# Latent extraction (tokenizer-agnostic)
# ======================================================================

def extract_latents_vqbet(model, chunks, n_chunks):
    """Extract post-VQ latents from VQ-BeT.

    Aux heads (verb, CLIP) operate on z_q — the quantized codebook vectors.
    This measures whether the discrete codes retain verb/language information.
    ResidualVQ uses straight-through estimator, so aux-head gradients flow
    back through z_q to the encoder for joint training.

    Args:
        model: VQBeTTokenizer
        chunks: (B, max_chunks, chunk_dim) from CalvinTokenizerDataset
        n_chunks: (B,) real chunk counts
    Returns:
        dict with recon_loss, vq_loss, traj_latents (B, max_chunks, latent_dim),
        real_counts (B,)
    """
    B = chunks.size(0)
    device = chunks.device

    # Flatten real chunks
    all_chunks = []
    counts = []
    for i in range(B):
        nc = n_chunks[i].item()
        all_chunks.append(chunks[i, :nc])  # (nc, chunk_dim)
        counts.append(nc)
    all_flat = torch.cat(all_chunks, dim=0)  # (total, chunk_dim)

    # Reshape: (total, window_size, action_dim) -> (total, window_size * action_dim)
    if all_flat.ndim == 3:
        all_flat = all_flat.reshape(all_flat.size(0), -1)

    # VQBeTTokenizer: forward returns (recon, recon_loss, vq_loss)
    recon, recon_loss, vq_loss = model(all_flat)

    # Post-VQ latents for aux heads (second encoder+VQ pass, with grad)
    # ResidualVQ uses STE so gradients flow back to encoder
    z = model.encoder(all_flat)
    z_unsq = z.unsqueeze(1)
    quantized, indices, _ = model.vq_layer(z_unsq)
    z_q = quantized.squeeze(1)
    # indices: (B, 1, groups) -> (B, groups) for codebook tracking
    codes = indices.squeeze(1).detach()

    # Repack per trajectory
    max_nc = max(counts)
    traj_latents = torch.zeros(B, max_nc, z_q.size(-1), device=device)
    offset = 0
    for i, nc in enumerate(counts):
        traj_latents[i, :nc] = z_q[offset:offset + nc]
        offset += nc

    return {
        'recon_loss': recon_loss,
        'vq_loss': vq_loss,
        'traj_latents': traj_latents,
        'real_counts': n_chunks,
        'codes': codes,  # (total_chunks, groups)
    }


def extract_latents_oat_quest(model, batch, device, pre_fsq=False):
    """Extract latents from OAT or QueST tokenizer for aux heads.

    Args:
        pre_fsq: If True and model is QueST, use pre-FSQ 256-d encoder output
                 instead of post-FSQ 4-d codes. Gives aux heads a richer
                 representation with full gradient flow (no FSQ bottleneck).

    Returns:
        dict with recon_loss, traj_latents (B, n_latent_tokens, latent_dim),
        real_counts.

    Latent dim:
        post-FSQ (default): OAT=4, QueST=4
        pre-FSQ (pre_fsq=True, QueST only): 256
    """
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    # CalvinTokenizerDataset returns (B, max_windows, ws, D); squeeze to (B, ws, D)
    if batch['action'].ndim == 4:
        batch['action'] = batch['action'].squeeze(1)
    # Forward pass for recon loss (encoder + quantizer + decoder)
    vq_loss = torch.tensor(0.0, device=device)
    if getattr(model, 'vq_type', 'fsq') == 'vq' and hasattr(model, 'encode') and hasattr(model, 'decode'):
        latents_q, indices, commit_loss = model.encode(batch['action'])
        recon = model.decode(latents_q)
        recon_loss = F.mse_loss(recon, batch['action'])
        vq_loss = commit_loss
    else:
        recon_loss = model(batch)

    # Second encoder pass WITH grad for aux heads.
    codes = None
    use_fsq_codes = (hasattr(model, 'encode_fsq_codes')
                     and getattr(model, 'vq_type', 'fsq') == 'fsq')

    if pre_fsq and hasattr(model, 'encode_pre_fsq'):
        # Pre-FSQ: 256-d encoder output before projection + FSQ.
        # Works for both QueST (transformer encoder output) and OAT (register embeddings).
        latents = model.encode_pre_fsq(batch['action'])  # (B, T', 256)
    elif use_fsq_codes:
        # QueST post-FSQ: 4-d quantized codes with STE gradient
        latents = model.encode_fsq_codes(batch['action'])  # (B, T', 4)
        with torch.no_grad():
            codes = model.vq.codes_to_indices(latents).unsqueeze(-1)
    else:
        # OAT post-FSQ (4-d) or QueST vq_type='vq' (256-d post-VQ)
        encoded = model.encode(batch['action'])
        if isinstance(encoded, tuple):
            latents = encoded[0]
            if len(encoded) > 1:
                codes = encoded[1].detach()
                if codes.ndim == 2:
                    codes = codes.unsqueeze(-1)
        elif isinstance(encoded, dict):
            latents = encoded.get('latents', encoded.get('state', None))
        else:
            latents = encoded

    if latents is None:
        return {'recon_loss': recon_loss, 'traj_latents': None, 'real_counts': None}

    B = latents.size(0)
    n_tokens = latents.size(1) if latents.ndim == 3 else 1
    if latents.ndim == 2:
        latents = latents.unsqueeze(1)

    return {
        'recon_loss': recon_loss,
        'vq_loss': vq_loss,
        'traj_latents': latents,
        'real_counts': torch.full((B,), n_tokens, dtype=torch.long, device=device),
        'codes': codes,  # (B, T', D) FSQ codes or None
    }



# ======================================================================
# Training loop
# ======================================================================

def _get_batch_field(batch, key, index=None):
    """Get a field from a batch that could be a dict or tuple."""
    if isinstance(batch, dict):
        return batch.get(key)
    if index is not None:
        return batch[index]
    return None


def train_epoch(model, loader, optimizer, device, args,
                extract_fn, verb_head=None, verb_criterion=None,
                clip_head=None, text_encoder=None, text_proj=None):
    model.train()
    if verb_head is not None:
        verb_head.train()
    if clip_head is not None:
        clip_head.train()

    totals = {'recon': 0, 'vq': 0, 'verb': 0, 'clip': 0}
    correct = total = 0
    all_preds, all_labels = [], []
    n_batches = 0

    for batch in loader:
        result = extract_fn(model, batch, device)
        loss = result['recon_loss']
        if 'vq_loss' in result:
            loss = loss + args.vq_weight * result['vq_loss']

        # Verb head
        if verb_head is not None and args.verb_cls_lambda > 0 and result['traj_latents'] is not None:
            verb_logits = verb_head(result['traj_latents'],
                                    result['real_counts'].to(device))
            verb_ids = _get_batch_field(batch, 'verb_label', 1).to(device)
            verb_loss = verb_criterion(verb_logits, verb_ids)
            loss = loss + args.verb_cls_lambda * verb_loss
            totals['verb'] += verb_loss.item()
            preds = verb_logits.argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(verb_ids.cpu())

        # CLIP head
        if clip_head is not None and args.clip_lambda > 0 and result['traj_latents'] is not None:
            action_emb = clip_head(result['traj_latents'],
                                   result['real_counts'].to(device))
            instructions = _get_batch_field(batch, 'instruction', 2)
            with torch.set_grad_enabled(text_encoder.lora_r > 0):
                text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)
            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions), clip_head.temperature)
            loss = loss + args.clip_lambda * clip_loss
            totals['clip'] += clip_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            all_params = [p for p in model.parameters() if p.requires_grad]
            if verb_head is not None:
                all_params += list(verb_head.parameters())
            if clip_head is not None:
                all_params += list(clip_head.parameters())
                all_params += list(text_proj.parameters())
                if text_encoder.lora_r > 0:
                    all_params += [p for p in text_encoder.parameters()
                                   if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
        optimizer.step()

        totals['recon'] += result['recon_loss'].item()
        totals['vq'] += result.get('vq_loss', torch.tensor(0)).item()
        n_batches += 1

    macro_f1 = 0.0
    if all_preds:
        macro_f1 = 100.0 * f1_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
            average='macro', zero_division=0)
    return {k: v / max(n_batches, 1) for k, v in totals.items()} | {
        'verb_acc': 100.0 * correct / max(total, 1),
        'verb_macro_f1': macro_f1,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, args,
               extract_fn, verb_head=None, verb_criterion=None,
               clip_head=None, text_encoder=None, text_proj=None):
    model.eval()
    if verb_head is not None:
        verb_head.eval()
    if clip_head is not None:
        clip_head.eval()

    totals = {'recon': 0, 'vq': 0, 'verb': 0, 'clip': 0}
    correct = total = 0
    all_preds, all_labels = [], []
    all_codes = []
    n_batches = 0

    for batch in loader:
        result = extract_fn(model, batch, device)

        if verb_head is not None and args.verb_cls_lambda > 0 and result['traj_latents'] is not None:
            verb_logits = verb_head(result['traj_latents'],
                                    result['real_counts'].to(device))
            verb_ids = _get_batch_field(batch, 'verb_label', 1).to(device)
            verb_loss = verb_criterion(verb_logits, verb_ids)
            totals['verb'] += verb_loss.item()
            preds = verb_logits.argmax(dim=1)
            correct += (preds == verb_ids).sum().item()
            total += verb_ids.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(verb_ids.cpu())

        if clip_head is not None and args.clip_lambda > 0 and result['traj_latents'] is not None:
            action_emb = clip_head(result['traj_latents'],
                                   result['real_counts'].to(device))
            instructions = _get_batch_field(batch, 'instruction', 2)
            text_features = text_encoder(list(instructions))
            text_emb = F.normalize(text_proj(text_features), dim=-1)
            clip_loss = contrastive_loss(
                action_emb, text_emb, list(instructions), clip_head.temperature)
            totals['clip'] += clip_loss.item()

        # Collect discrete codes for utilization tracking
        if result.get('codes') is not None:
            codes = result['codes']
            if codes.ndim == 3:
                # (B, T, D) -> (B*T, D) — flatten token positions
                codes = codes.reshape(-1, codes.size(-1))
            # (B*T, D) or (B, groups) — each row is one code tuple
            all_codes.append(codes.cpu())

        totals['recon'] += result['recon_loss'].item()
        totals['vq'] += result.get('vq_loss', torch.tensor(0)).item()
        n_batches += 1

    macro_f1 = 0.0
    if all_preds:
        macro_f1 = 100.0 * f1_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),
            average='macro', zero_division=0)

    # Compute codebook utilization
    from analysis.codebook_util import codes_to_unique_count
    codebook_util = codes_to_unique_count(all_codes)

    return {k: v / max(n_batches, 1) for k, v in totals.items()} | {
        'verb_acc': 100.0 * correct / max(total, 1),
        'verb_macro_f1': macro_f1,
        'codebook_util': codebook_util,
    }


@torch.no_grad()
def eval_clip_retrieval(model, loader, device, extract_fn,
                        clip_head, text_encoder, text_proj, ks=(1, 5, 10)):
    """Action→text and text→action top-k retrieval on the full val set.

    Iterates the val loader (shuffle=False) once, collecting action and text
    embeddings, then computes recall@k using cosine similarity.

    Returns dict: {'r@1': float, 'r@5': float, 'r@10': float}  (percentages)
    """
    model.eval()
    clip_head.eval()

    all_action_emb, all_text_emb = [], []

    for batch in loader:
        result = extract_fn(model, batch, device)
        if result['traj_latents'] is None:
            continue
        action_emb = clip_head(result['traj_latents'],
                               result['real_counts'].to(device))  # (B, proj_dim)
        instructions = _get_batch_field(batch, 'instruction', 2)
        text_features = text_encoder(list(instructions))
        text_emb = F.normalize(text_proj(text_features), dim=-1)

        all_action_emb.append(action_emb.cpu())
        all_text_emb.append(text_emb.cpu())

    if not all_action_emb:
        return {f'r@{k}': 0.0 for k in ks}

    A = torch.cat(all_action_emb, 0)  # (N, proj_dim)
    T = torch.cat(all_text_emb,   0)  # (N, proj_dim)
    N = A.shape[0]

    # Cosine similarity matrix (N, N)
    sim = A @ T.t()  # both already l2-normalised by contrastive_loss / clip_head

    results = {}
    for k in ks:
        k_clamped = min(k, N)
        # Action → text retrieval
        topk_idx = sim.topk(k_clamped, dim=1).indices          # (N, k)
        gt = torch.arange(N).unsqueeze(1)                       # (N, 1)
        hit_a2t = (topk_idx == gt).any(dim=1).float().mean().item()
        # Text → action retrieval
        topk_idx_t = sim.t().topk(k_clamped, dim=1).indices    # (N, k)
        hit_t2a = (topk_idx_t == gt).any(dim=1).float().mean().item()
        results[f'r@{k}'] = 100.0 * (hit_a2t + hit_t2a) / 2.0

    return results


# ======================================================================
# Extract function wrappers (adapt each tokenizer to uniform interface)
# ======================================================================

def make_extract_fn(tok_type, model, pre_fsq=False):
    """Return an extract_fn(model, batch, device) for the given tokenizer type.

    Args:
        pre_fsq: If True, QueST aux heads use 256-d pre-FSQ encoder output
                 instead of 4-d post-FSQ codes. Ignored for VQ-BeT and OAT.
    """

    if tok_type == 'vq_bet':
        def fn(model, batch, device):
            if isinstance(batch, torch.Tensor):
                # Bridge flat chunks: (B, chunk_size * action_dim)
                x = batch.to(device)
                _, recon_loss, vq_loss = model(x)
                # Get codebook indices for utilization tracking
                with torch.no_grad():
                    _, indices, _ = model.encode(x)  # (B, groups)
                return {'recon_loss': recon_loss, 'vq_loss': vq_loss,
                        'traj_latents': None, 'real_counts': None,
                        'codes': indices.detach()}
            windows = batch[0].to(device)     # (B, max_windows, window_size, action_dim)
            n_windows = batch[-1]             # last element is always n_windows
            return extract_latents_vqbet(model, windows, n_windows)
        return fn

    if tok_type in ('oat', 'quest'):
        def fn(model, batch, device):
            if isinstance(batch, dict):
                action = batch["action"].to(device)
                if action.ndim == 4:
                    action = action.squeeze(1)
                action_dict = {"action": action}
                return extract_latents_oat_quest(model, action_dict, device, pre_fsq=pre_fsq)
            return extract_latents_oat_quest(model, {k: v.to(device) for k, v in batch.items()}, device, pre_fsq=pre_fsq)
        return fn

    raise ValueError(f"Unknown tokenizer type: {tok_type}")


# ======================================================================
# Normalizer fitting (for OAT/QueST)
# ======================================================================

def fit_normalizer(data_dir, max_trajs=2000):
    """Fit oat LinearNormalizer on CALVIN actions."""
    from oat.model.common.normalizer import LinearNormalizer
    from analysis.cluster_analysis import load_all_actions

    df = load_calvin_to_dataframe(data_dir)
    if max_trajs:
        df = df.head(min(max_trajs, len(df))).copy()
    all_actions, _ = load_all_actions(df, num_workers=8)
    actions_t = torch.from_numpy(all_actions)

    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions_t}, last_n_dims=1, mode="limits",
                   output_min=-1.0, output_max=1.0)
    return normalizer


# ======================================================================
# FAST fitting (non-gradient)
# ======================================================================

def fit_fast(args):
    """Fit FAST tokenizer (DCT + BPE). No gradient training."""
    FASTTokenizer, collect_trajectories = _import_fast()

    if args.dataset == "droid":
        raise ValueError("FAST fitting is currently only wired for CALVIN in this script")

    train_df = load_calvin_to_dataframe(args.data_dir)
    trajectories = collect_trajectories(train_df, args.data_dir)
    print(f"Collected {len(trajectories)} trajectories for FAST fitting")

    tok = FASTTokenizer.fit(trajectories, scale=args.fast_scale,
                            vocab_size=args.fast_vocab_size)

    save_dir = args.save_dir or os.path.join("checkpoints", f"fast_s{args.fast_scale}_v{args.fast_vocab_size}")
    os.makedirs(save_dir, exist_ok=True)
    tok.save(os.path.join(save_dir, "fast_tokenizer"))
    print(f"FAST tokenizer saved to {save_dir}")
    return tok


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified action tokenizer training")

    # Core
    parser.add_argument("--tokenizer", type=str, required=True,
                        choices=["vq_vae", "vq_bet", "vqvla", "oat", "quest", "fast", "bin"])
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix appended to auto-generated run name")
    parser.add_argument("--dataset", type=str, default="calvin",
                        choices=["calvin", "bridge", "droid"],
                        help="Dataset to train on (default: calvin)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--shard_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/bridge_actions",
                        help="BridgeV2 action shard directory (only used with --dataset bridge)")
    parser.add_argument("--droid_actions_dir", type=str, default=DROID_ACTIONS_DIR,
                        help="DROID action shard directory (used with --dataset droid)")
    parser.add_argument("--droid_metadata_cache", type=str, default=None,
                        help="Optional CSV cache path for DROID metadata")
    parser.add_argument("--rebuild_droid_metadata", action="store_true",
                        help="Rebuild the cached DROID metadata CSV from shard files")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Optional cap on number of DROID shards scanned when building metadata")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of episodes for validation (Bridge/DROID only)")
    parser.add_argument("--max_train_episodes", type=int, default=None,
                        help="Optional cap on training episodes after split (useful for smoke tests)")
    parser.add_argument("--max_val_episodes", type=int, default=None,
                        help="Optional cap on validation episodes after split (useful for smoke tests)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vq_weight", type=float, default=5.0,
                        help="Weight for VQ loss (default 5.0 for VQ-VLA, 1.0 for VQ-VAE)")

    # Tokenizer-specific
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="VQ-VAE chunk size")
    parser.add_argument("--num_codes", type=int, default=512,
                        help="VQ-VAE codebook size")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="VQ-VAE latent dimension")
    parser.add_argument("--vq_groups", type=int, default=4,
                        help="Number of residual quantizer groups (VQ-BeT)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="MLP hidden dim (VQ-BeT)")
    parser.add_argument("--num_mlp_layers", type=int, default=1,
                        help="Number of hidden layers in encoder/decoder MLP (VQ-BeT)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Window size for chunked datasets (VQ-VLA=5)")
    parser.add_argument("--max_windows", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=TOKENIZER_HORIZON,
                        help="Action horizon for OAT/QueST")
    parser.add_argument("--fsq_levels", type=int, nargs='+', default=[8, 5, 5, 5],
                        help="FSQ quantization levels for OAT/QueST (default: 8 5 5 5 = 1000 codes)")
    parser.add_argument("--num_registers", type=int, default=OAT_NUM_REGISTERS,
                        help="Number of register tokens for OAT (default: 8)")
    parser.add_argument("--downsample_factor", type=int, default=TOKENIZER_DOWNSAMPLE_FACTOR,
                        help="Temporal downsampling factor for QueST (default: 4)")
    parser.add_argument("--vq_type", type=str, default="fsq", choices=["fsq", "vq"],
                        help="QueST quantization type: fsq (default) or vq (learned codebook)")
    parser.add_argument("--vq_codebook_size", type=int, default=256,
                        help="Codebook size for QueST vq_type=vq (default: 256)")
    parser.add_argument("--vq_codebook_dim", type=int, default=256,
                        help="Codebook vector dimension for QueST vq_type=vq (default: 256)")
    parser.add_argument("--vqvla_config_dir", type=str, default=None)
    parser.add_argument("--vqvla_pretrained", type=str, default=None)

    # FAST-specific
    parser.add_argument("--fast_vocab_size", type=int, default=1024)
    parser.add_argument("--fast_scale", type=float, default=10.0)

    # Aux heads
    parser.add_argument("--verb_cls_lambda", type=float, default=0.0,
                        help="Lambda for verb classification head (0=disabled)")
    parser.add_argument("--clip_lambda", type=float, default=0.0,
                        help="Lambda for CLIP contrastive head (0=disabled)")
    parser.add_argument("--pre_fsq_aux", action="store_true",
                        help="QueST: attach aux heads to 256-d pre-FSQ encoder output "
                             "instead of 4-d post-FSQ codes")
    parser.add_argument("--min_class_count", type=int, default=30,
                        help="Min samples per verb class (sparse filtering)")
    parser.add_argument("--weighted_verb_loss", action="store_true", default=True)
    parser.add_argument("--no_weighted_verb_loss", dest="weighted_verb_loss",
                        action="store_false")

    # CLIP-specific
    parser.add_argument("--text_model", type=str,
                        default='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    parser.add_argument("--text_type", type=str, default='clip',
                        choices=['clip', 'gpt2'])
    parser.add_argument("--text_lora_r", type=int, default=0)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--clip_d_model", type=int, default=128)
    parser.add_argument("--clip_transformer_layers", type=int, default=2)

    # Checkpoint
    parser.add_argument("--max_chunks_per_epoch", type=int, default=None,
                        help="Subsample training chunks per epoch (for large datasets)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full.pth checkpoint to resume from")
    parser.add_argument("--freeze_tokenizer", action="store_true",
                        help="Freeze tokenizer weights; only train aux heads (verb/CLIP). "
                             "Use with --resume to probe a pretrained tokenizer.")

    args = parser.parse_args()

    # ── Handle non-gradient tokenizers ──────────────────────────────────
    if args.tokenizer == "fast":
        fit_fast(args)
        return

    if args.tokenizer == "bin":
        print("Bin tokenizer is analytical — no training needed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Tokenizer: {args.tokenizer}")

    # ── Load data ───────────────────────────────────────────────────────
    has_verb = args.verb_cls_lambda > 0
    has_clip = args.clip_lambda > 0
    include_instruction = has_clip

    train_df = None
    val_df = None

    if args.dataset == "bridge":
        # Bridge: load shards, split by episode
        all_actions = load_bridge_actions(args.shard_dir)
        np.random.seed(42)
        perm = np.random.permutation(len(all_actions))
        n_val = max(1, int(len(all_actions) * args.val_fraction))
        train_actions = [all_actions[i] for i in perm[n_val:]]
        val_actions = [all_actions[i] for i in perm[:n_val]]
        print(f"Train: {len(train_actions)} episodes, Val: {len(val_actions)} episodes")

        if args.tokenizer in ('oat', 'quest'):
            train_ds = BridgeActionChunkDataset(train_actions, horizon=args.horizon)
            val_ds = BridgeActionChunkDataset(val_actions, horizon=args.horizon)
        elif args.tokenizer in ('vq_vae', 'vq_bet'):
            train_ds = BridgeFlatChunkDataset(train_actions, chunk_size=args.chunk_size)
            val_ds = BridgeFlatChunkDataset(val_actions, chunk_size=args.chunk_size)
        else:
            raise ValueError(f"Bridge dataset not supported for tokenizer {args.tokenizer}")
    elif args.dataset == "droid":
        droid_df = load_droid_metadata(
            args.droid_actions_dir,
            metadata_cache=args.droid_metadata_cache,
            rebuild=args.rebuild_droid_metadata,
            max_shards=args.max_shards,
        )
        train_df, val_df = split_episode_dataframe(
            droid_df,
            val_fraction=args.val_fraction,
            seed=42,
            max_train_episodes=args.max_train_episodes,
            max_val_episodes=args.max_val_episodes,
        )
        print(f"Train: {len(train_df)} episodes, Val: {len(val_df)} episodes")

        if has_verb and args.min_class_count > 0:
            verb_col = 'primary_verb' if 'primary_verb' in train_df.columns else 'verb'
            verb_counts = train_df[verb_col].value_counts()
            keep_verbs = set(verb_counts[verb_counts >= args.min_class_count].index)
            train_df = train_df[train_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
            val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
            print(f"Filtered to {len(keep_verbs)} verb classes, {len(train_df)} train / {len(val_df)} val")

        if args.tokenizer in ('oat', 'quest') and not has_verb and not has_clip:
            train_ds = DroidActionCropDataset(train_df, horizon=args.horizon, random_crop=True)
            val_ds = DroidActionCropDataset(val_df, horizon=args.horizon, random_crop=False)
        elif args.tokenizer in ('oat', 'quest'):
            train_ds = DroidTokenizerDataset(
                train_df, window_size=args.horizon, max_windows=1,
                include_instruction=include_instruction, return_format="dict",
                sample_windows=True,
            )
            val_ds = DroidTokenizerDataset(
                val_df, window_size=args.horizon, max_windows=1,
                include_instruction=include_instruction, return_format="dict",
                verb_to_id=train_ds.verb_to_id, sample_windows=False,
            )
        elif args.tokenizer in ('vq_vae', 'vq_bet') and not has_verb and not has_clip:
            train_ds = DroidFlatCropDataset(train_df, chunk_size=args.chunk_size, random_crop=True)
            val_ds = DroidFlatCropDataset(val_df, chunk_size=args.chunk_size, random_crop=False)
        else:
            ws = args.window_size
            if args.tokenizer in ('vq_vae', 'vq_bet'):
                ws = args.chunk_size
            train_ds = DroidTokenizerDataset(
                train_df, window_size=ws, max_windows=args.max_windows,
                include_instruction=include_instruction, sample_windows=True,
            )
            val_ds = DroidTokenizerDataset(
                val_df, window_size=ws, max_windows=args.max_windows,
                include_instruction=include_instruction,
                verb_to_id=train_ds.verb_to_id, sample_windows=False,
            )
    else:
        # CALVIN: load dataframes
        train_df = load_calvin_to_dataframe(args.data_dir)
        val_df = load_calvin_to_dataframe(args.val_dir)

        if has_verb and args.min_class_count > 0:
            verb_col = 'primary_verb' if 'primary_verb' in train_df.columns else 'verb'
            verb_counts = train_df[verb_col].value_counts()
            keep_verbs = set(verb_counts[verb_counts >= args.min_class_count].index)
            train_df = train_df[train_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
            val_df = val_df[val_df[verb_col].isin(keep_verbs)].reset_index(drop=True)
            print(f"Filtered to {len(keep_verbs)} verb classes, "
                  f"{len(train_df)} train / {len(val_df)} val")

        if args.tokenizer in ('oat', 'quest') and not has_verb and not has_clip:
            train_ds = CalvinActionCropDataset(
                args.data_dir, train_df, horizon=args.horizon, cache_actions=True)
            val_ds = CalvinActionCropDataset(
                args.val_dir, val_df, horizon=args.horizon, cache_actions=True)
        elif args.tokenizer in ('oat', 'quest'):
            ws = args.horizon
            train_ds = CalvinTokenizerDataset(
                args.data_dir, train_df, window_size=ws,
                max_windows=1, cache_actions=True,
                include_instruction=include_instruction,
                return_format="dict")
            val_ds = CalvinTokenizerDataset(
                args.val_dir, val_df, window_size=ws,
                max_windows=1, cache_actions=True,
                include_instruction=include_instruction,
                verb_to_id=train_ds.verb_to_id,
                return_format="dict")
        else:
            ws = args.window_size
            if args.tokenizer in ('vq_vae', 'vq_bet'):
                ws = args.chunk_size
            train_ds = CalvinTokenizerDataset(
                args.data_dir, train_df, window_size=ws,
                max_windows=args.max_windows, cache_actions=True,
                include_instruction=include_instruction)
            val_ds = CalvinTokenizerDataset(
                args.val_dir, val_df, window_size=ws,
                max_windows=args.max_windows, cache_actions=True,
                include_instruction=include_instruction,
                verb_to_id=train_ds.verb_to_id)

    # ── Build datasets ──────────────────────────────────────────────────

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ── Fit normalizer ────────────────────────────────────────────────────
    if args.dataset == "bridge":
        normalizer = fit_bridge_normalizer(train_actions)
    elif args.dataset == "droid":
        normalizer = fit_droid_normalizer(train_df)
    else:
        normalizer = fit_normalizer(args.data_dir)

    # ── Build tokenizer model ───────────────────────────────────────────
    if args.tokenizer == 'vq_vae':
        if args.vq_weight == 5.0:
            args.vq_weight = 1.0  # default for VQ-VAE
        model = build_vqvae(args).to(device)
    elif args.tokenizer == 'vq_bet':
        if args.vq_weight == 5.0:
            args.vq_weight = 1.0  # default for VQ-BeT
        model = build_vqbet(args)
        model.set_normalizer(normalizer)
        model = model.to(device)
    elif args.tokenizer == 'vqvla':
        model = build_vqvla(args).to(device)
    elif args.tokenizer == 'oat':
        model = build_oat(args)
        model.set_normalizer(normalizer)
        model = model.to(device)
    elif args.tokenizer == 'quest':
        model = build_quest(args)
        model.set_normalizer(normalizer)
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tokenizer model: {n_params:,} trainable params")

    extract_fn = make_extract_fn(args.tokenizer, model,
                                   pre_fsq=getattr(args, 'pre_fsq_aux', False))

    # ── Build aux heads ─────────────────────────────────────────────────
    verb_head = None
    verb_criterion = None
    clip_head = None
    text_encoder = None
    text_proj = None

    # Determine latent dim for heads
    if args.tokenizer in ('vq_vae', 'vq_bet'):
        head_latent_dim = args.latent_dim
    elif args.tokenizer == 'vqvla':
        head_latent_dim = 128  # fixed by VQ-VLA architecture
    elif args.tokenizer in ('oat', 'quest'):
        if getattr(args, 'vq_type', 'fsq') == 'vq':
            head_latent_dim = 256  # VQ mode: quantization in encoder_dim space
        elif getattr(args, 'pre_fsq_aux', False):
            head_latent_dim = 256  # pre-FSQ: emb_dim (both OAT and QueST)
        else:
            head_latent_dim = 4  # post-FSQ: len(fsq_levels)
    else:
        head_latent_dim = 128

    if has_verb:
        num_verbs = len(train_ds.verb_to_id)
        verb_head = VerbHead(head_latent_dim, num_verbs,
                             max_windows=args.max_windows).to(device)
        print(f"Verb head: {head_latent_dim} -> CLS -> {num_verbs} classes")

        if args.weighted_verb_loss:
            verb_col = train_ds._verb_col
            class_counts = train_df[verb_col].value_counts()
            weights = torch.zeros(num_verbs)
            for verb, cid in train_ds.verb_to_id.items():
                weights[cid] = 1.0 / class_counts.get(verb, 1)
            weights = weights / weights.sum() * num_verbs
            verb_criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            verb_criterion = nn.CrossEntropyLoss()

    if has_clip:
        clip_head = ContrastiveHead(
            latent_dim=head_latent_dim, d_model=args.clip_d_model,
            nhead=4, transformer_layers=args.clip_transformer_layers,
            proj_dim=args.proj_dim, max_windows=args.max_windows,
        ).to(device)
        text_encoder = TextEncoderWrapper(
            model_name=args.text_model, model_type=args.text_type,
            freeze=(args.text_lora_r == 0), lora_r=args.text_lora_r,
        ).to(device)
        text_proj = nn.Linear(text_encoder.output_dim, args.proj_dim).to(device)
        print(f"CLIP head: latent_dim={head_latent_dim}, proj_dim={args.proj_dim}")

    # ── Freeze tokenizer if requested ──────────────────────────────────
    if args.freeze_tokenizer:
        for p in model.parameters():
            p.requires_grad = False
        print("Tokenizer frozen — only training aux heads")

    # ── Optimizer ───────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    if verb_head is not None:
        params += list(verb_head.parameters())
    if clip_head is not None:
        params += list(clip_head.parameters())
        params += list(text_proj.parameters())
        if text_encoder.lora_r > 0:
            params += [p for p in text_encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 0
    best_metric = float('inf')
    best_verb_acc = 0.0
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict',
                                       ckpt.get('vqvae_state_dict', {})))
        if args.freeze_tokenizer:
            # Probe mode: only load tokenizer weights, start aux heads fresh
            print("Probe mode: tokenizer loaded & frozen, aux heads initialized fresh")
        else:
            # Full resume: restore everything
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if verb_head is not None and 'verb_head_state_dict' in ckpt:
                verb_head.load_state_dict(ckpt['verb_head_state_dict'])
            if clip_head is not None and 'clip_head_state_dict' in ckpt:
                clip_head.load_state_dict(ckpt['clip_head_state_dict'])
            if text_proj is not None and 'text_proj_state_dict' in ckpt:
                text_proj.load_state_dict(ckpt['text_proj_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_metric = ckpt.get('best_metric', float('inf'))
            best_verb_acc = ckpt.get('best_verb_acc', 0.0)
            print(f"Resumed at epoch {start_epoch}")

    # ── Output directory (auto-generated name) ─────────────────────────
    run_name = args.tokenizer
    if args.verb_cls_lambda > 0:
        run_name += f"_verb{args.verb_cls_lambda}"
    if args.clip_lambda > 0:
        run_name += f"_clip{args.clip_lambda}"
    if args.tag:
        run_name += f"_{args.tag}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "a" if args.resume else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not args.resume:
        header = ["epoch", "train_recon", "train_vq",
                  "train_verb", "train_verb_acc", "train_verb_macro_f1",
                  "train_clip",
                  "val_recon", "val_vq",
                  "val_verb", "val_verb_acc", "val_verb_macro_f1",
                  "val_clip", "val_r1", "val_r5", "val_r10",
                  "lr", "time"]
        csv_writer.writerow(header)

    # ── Training ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs")
    print(f"  verb_cls_lambda={args.verb_cls_lambda}, clip_lambda={args.clip_lambda}")
    print(f"  vq_weight={args.vq_weight}")
    print(f"  Save dir: {save_dir}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        t0 = time_mod.time()

        # Subsample training data if requested
        if args.max_chunks_per_epoch and len(train_ds) > args.max_chunks_per_epoch:
            from torch.utils.data import Subset
            subset_idx = np.random.choice(len(train_ds), args.max_chunks_per_epoch, replace=False)
            epoch_loader = DataLoader(Subset(train_ds, subset_idx),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        else:
            epoch_loader = train_loader

        train_m = train_epoch(
            model, epoch_loader, optimizer, device, args, extract_fn,
            verb_head=verb_head, verb_criterion=verb_criterion,
            clip_head=clip_head, text_encoder=text_encoder, text_proj=text_proj)

        val_m = eval_epoch(
            model, val_loader, device, args, extract_fn,
            verb_head=verb_head, verb_criterion=verb_criterion,
            clip_head=clip_head, text_encoder=text_encoder, text_proj=text_proj)

        # CLIP retrieval metrics (full val set pass, no grad)
        retrieval = {}
        if has_clip:
            retrieval = eval_clip_retrieval(
                model, val_loader, device, extract_fn,
                clip_head, text_encoder, text_proj, ks=(1, 5, 10))

        scheduler.step()
        dt = time_mod.time() - t0

        # Print
        line = f"Epoch {epoch+1:3d}/{args.epochs} ({dt:.1f}s)"
        line += f" | train: recon={train_m['recon']:.5f} vq={train_m['vq']:.5f}"
        if has_verb:
            line += f" verb={train_m['verb']:.4f} acc={train_m['verb_acc']:.1f}% mF1={train_m['verb_macro_f1']:.1f}%"
        if has_clip:
            line += f" clip={train_m['clip']:.4f}"
        line += f" | val: recon={val_m['recon']:.5f} vq={val_m['vq']:.5f}"
        if has_verb:
            line += f" verb={val_m['verb']:.4f} acc={val_m['verb_acc']:.1f}% mF1={val_m['verb_macro_f1']:.1f}%"
        if has_clip:
            line += f" clip={val_m['clip']:.4f}"
            line += f" R@1={retrieval.get('r@1', 0):.1f}% R@5={retrieval.get('r@5', 0):.1f}%"
        if val_m.get('codebook_util') is not None:
            line += f" | codes={val_m['codebook_util']}"
        print(line)

        # Best checkpoint: monitor total weighted val loss (lower = better)
        val_total = val_m['recon'] + args.vq_weight * val_m['vq']
        if has_verb:
            val_total += args.verb_cls_lambda * val_m['verb']
        if has_clip:
            val_total += args.clip_lambda * val_m['clip']
        is_best = val_total < best_metric
        if is_best:
            best_metric = val_total
            if has_verb:
                best_verb_acc = val_m['verb_acc']

        if is_best:
            patience_counter = 0
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_m,
                'val_metrics': val_m,
                'args': vars(args),
                'best_metric': best_metric,
                'best_verb_acc': best_verb_acc,
            }
            if has_verb:
                ckpt['verb_head_state_dict'] = verb_head.state_dict()
                ckpt['verb_to_id'] = train_ds.verb_to_id
                ckpt['id_to_verb'] = train_ds.id_to_verb
            if has_clip:
                ckpt['clip_head_state_dict'] = clip_head.state_dict()
                ckpt['text_proj_state_dict'] = text_proj.state_dict()

            # Save tokenizer weights (compatible with load_action_tokenizer)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "tokenizer_weights.pth"))
            # Save full checkpoint for resume
            torch.save(ckpt, os.path.join(save_dir, "full.pth"))
            print(f"  -> Saved best checkpoint (epoch {epoch+1})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"({patience_counter} epochs without improvement)")
                break

        # CSV
        current_lr = optimizer.param_groups[0]['lr']
        csv_writer.writerow([
            epoch + 1,
            f"{train_m['recon']:.6f}", f"{train_m['vq']:.6f}",
            f"{train_m['verb']:.6f}", f"{train_m['verb_acc']:.2f}",
            f"{train_m['verb_macro_f1']:.2f}",
            f"{train_m['clip']:.6f}",
            f"{val_m['recon']:.6f}", f"{val_m['vq']:.6f}",
            f"{val_m['verb']:.6f}", f"{val_m['verb_acc']:.2f}",
            f"{val_m['verb_macro_f1']:.2f}",
            f"{val_m['clip']:.6f}",
            f"{retrieval.get('r@1', 0.0):.2f}",
            f"{retrieval.get('r@5', 0.0):.2f}",
            f"{retrieval.get('r@10', 0.0):.2f}",
            f"{current_lr:.8f}", f"{dt:.1f}",
        ])
        csv_file.flush()

    csv_file.close()

    # Save config
    config = {
        'dataset': args.dataset,
        'tokenizer': args.tokenizer,
        'run_name': run_name,
        'verb_cls_lambda': args.verb_cls_lambda,
        'clip_lambda': args.clip_lambda,
        'epochs_run': epoch + 1,
        'best_metric': float(best_metric),
        'best_verb_acc': float(best_verb_acc),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
