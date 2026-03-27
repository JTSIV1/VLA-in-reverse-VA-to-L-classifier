"""
calvin_sweep_action_tokenizer.py

Drop-in ActionTokenizer for VQ-BeT / OAT / QueST tokenizers from the CALVIN-D
tokenizer sweep. Each tokenizer produces a sequence of discrete codes per action
chunk; these codes are mapped to the last N tokens of the LLM vocabulary.

Supported tokenizers and their code structure:
  - VQ-BeT:  ResidualVQ with `groups` codebooks, each of size `n_embed`.
             One chunk → `groups` integer codes. (e.g., 2 codes, each 0-15)
  - OAT:    FSQ with register tokens. One chunk → `n_registers` tokens,
             each a 4-dim integer vector → flattened to single index.
  - QueST:  FSQ with causal conv downsampling. One chunk → `horizon//downsample`
             tokens, each a 4-dim integer vector → flattened to single index.

Usage:
    from prismatic.vla.calvin_sweep_action_tokenizer import CalvinSweepActionTokenizer

    tokenizer = CalvinSweepActionTokenizer(
        base_tokenizer,
        tokenizer_type="vq_bet",  # or "oat", "quest"
        checkpoint_path="path/to/full.pth",
        device="cuda",
    )
"""

import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from prismatic.vla.action_tokenizer import ActionTokenizer

# Use current runtime sys.path for imports.

def _load_tokenizer_model(tokenizer_type, checkpoint_path, device="cpu"):
    """Load a trained tokenizer from a sweep checkpoint.

    Returns (model, chunk_size, n_codes_per_chunk, codebook_size, fsq_levels).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})

    if tokenizer_type == "vq_bet":
        from tokenization.vqbet_tokenizer import VQBeTTokenizer
        from tokenization.train_tokenizer import fit_normalizer

        chunk_size = args.get("chunk_size", 5)
        latent_dim = args.get("latent_dim", 512)
        n_embed = args.get("num_codes", 16)
        groups = args.get("vq_groups", 2)

        model = VQBeTTokenizer(
            action_dim=7, chunk_size=chunk_size,
            latent_dim=latent_dim, n_embed=n_embed, groups=groups,
        )
        # Fit normalizer (same as training)
        data_dir = args.get("data_dir", "/data/user_data/yashagar/task_D_D/training/")
        normalizer = fit_normalizer(data_dir)
        model.set_normalizer(normalizer)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        # VQ-BeT: `groups` codes per chunk, each in [0, n_embed)
        # Total vocab = n_embed (codes from different groups share the same range
        # but have different semantic meaning, so we interleave them)
        return model, chunk_size, groups, n_embed, None  # no FSQ levels for VQ-BeT

    elif tokenizer_type in ("oat", "quest"):
        from tokenization.train_tokenizer import fit_normalizer

        if tokenizer_type == "oat":
            from tokenization.train_tokenizer import build_oat
            import argparse
            build_args = argparse.Namespace(**args)
            model = build_oat(build_args)
        else:
            from tokenization.train_tokenizer import build_quest
            import argparse
            build_args = argparse.Namespace(**args)
            model = build_quest(build_args)

        data_dir = args.get("data_dir", "/data/user_data/yashagar/task_D_D/training/")
        normalizer = fit_normalizer(data_dir)
        model.set_normalizer(normalizer)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        # Read FSQ levels from checkpoint args (may differ from default)
        fsq_levels = args.get("fsq_levels", [8, 5, 5, 5])
        codebook_size = 1
        for l in fsq_levels:
            codebook_size *= l

        if tokenizer_type == "oat":
            n_tokens = args.get("num_registers", 8)
        else:
            horizon = args.get("horizon", 32)
            downsample = args.get("downsample_factor", 4)
            n_tokens = horizon // downsample

        chunk_size = args.get("horizon", 32)
        return model, chunk_size, n_tokens, codebook_size, fsq_levels

    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def _fsq_codes_to_indices(codes, levels=(8, 5, 5, 5)):
    """Convert FSQ code vectors (B, T, D) to flat indices (B, T)."""
    indices = torch.zeros(codes.shape[:-1], dtype=torch.long, device=codes.device)
    # Shift codes from [-L//2, L//2] to [0, L-1]
    for d, L in enumerate(levels):
        shifted = codes[..., d] + L // 2
        shifted = shifted.clamp(0, L - 1).long()
        multiplier = 1
        for l in levels[d+1:]:
            multiplier *= l
        indices += shifted * multiplier
    return indices


def _indices_to_fsq_codes(indices, levels=(8, 5, 5, 5)):
    """Convert flat indices (B, T) back to FSQ code vectors (B, T, D)."""
    codes = []
    remainder = indices.clone()
    for d, L in enumerate(levels):
        multiplier = 1
        for l in levels[d+1:]:
            multiplier *= l
        val = remainder // multiplier
        remainder = remainder % multiplier
        codes.append(val - L // 2)  # shift back to [-L//2, L//2]
    return torch.stack(codes, dim=-1).float()


class CalvinSweepActionTokenizer(ActionTokenizer):
    """
    Action tokenizer adapter for VQ-BeT / OAT / QueST from the CALVIN-D sweep.

    Maps action chunks to sequences of discrete tokens in the LLM vocabulary.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_type: str,
        checkpoint_path: str,
        device: str = "cpu",
        use_extra: bool = False,
    ):
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_type = tokenizer_type

        self.model, self.chunk_size, self.n_codes_per_chunk, self.codebook_size, self.fsq_levels = \
            _load_tokenizer_model(tokenizer_type, checkpoint_path, device)

        self.n_bins = self.codebook_size

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("use_extra not supported for non-Qwen2 tokenizers")

        self.action_token_begin_idx = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx = int(self.tokenizer_len)

        print(f"[CalvinSweepActionTokenizer] type={tokenizer_type}, "
              f"chunk_size={self.chunk_size}, codes_per_chunk={self.n_codes_per_chunk}, "
              f"codebook_size={self.codebook_size}")

    # ------------------------------------------------------------------
    # Encoding: continuous action chunk → token string
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        action: (chunk_size, 7) numpy array
        Returns: decoded token string (n_codes_per_chunk special tokens)
        """
        x = torch.from_numpy(action).to(self.device).float()

        if self.tokenizer_type == "vq_bet":
            # (chunk_size, 7) → (1, chunk_size*7)
            x_flat = x.reshape(1, -1)
            _, indices, _ = self.model.encode(x_flat)  # (1, groups)
            token_codes = indices[0].cpu().tolist()  # list of `groups` ints

        elif self.tokenizer_type == "oat":
            x = x.unsqueeze(0)  # (1, T, 7)
            latents, _ = self.model.encode(x)  # post-FSQ (1, n_reg, D)
            flat_idx = _fsq_codes_to_indices(latents, levels=tuple(self.fsq_levels))
            token_codes = flat_idx[0].cpu().tolist()

        elif self.tokenizer_type == "quest":
            x = x.unsqueeze(0)  # (1, T, 7)
            codes = self.model.encode_fsq_codes(x)  # (1, T', D)
            flat_idx = _fsq_codes_to_indices(codes, levels=tuple(self.fsq_levels))
            token_codes = flat_idx[0].cpu().tolist()

        # Map codes to LLM vocab tokens (last n_bins tokens)
        token_ids = [self.tokenizer_len - 1 - c for c in token_codes]
        return self.tokenizer.decode(token_ids)

    # ------------------------------------------------------------------
    # Decoding: token IDs → continuous actions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_chunks(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Core decode: token IDs → (B, chunk_size, 7) continuous actions.

        action_token_ids: flat (N,) or grouped (B, n_codes_per_chunk)
        Returns: (B, chunk_size, 7) numpy array of decoded actions.
        """
        codes = self.tokenizer_len - 1 - action_token_ids
        codes = np.clip(codes, 0, self.n_bins - 1)

        # Reshape flat token arrays into (B, n_codes_per_chunk) groups
        if codes.ndim == 1:
            n = len(codes) - (len(codes) % self.n_codes_per_chunk)
            if n == 0:
                return np.zeros((1, self.chunk_size, 7), dtype=np.float32)
            codes = codes[:n].reshape(-1, self.n_codes_per_chunk)

        codes_t = torch.from_numpy(codes).to(self.device).long()

        if self.tokenizer_type == "vq_bet":
            recon = self.model.decode_from_indices(codes_t)  # (B, chunk_size*7)
            recon = self.model._unnormalize_flat(recon)
            actions = recon.view(-1, self.chunk_size, 7)

        elif self.tokenizer_type in ("oat", "quest"):
            actions = self.model.detokenize(codes_t)  # (B, T, 7), already unnormalized

        return actions.cpu().numpy()

    @torch.no_grad()
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Decode token IDs → first timestep only (for offline eval / training loss).
        Returns: (B, 7) numpy array.
        """
        actions = self._decode_chunks(action_token_ids)  # (B, chunk_size, 7)
        return actions[:, 0]

    @torch.no_grad()
    def decode_full_chunk(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Decode token IDs → all timesteps (for rollout execution).
        Returns: (chunk_size, 7) numpy array for a single chunk.
        """
        actions = self._decode_chunks(action_token_ids)  # (B, chunk_size, 7)
        return actions[0]  # (chunk_size, 7)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        """Number of future steps needed beyond current step to form a chunk."""
        return self.chunk_size - 1
