"""Fit and use the FAST (Frequency-space Action Sequence Tokenization) tokenizer
on CALVIN action trajectories.

Implements the DCT + BPE pipeline from the FAST paper (arxiv 2501.09747),
vendored locally to support Python 3.9 (the HuggingFace version uses 3.10+ syntax).

Usage (standalone):
    python -m tokenization.fast_tokenizer --save_path ./checkpoints/fast_tokenizer

Usage (as module):
    from tokenization.fast_tokenizer import load_fast_tokenizer, tokenize_trajectory
    tok = load_fast_tokenizer("./checkpoints/fast_tokenizer")
    token_ids = tokenize_trajectory(tok, actions_np)  # (T, 7) -> list[int]
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, List

# Ensure project root is on path for standalone execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.fft import dct, idct
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from utils import load_calvin_to_dataframe
from config import DATA_DIR, ACTION_KEY, EPISODE_TEMPLATE, FAST_TOKENIZER_PATH


class FASTTokenizer:
    """FAST action tokenizer: DCT -> quantize -> BPE.
    Vendored from physical-intelligence/fast with Python 3.9 compat."""

    def __init__(self, bpe_tokenizer: PreTrainedTokenizerFast,
                 scale: float = 10, vocab_size: int = 1024,
                 min_token: int = 0,
                 action_dim: Optional[int] = None,
                 time_horizon: Optional[int] = None):
        self.bpe_tokenizer = bpe_tokenizer
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        self.action_dim = action_dim
        self.time_horizon = time_horizon

    def __call__(self, action_chunk: np.ndarray) -> List[List[int]]:
        """Tokenize action chunks.
        Args:
            action_chunk: (batch, T, action_dim) or (T, action_dim)
        Returns:
            list of list[int] token IDs, one per batch element
        """
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]
        assert action_chunk.ndim == 3

        self.time_horizon = action_chunk.shape[1]
        self.action_dim = action_chunk.shape[2]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens = []
        for elem in dct_coeff:
            token_str = "".join(
                map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(self, tokens: List[List[int]],
               time_horizon: Optional[int] = None,
               action_dim: Optional[int] = None) -> np.ndarray:
        th = time_horizon or self.time_horizon
        ad = action_dim or self.action_dim
        assert th is not None and ad is not None, \
            "Call encode() first or pass time_horizon and action_dim."

        decoded_actions = []
        for token in tokens:
            try:
                decoded_str = self.bpe_tokenizer.decode(token)
                decoded_dct = np.array(list(map(ord, decoded_str))) + self.min_token
                decoded_dct = decoded_dct.reshape(th, ad)
            except Exception as e:
                logging.warning(f"Error decoding tokens: {e}")
                decoded_dct = np.zeros((th, ad))
            decoded_actions.append(idct(decoded_dct / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    @classmethod
    def fit(cls, action_data: List[np.ndarray], scale: float = 10,
            vocab_size: int = 1024) -> "FASTTokenizer":
        """Train a FAST tokenizer from a list of (T_i, action_dim) trajectories."""
        # DCT transform all trajectories
        dct_tokens = [dct(a, axis=0, norm="ortho").flatten() for a in action_data]

        # Find quantization range
        all_vals = np.around(np.concatenate(dct_tokens) * scale)
        max_token = int(all_vals.max())
        min_token = int(all_vals.min())
        min_vocab_size = max_token - min_token

        assert min_vocab_size <= vocab_size, \
            f"Vocab size {vocab_size} too small for token range {min_vocab_size}"
        if min_vocab_size + 100 > vocab_size:
            logging.warning(
                f"Initial alphabet size {min_vocab_size} nearly fills "
                f"vocab_size {vocab_size}, consider increasing")

        # Build string iterator for BPE training
        def _token_iter():
            for tokens in dct_tokens:
                rounded = (np.around(tokens * scale) - min_token).astype(int)
                yield "".join(map(chr, rounded))

        # Train BPE
        bpe = ByteLevelBPETokenizer()
        alphabet = [chr(i) for i in range(max_token - min_token + 1)]
        trainer = BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, show_progress=True,
            special_tokens=[], initial_alphabet=alphabet, max_token_length=10000)
        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=bpe, clean_up_tokenization_spaces=False)
        return cls(wrapped, scale=scale, vocab_size=vocab_size, min_token=min_token)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.bpe_tokenizer.save_pretrained(path)
        meta = {"scale": self.scale, "vocab_size": self.vocab_size,
                "min_token": self.min_token}
        with open(os.path.join(path, "fast_config.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "FASTTokenizer":
        bpe = PreTrainedTokenizerFast.from_pretrained(path)
        with open(os.path.join(path, "fast_config.json")) as f:
            meta = json.load(f)
        return cls(bpe, **meta)


def collect_trajectories(df, data_dir):
    """Collect all action trajectories from the dataset.
    Returns list of (T_i, 7) arrays."""
    trajectories = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting trajectories"):
        actions = []
        for i in range(row['start_idx'], row['end_idx'] + 1):
            path = os.path.join(data_dir, EPISODE_TEMPLATE.format(i))
            data = np.load(path, mmap_mode='r')
            actions.append(np.array(data[ACTION_KEY]))
        trajectories.append(np.array(actions))
    return trajectories


def fit_fast_tokenizer(df, data_dir, save_path, vocab_size=1024, scale=10):
    """Collect CALVIN training trajectories, fit FAST tokenizer, save."""
    trajectories = collect_trajectories(df, data_dir)

    print(f"Fitting FAST tokenizer on {len(trajectories)} trajectories "
          f"(max_T={max(a.shape[0] for a in trajectories)}, "
          f"action_dim={trajectories[0].shape[1]}, vocab_size={vocab_size}, scale={scale})...")
    tokenizer = FASTTokenizer.fit(trajectories, vocab_size=vocab_size, scale=scale)

    tokenizer.save_pretrained(save_path)
    print(f"FAST tokenizer saved to {save_path}")
    return tokenizer


def load_fast_tokenizer(path=FAST_TOKENIZER_PATH):
    """Load a previously fitted FAST tokenizer."""
    return FASTTokenizer.from_pretrained(path)


def download_pretrained_fast(save_path="./checkpoints/fast_pretrained"):
    """Download the pretrained FAST+ tokenizer from HuggingFace and save in our format."""
    import shutil
    from huggingface_hub import hf_hub_download

    os.makedirs(save_path, exist_ok=True)
    repo_id = "physical-intelligence/fast"

    for fname in ["tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "processor_config.json"]:
        src = hf_hub_download(repo_id, fname)
        shutil.copy2(src, os.path.join(save_path, fname))

    # Read processor_config to extract FAST-specific params
    with open(os.path.join(save_path, "processor_config.json")) as f:
        proc_config = json.load(f)

    # Save in our format (fast_config.json)
    meta = {
        "scale": proc_config.get("scale", 10),
        "vocab_size": proc_config.get("vocab_size", 2048),
        "min_token": proc_config.get("min_token", 0),
    }
    with open(os.path.join(save_path, "fast_config.json"), "w") as f:
        json.dump(meta, f)

    print(f"Pretrained FAST+ saved to {save_path} "
          f"(scale={meta['scale']}, vocab={meta['vocab_size']}, "
          f"min_token={meta['min_token']})")
    return save_path


def tokenize_trajectory(tokenizer, actions_np):
    """Tokenize a single trajectory.
    Args:
        actions_np: (T, action_dim) numpy array
    Returns:
        list[int] of FAST token IDs
    """
    return tokenizer(actions_np[np.newaxis])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to CALVIN training split")
    parser.add_argument("--save_path", type=str, default=FAST_TOKENIZER_PATH)
    parser.add_argument("--vocab_size", type=int, default=1024,
                        help="BPE vocabulary size for FAST tokenizer")
    parser.add_argument("--scale", type=float, default=10,
                        help="DCT quantization scale (higher = finer quantization, larger alphabet)")
    parser.add_argument("--debug", type=int, default=0, metavar="N",
                        help="Use only N samples for quick testing")
    args = parser.parse_args()

    df = load_calvin_to_dataframe(args.data_dir)
    if args.debug:
        df = df.head(min(args.debug, len(df))).copy()
        print(f"[DEBUG] Using {len(df)} samples")

    fit_fast_tokenizer(df, args.data_dir, args.save_path,
                       vocab_size=args.vocab_size, scale=args.scale)
