import os
import numpy as np
import torch

from config import ACTION_KEY, EPISODE_TEMPLATE
from action_tokenizers_training import train_tokenizer, fit_calvin_normalizer
from cluster_analysis import build_features

from config import (
    MAX_SEQ_LEN,
    TOKENIZER_HORIZON,
    QUEST_TOKENIZER_CKPT,
    OAT_TOKENIZER_CKPT,
    TOKENIZER_FIT_NORM_MAX_TRAJS,
    TOKENIZER_DOWNSAMPLE_FACTOR,
    OAT_NUM_REGISTERS,
    ACTION_DIM,
    BINNING_VOCAB_SIZE
)

# oat tokenizers
from oat.tokenizer.bin.tokenizer import BinTok
from oat.tokenizer.fast.tokenizer_wrapper import FASTTok
from oat.tokenizer.quest.tokenizer import QueSTTok
from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ



class TokenizerAdapter:
    """
    Adapts oat tokenizers to your training code’s expectation:
    callable(actions_np) -> List[List[int]] or List[int]
    """
    def __init__(self, tok, mode: str, horizon: int, max_tokens: int):
        self.tok = tok
        self.mode = mode
        self.horizon = horizon
        self.max_tokens = max_tokens

        # vocab size (used to size nn.Embedding)
        if hasattr(tok, "vocab_size"):
            self.vocab_size = int(tok.vocab_size)
        elif hasattr(tok, "codebook_size"):
            self.vocab_size = int(tok.codebook_size)
        else:
            # BinTok uses num_bins
            self.vocab_size = int(getattr(tok, "num_bins", 256))

    def __call__(self, actions_np: np.ndarray):
        """
        actions_np: (T,D) or (B,T,D)
        returns: List[List[int]]
        """
        if actions_np.ndim == 2:
            actions_np = actions_np[None, ...]

        B, T, D = actions_np.shape

        # make fixed horizon by pad/truncate in time (important for QueST/OAT)
        if T < self.horizon:
            pad = np.zeros((B, self.horizon - T, D), dtype=np.float32)
            x = np.concatenate([actions_np, pad], axis=1)
        else:
            x = actions_np[:, : self.horizon, :]

        x_t = torch.from_numpy(x).float()

        if self.mode == "fast":
            # FASTTok.tokenize returns List[List[int]] variable-length
            tokens = self.tok.tokenize(x_t)
            return tokens

        if self.mode == "bin":
            # BinTok.tokenize returns (B,T,D) ints -> flatten to (B,T*D)
            ids = self.tok.tokenize(x_t).reshape(B, -1)
            return [row.tolist() for row in ids]

        if self.mode in ("quest", "oat"):
            # QueSTTok.tokenize returns (B, T') ints
            # OATTok.tokenize returns (B, latent_horizon) ints
            ids = self.tok.tokenize(x_t)
            if ids.ndim > 2:
                ids = ids.reshape(B, -1)
            return [row.tolist() for row in ids]

        raise ValueError(f"Unknown tokenizer mode {self.mode}")


def load_action_tokenizer(
    name: str,
    train_dir: str,
    *,
    horizon: int = TOKENIZER_HORIZON,
    max_tokens: int = MAX_SEQ_LEN,  # max tokens your LM can take (after which you truncate/pad)
    quest_ckpt: str = QUEST_TOKENIZER_CKPT,
    oat_ckpt: str = OAT_TOKENIZER_CKPT,
    fit_norm_max_trajs: int = TOKENIZER_FIT_NORM_MAX_TRAJS,
):
    """
    name: "fast" | "bin" | "quest" | "oat"
    """
    name = name.lower()

    # normalizer for everything except raw HF FAST (FASTTok expects normalized [-1,1], and wraps a normalizer too)
    normalizer = fit_calvin_normalizer(train_dir, max_trajs=fit_norm_max_trajs)
  
    if name == "fast":
        tok = FASTTok("physical-intelligence/fast")  # pretrained from HF
        tok.set_normalizer(normalizer)
        return TokenizerAdapter(tok, "fast", horizon=horizon, max_tokens=max_tokens)

    if name == "bin":
        tok = BinTok(num_bins=BINNING_VOCAB_SIZE, min_val=-1.0, max_val=1.0)
        tok.set_normalizer(normalizer)
        return TokenizerAdapter(tok, "bin", horizon=horizon, max_tokens=max_tokens)

    if name == "quest":
        # vocab size is product of fsq levels.
        # need weights; if not present you’ll train (next section)
        tok = QueSTTok(action_dim=ACTION_DIM, horizon=horizon, vq_type="fsq", fsq_level=[8, 5, 5, 5], downsample_factor=TOKENIZER_DOWNSAMPLE_FACTOR)
        tok.set_normalizer(normalizer)
        if not os.path.exists(quest_ckpt):
            print(f"No checkpoint found for QueST tokenizer at {quest_ckpt}, starting training")
            train_tokenizer(name, train_dir, quest_ckpt, normalizer=normalizer)
            assert os.path.exists(quest_ckpt), "Tokenizer training did not produce checkpoint!"

        sd = torch.load(quest_ckpt, map_location="cpu", weights_only=False)
        tok.load_state_dict(sd["model"])
        tok.set_normalizer(sd["normalizer"])
            
        return TokenizerAdapter(tok, "quest", horizon=horizon, max_tokens=max_tokens)

    if name == "oat":
        # vocab size is product of latent_levels
        # need weights; if not present you’ll train
        latent_levels = [8, 5, 5, 5]
        latent_dim = len(latent_levels)
        num_registers = OAT_NUM_REGISTERS

        encoder = RegisterEncoder(
            sample_dim=7, sample_horizon=horizon,
            emb_dim=256, head_dim=64, depth=2, pdropout=0.1,
            latent_dim=latent_dim, num_registers=num_registers
        )
        decoder = SinglePassDecoder(
            sample_dim=7, sample_horizon=horizon,
            emb_dim=256, head_dim=64, depth=4, pdropout=0.1,
            token_dropout_mode="pow2", latent_dim=latent_dim,
            latent_horizon=num_registers, use_causal_decoder=True
        )
        quantizer = FSQ(levels=latent_levels)
        tok = OATTok(encoder=encoder, decoder=decoder, quantizer=quantizer)
        tok.set_normalizer(normalizer)

        if not os.path.exists(oat_ckpt):
            print(f"No checkpoint found for OAT tokenizer at {oat_ckpt}, starting training")
            train_tokenizer(name, train_dir, oat_ckpt, normalizer=normalizer)
            assert os.path.exists(oat_ckpt), "Tokenizer training did not produce checkpoint!"

        sd = torch.load(oat_ckpt, map_location="cpu", weights_only=False)
        tok.load_state_dict(sd["model"])
        tok.set_normalizer(sd["normalizer"])
        return TokenizerAdapter(tok, "oat", horizon=horizon, max_tokens=max_tokens)

    raise ValueError(f"Unknown tokenizer name {name}")