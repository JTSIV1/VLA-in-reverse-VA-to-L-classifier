"""Action tokenizer loading and trajectory tokenization.

Supports multiple action representations for the verb classifier:
  - native:  raw 7-DoF actions, projected via nn.Linear (no tokenizer needed)
  - fast:    DCT + BPE discrete tokens (FAST paper, arxiv 2501.09747)
  - bin:     per-dimension uniform binning
  - vq_vae:  chunk-based VQ-VAE (simple MLP encoder, single codebook)
  - vqvla:   pretrained VQ-VLA (causal conv VAE + 4-group ResidualVQ)
  - vq_bet:  alias for vqvla (VQ-BeT uses same architecture)
  - quest:   QueST tokenizer (conv-based)
  - oat:     Object Action Tokenizer (register-based + FSQ)

All tokenizers are loaded via load_action_tokenizer() which returns a
TokenizerAdapter with a uniform call interface:
    adapter(actions_np)  # (T, D) or (B, T, D) -> List[List[int]]

Heavy dependencies (dill, zarr, vector_quantize_pytorch) are imported lazily
so the mmml conda env can load this module without installing everything.
"""
import os
import sys
import numpy as np
import torch

# Ensure project root is on path for standalone execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Vendored oat/ package uses absolute imports (from oat.X); needs its parent on path
_TOKENIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZATION_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZATION_DIR)

from config import ACTION_KEY, EPISODE_TEMPLATE
from analysis.cluster_analysis import build_features


def fit_calvin_normalizer(data_dir, max_trajs=None):
    """Fit oat LinearNormalizer on all CALVIN actions in data_dir."""
    from oat.model.common.normalizer import LinearNormalizer
    from analysis.cluster_analysis import load_all_actions
    from utils import load_calvin_to_dataframe

    print("Fitting tokenizer normalizer on actions from", data_dir)
    df = load_calvin_to_dataframe(data_dir)
    if max_trajs:
        df = df.head(min(max_trajs, len(df))).copy()
    all_actions, _ = load_all_actions(df, num_workers=8)
    actions_t = torch.from_numpy(all_actions)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": actions_t}, last_n_dims=1, mode="limits",
                   output_min=-1.0, output_max=1.0)
    return normalizer

from config import (
    MAX_SEQ_LEN,
    TOKENIZER_HORIZON,
    QUEST_TOKENIZER_CKPT,
    OAT_TOKENIZER_CKPT,
    TOKENIZER_FIT_NORM_MAX_TRAJS,
    TOKENIZER_DOWNSAMPLE_FACTOR,
    OAT_NUM_REGISTERS,
    ACTION_DIM,
    BINNING_VOCAB_SIZE,
    VQVAE_TOKENIZER_PATH,
)

# All oat tokenizer imports are lazy to avoid hard dependencies at module load time
def _import_bin_tok():
    from oat.tokenizer.bin.tokenizer import BinTok
    return BinTok

def _import_fast_tok():
    from oat.tokenizer.fast.tokenizer_wrapper import FASTTok
    return FASTTok

def _import_quest_tok():
    from oat.tokenizer.quest.tokenizer import QueSTTok
    return QueSTTok

def _import_oat_tok():
    from oat.tokenizer.oat.tokenizer import OATTok
    from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
    from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
    from oat.tokenizer.oat.quantizer.fsq import FSQ
    return OATTok, RegisterEncoder, SinglePassDecoder, FSQ

def _import_vqvae_tok():
    from tokenization.vqvae_tokenizer import (
        load_vqvae_tokenizer, tokenize_trajectory_vqvae,
        load_vqvla_tokenizer, tokenize_trajectory_vqvla,
        VQVLA_VOCAB_SIZE, VQVLA_WINDOW_SIZE,
    )
    return (load_vqvae_tokenizer, tokenize_trajectory_vqvae,
            load_vqvla_tokenizer, tokenize_trajectory_vqvla,
            VQVLA_VOCAB_SIZE, VQVLA_WINDOW_SIZE)



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
        if mode in ("vqvla", "vq_bet"):
            (_, _, _, _, VQVLA_VOCAB_SIZE, _) = _import_vqvae_tok()
            self.vocab_size = VQVLA_VOCAB_SIZE
        elif mode == "vq_vae":
            self.vocab_size = int(getattr(tok, "num_codes", 512))
        elif hasattr(tok, "vocab_size"):
            self.vocab_size = int(tok.vocab_size)
        elif hasattr(tok, "codebook_size"):
            self.vocab_size = int(tok.codebook_size)
        else:
            # BinTok uses num_bins
            self.vocab_size = int(getattr(tok, "num_bins", 256))

        # Stash tokenize helpers for vq_vae/vqvla (loaded lazily)
        self._tokenize_vqvae_fn = None
        self._tokenize_vqvla_fn = None

    def _tokenize_vqvae(self, actions_1d):
        """Tokenize a single (T, D) trajectory with VQ-VAE."""
        if self._tokenize_vqvae_fn is None:
            (_, tokenize_trajectory_vqvae, _, _, _, _) = _import_vqvae_tok()
            self._tokenize_vqvae_fn = tokenize_trajectory_vqvae
        return self._tokenize_vqvae_fn(self.tok, actions_1d)

    def _tokenize_vqvla(self, actions_1d):
        """Tokenize a single (T, D) trajectory with VQ-VLA."""
        if self._tokenize_vqvla_fn is None:
            (_, _, _, tokenize_trajectory_vqvla, _, _) = _import_vqvae_tok()
            self._tokenize_vqvla_fn = tokenize_trajectory_vqvla
        return self._tokenize_vqvla_fn(self.tok, actions_1d)

    def __call__(self, actions_np: np.ndarray):
        """
        actions_np: (T,D) or (B,T,D)
        returns: List[List[int]]
        """
        if actions_np.ndim == 2:
            actions_np = actions_np[None, ...]

        B, T, D = actions_np.shape

        if self.mode == "vq_vae":
            # VQ-VAE processes each trajectory independently, variable-length OK
            results = []
            for b in range(B):
                ids = self._tokenize_vqvae(actions_np[b])
                results.append(ids.tolist())
            return results

        if self.mode in ("vqvla", "vq_bet"):
            # VQ-VLA processes 5-step windows, variable-length OK
            results = []
            for b in range(B):
                ids = self._tokenize_vqvla(actions_np[b])
                results.append(ids.tolist())
            return results

        # Fixed-horizon tokenizers (fast, bin, quest, oat)
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
    train_dir: str = None,
    *,
    horizon: int = TOKENIZER_HORIZON,
    max_tokens: int = MAX_SEQ_LEN,
    quest_ckpt: str = QUEST_TOKENIZER_CKPT,
    oat_ckpt: str = OAT_TOKENIZER_CKPT,
    vqvae_ckpt: str = VQVAE_TOKENIZER_PATH,
    vqvla_ckpt: str = None,
    vqvla_config_dir: str = None,
    fit_norm_max_trajs: int = TOKENIZER_FIT_NORM_MAX_TRAJS,
):
    """Load any supported action tokenizer, returning a TokenizerAdapter.

    name: "fast" | "bin" | "quest" | "oat" | "vq_vae" | "vqvla" | "vq_bet"
    train_dir: CALVIN training dir (needed for normalizer fitting; not needed for vq_vae/vqvla)
    """
    name = name.lower()

    # vq_vae and vqvla don't need a normalizer or train_dir
    if name in ("vq_vae", "vqvla", "vq_bet"):
        pass  # handled below without normalizer
    else:
        assert train_dir is not None, f"train_dir required for tokenizer '{name}'"
        normalizer = fit_calvin_normalizer(train_dir, max_trajs=fit_norm_max_trajs)
  
    if name == "fast":
        FASTTok = _import_fast_tok()
        tok = FASTTok("physical-intelligence/fast")  # pretrained from HF
        tok.set_normalizer(normalizer)
        return TokenizerAdapter(tok, "fast", horizon=horizon, max_tokens=max_tokens)

    if name == "bin":
        BinTok = _import_bin_tok()
        tok = BinTok(num_bins=BINNING_VOCAB_SIZE, min_val=-1.0, max_val=1.0)
        tok.set_normalizer(normalizer)
        return TokenizerAdapter(tok, "bin", horizon=horizon, max_tokens=max_tokens)

    if name == "quest":
        QueSTTok = _import_quest_tok()
        tok = QueSTTok(action_dim=ACTION_DIM, horizon=horizon, vq_type="fsq", fsq_level=[8, 5, 5, 5], downsample_factor=TOKENIZER_DOWNSAMPLE_FACTOR)
        tok.set_normalizer(normalizer)
        assert os.path.exists(quest_ckpt), \
            f"No QueST checkpoint at {quest_ckpt}. Train first via: python tokenization/train_tokenizer.py --tokenizer quest"
        sd = torch.load(quest_ckpt, map_location="cpu", weights_only=False)
        tok.load_state_dict(sd["model"])
        tok.set_normalizer(sd["normalizer"])
        return TokenizerAdapter(tok, "quest", horizon=horizon, max_tokens=max_tokens)

    if name == "oat":
        OATTok, RegisterEncoder, SinglePassDecoder, FSQ = _import_oat_tok()
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
        assert os.path.exists(oat_ckpt), \
            f"No OAT checkpoint at {oat_ckpt}. Train first via: python tokenization/train_tokenizer.py --tokenizer oat"
        sd = torch.load(oat_ckpt, map_location="cpu", weights_only=False)
        tok.load_state_dict(sd["model"])
        tok.set_normalizer(sd["normalizer"])
        return TokenizerAdapter(tok, "oat", horizon=horizon, max_tokens=max_tokens)

    if name == "vq_vae":
        (load_vqvae_tokenizer, _, _, _, _, _) = _import_vqvae_tok()
        tok = load_vqvae_tokenizer(vqvae_ckpt)
        return TokenizerAdapter(tok, "vq_vae", horizon=horizon, max_tokens=max_tokens)

    if name in ("vqvla", "vq_bet"):
        (_, _, load_vqvla_tokenizer, _, _, _) = _import_vqvae_tok()
        # Use defaults from vqvae_tokenizer module if not specified
        if vqvla_config_dir is None or vqvla_ckpt is None:
            from tokenization.vqvae_tokenizer import VQVLA_CONFIG_DIR, VQVLA_CHECKPOINT_PATH
            vqvla_config_dir = vqvla_config_dir or VQVLA_CONFIG_DIR
            vqvla_ckpt = vqvla_ckpt or VQVLA_CHECKPOINT_PATH
        tok = load_vqvla_tokenizer(
            config_dir=vqvla_config_dir, checkpoint_path=vqvla_ckpt)
        return TokenizerAdapter(tok, "vqvla", horizon=horizon, max_tokens=max_tokens)

    raise ValueError(f"Unknown tokenizer name {name}")