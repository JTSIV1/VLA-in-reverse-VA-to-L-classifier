"""Load frozen tokenizers and encode batches for downstream probes.

Used by verb_probe/train_verb_probe.py to encode CalvinTokenizerDataset
batches on-the-fly through a frozen tokenizer.
"""

import argparse
from pathlib import Path

import torch

from tokenization.train_utils import extract_episode_batch

def load_fast_tokenizer(args):
    """Load FAST tokenizer for on-the-fly tokenization.

    Returns: (tok_wrapper, vocab_size) or (None, None).
    """
    if args.action_rep != "fast":
        return None, None
    from tokenization.fast.fast_tokenizer import load_fast_tokenizer, tokenize_trajectory
    _fast_tok = load_fast_tokenizer(args.fast_tokenizer_path)
    def _fast_wrapper(actions_batch):
        return [tokenize_trajectory(_fast_tok, actions_batch[0])]
    _fast_wrapper.vocab_size = _fast_tok.vocab_size
    print(f"Loaded FAST tokenizer (vocab_size={_fast_tok.vocab_size})")
    return _fast_wrapper, _fast_tok.vocab_size

def load_frozen_tokenizer(tokenizer_type, tokenizer_ckpt):
    """Load a frozen tokenizer from checkpoint.

    Args:
        tokenizer_type: 'vq_bet', 'oat', or 'quest'.
        tokenizer_ckpt: path to full.pth checkpoint.

    Returns:
        Frozen model (eval mode, no gradients).
    """
    from tokenization.train_tokenizer import build_vqbet, build_oat, build_quest

    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    build_args = argparse.Namespace(**ckpt_args)

    builders = {"vq_bet": build_vqbet, "oat": build_oat, "quest": build_quest}
    if tokenizer_type not in builders:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
    model = builders[tokenizer_type](build_args)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if "normalizer" in ckpt:
        model.set_normalizer(ckpt["normalizer"])

    return model


def get_tokenizer_chunk_params(tokenizer_ckpt):
    """Read chunk_size, sampling, max_chunks from a tokenizer checkpoint.

    Returns dict with keys: chunk_size, sampling, max_chunks.
    """
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)
    return {
        "chunk_size": ckpt_args.get("chunk_size", 16),
        "sampling": ckpt_args.get("sampling", "random"),
        "max_chunks": ckpt_args.get("max_chunks", 8),
    }


def encode_tokenizer_batch(batch, device, tok_model, tok_type, mode,
                           vla_embed_info=None):
    """Encode a CalvinTokenizerDataset batch through a frozen tokenizer.

    Args:
        batch: dict from CalvinTokenizerDataset DataLoader.
        device: torch device.
        tok_model: frozen tokenizer model.
        tok_type: 'vq_bet', 'oat', or 'quest'.
        mode: 'latent', 'token_id', or 'vla_embed'.
        vla_embed_info: dict from load_vla_embedding (required for vla_embed).

    Returns:
        (actions, labels, n_valid) tensors on device.
    """
    with torch.no_grad():
        result = extract_episode_batch(tok_model, batch, device, tok_type)

    n_valid = result['n_valid']
    labels = result['verb_ids']

    if mode == 'latent':
        actions = result['latents']
    elif mode == 'vla_embed':
        # Get codes, then map to LLM token IDs, then look up VLA embeddings.
        # The CalvinSweepActionTokenizer maps each raw code c → tokenizer_len-1-c
        # WITHOUT group offsets (each VQ-BeT group code is mapped independently).
        codes = result['codes']
        if tok_type == 'vq_bet' and codes.ndim == 3:
            B, K, G = codes.shape
            codes = codes.reshape(B, K * G)  # flatten groups, NO offset
            n_valid = n_valid * G
        codes = codes.long()
        tokenizer_len = vla_embed_info['tokenizer_len']
        embed_weight = vla_embed_info['embed_weight']
        llm_ids = (tokenizer_len - 1 - codes).clamp(0, embed_weight.shape[0] - 1)
        actions = embed_weight[llm_ids]  # (B, T, embed_dim)
    else:
        codes = result['codes']
        if tok_type == 'vq_bet' and codes.ndim == 3:
            # (B, K, groups) → group-offset encoding → (B, K*groups)
            B, K, G = codes.shape
            offsets = torch.arange(G, device=device) * tok_model.n_embed
            codes = (codes + offsets.view(1, 1, G)).reshape(B, K * G)
            n_valid = n_valid * G
        actions = codes.long()

    return actions, labels, n_valid


def get_vocab_size(tok_model, tokenizer_type):
    """Get the effective vocabulary size for an embedding table.

    VQ-BeT uses group-offset encoding: vocab = n_embed * groups.
    OAT/QueST FSQ: vocab = codebook_size (product of FSQ levels).
    """
    if tokenizer_type == "vq_bet":
        return tok_model.n_embed * tok_model.groups
    elif hasattr(tok_model, 'vocab_size'):
        return tok_model.vocab_size
    else:
        return tok_model.codebook_size


def load_vla_embedding(policy_dir, device="cpu"):
    """Load the LLM embedding table from a trained VLA policy.

    Extracts the frozen input embedding weights and the token ID mapping
    needed to convert tokenizer codes → LLM vocab IDs.

    Args:
        policy_dir: path to policy run directory (e.g.
            checkpoints/calvin_sweep/policy/minivla_quest_16_4444_2).
        device: torch device.

    Returns:
        dict with keys:
            'embed_weight': (vocab_size, embed_dim) tensor on device
            'tokenizer_len': int, LLM tokenizer vocab size
            'embed_dim': int, embedding dimension
    """
    import re

    policy_dir = Path(policy_dir)

    # Find last checkpoint
    ckpt_dir = policy_dir / "checkpoints"
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint in {ckpt_dir}")
    def step_num(p):
        m = re.search(r"step-(\d+)", p.name)
        return int(m.group(1)) if m else -1
    fsdp_ckpt = str(max(candidates, key=step_num))

    # Load only the LLM embedding weight from the checkpoint (skip vision/projector)
    print(f"Loading VLA embedding from {fsdp_ckpt} ...")
    ckpt = torch.load(fsdp_ckpt, map_location="cpu")
    embed_weight = ckpt['model']['llm_backbone']['llm.model.embed_tokens.weight']
    del ckpt  # free the rest immediately
    embed_dim = embed_weight.shape[-1]
    tokenizer_len = embed_weight.shape[0]

    print(f"  embed_dim={embed_dim}, tokenizer_len={tokenizer_len}")

    return {
        "embed_weight": embed_weight.to(device),
        "tokenizer_len": tokenizer_len,
        "embed_dim": embed_dim,
    }
