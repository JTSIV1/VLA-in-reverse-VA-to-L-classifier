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
        # Look up from (precomputed) embedding table.
        # For vanilla VLA: codes → LLM token IDs → embed_weight lookup.
        # For static fullproj: embed_weight already has proj(codebook) baked
        #   into the last n_action token slots.
        codes = result['codes']
        if tok_type == 'vq_bet' and codes.ndim == 3:
            B, K, G = codes.shape
            codes = codes.reshape(B, K * G)  # flatten groups, NO offset
            n_valid = n_valid * G
        codes = codes.long()
        tokenizer_len = vla_embed_info['tokenizer_len']
        embed_weight = vla_embed_info['embed_weight'].to(device)
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

    Handles three checkpoint formats:
      - Standard: llm.model.embed_tokens.weight (vanilla VLA)
      - Static fullproj (VQ-BeT): base_embedding + proj(codebook) → precomputed
        action embedding table replaces last n_action tokens
      - Dynamic fullproj (OAT/QueST): base_embedding + proj — requires per-input
        latents at inference time. Returns proj weight for external use.

    Args:
        policy_dir: path to policy run directory (e.g.
            checkpoints/bridge_sweep/policy/vq_bet_10_16_2_256/vanilla).
        device: torch device.

    Returns:
        dict with keys:
            'embed_weight': (vocab_size, embed_dim) tensor on device
            'tokenizer_len': int, LLM tokenizer vocab size
            'embed_dim': int, embedding dimension
            'is_fullproj': bool, whether this is a fullproj checkpoint
            'is_dynamic': bool, whether this is dynamic fullproj (OAT)
            'proj_weight': (d_out, d_in) tensor if fullproj, else None
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

    print(f"Loading VLA embedding from {fsdp_ckpt} ...")
    ckpt = torch.load(fsdp_ckpt, map_location="cpu")
    llm_sd = ckpt['model']['llm_backbone']

    is_fullproj = any("embed_tokens.base_embedding." in k for k in llm_sd)

    if not is_fullproj:
        # Standard vanilla: single embed_tokens.weight
        embed_weight = llm_sd['llm.model.embed_tokens.weight']
        del ckpt
        embed_dim = embed_weight.shape[-1]
        tokenizer_len = embed_weight.shape[0]
        print(f"  [vanilla] embed_dim={embed_dim}, tokenizer_len={tokenizer_len}")
        return {
            "embed_weight": embed_weight.to(device),
            "tokenizer_len": tokenizer_len,
            "embed_dim": embed_dim,
            "is_fullproj": False,
            "is_dynamic": False,
            "proj_weight": None,
        }

    # Fullproj: base_embedding + proj (+ optional codebook for static)
    base_weight = llm_sd['llm.model.embed_tokens.base_embedding.weight']
    proj_weight = llm_sd['llm.model.embed_tokens.proj.weight']
    codebook_key = 'llm.model.embed_tokens.codebook'
    has_codebook = codebook_key in llm_sd
    embed_dim = base_weight.shape[-1]
    tokenizer_len = base_weight.shape[0]

    if has_codebook:
        # Static fullproj (VQ-BeT): precompute action embeddings
        codebook = llm_sd[codebook_key]  # (n_action_tokens, d_vq)
        n_actions = codebook.shape[0]
        # proj: (d_out, d_vq) — maps codebook → embedding space
        action_embeds = codebook @ proj_weight.T  # (n_actions, d_out)

        d_fixed = proj_weight.shape[0]
        if d_fixed < embed_dim:
            # Partial: first d_fixed from proj, rest from free embedding
            free_key = 'llm.model.embed_tokens.action_embed_free.weight'
            free_weight = llm_sd[free_key]  # (n_actions, d_llm - d_fixed)
            action_embeds = torch.cat([action_embeds, free_weight], dim=-1)

        # Replace last n_actions tokens in base_weight
        embed_weight = base_weight.clone()
        embed_weight[-n_actions:] = action_embeds
        del ckpt
        print(f"  [static fullproj] embed_dim={embed_dim}, "
              f"n_actions={n_actions}, d_fixed={d_fixed}")
        return {
            "embed_weight": embed_weight.to(device),
            "tokenizer_len": tokenizer_len,
            "embed_dim": embed_dim,
            "is_fullproj": True,
            "is_dynamic": False,
            "proj_weight": proj_weight.to(device),
        }
    else:
        # Dynamic fullproj (OAT/QueST): proj maps per-input latents
        # Can't precompute — return proj for external use
        del ckpt
        d_latent = proj_weight.shape[1]
        print(f"  [dynamic fullproj] embed_dim={embed_dim}, "
              f"d_latent={d_latent}, proj={proj_weight.shape}")
        return {
            "embed_weight": base_weight.to(device),
            "tokenizer_len": tokenizer_len,
            "embed_dim": embed_dim,
            "is_fullproj": True,
            "is_dynamic": True,
            "proj_weight": proj_weight.to(device),
        }
