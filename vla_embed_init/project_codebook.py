#!/usr/bin/env python3
"""Project tokenizer codebook vectors into VLA embedding space.

Replaces the VLA's action token embeddings (indices 151665–151920) with
codebook vectors projected via PCA-aligned distribution matching, so that:
  - The projected vectors have the same mean/covariance as text embeddings
  - The codebook's internal geometry (pairwise distances) is preserved

Usage:
    python vla_embed_init/project_codebook.py \
        --tokenizer_type quest \
        --tokenizer_ckpt checkpoints/calvin_sweep/tokenizers/quest_16_4444_4/full.pth \
        --base_vlm_ckpt <base_vlm>/checkpoints/latest-checkpoint.pt \
        --output_ckpt output/base_init.pt
"""

import argparse
import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


# ── Codebook extraction ──────────────────────────────────────────────

def extract_quest_codebook(tokenizer_ckpt: str) -> torch.Tensor:
    """Enumerate all FSQ grid points for a QueST tokenizer.

    Returns: (K, d_c) tensor of codebook vectors, where K = prod(fsq_levels)
    and d_c = len(fsq_levels).
    """
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    args = ckpt["args"]
    if not isinstance(args, dict):
        args = vars(args)
    fsq_levels = args["fsq_levels"]
    if isinstance(fsq_levels, str):
        fsq_levels = eval(fsq_levels)

    # FSQ quantization grid: for level L, values are round(x*(L-1)/2) / ((L-1)/2)
    # which produces L evenly spaced values in [-1, 1]
    def fsq_grid(L):
        return torch.linspace(-1, 1, L)

    grids = [fsq_grid(L) for L in fsq_levels]
    # Cartesian product of all dimensions
    points = torch.tensor(list(itertools.product(*grids)), dtype=torch.float32)

    print(f"  QueST FSQ levels={fsq_levels}, codebook size={len(points)}, d_c={len(fsq_levels)}")
    return points


def extract_vqbet_codebook(tokenizer_ckpt: str) -> torch.Tensor:
    """Extract VQ-BeT learned codebook embeddings.

    VQ-BeT uses ResidualVQ with multiple layers (groups). Each group has its
    own codebook of shape (1, num_codes, dim). The CalvinSweepActionTokenizer
    maps each per-group code c to the SAME LLM token ID (tokenizer_len - 1 - c),
    so all groups share the same num_codes token IDs.

    We average each group's codebook vectors for the same code index to get a
    single representative vector per code.

    Returns: (num_codes, dim) tensor.
    """
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    state = ckpt["model_state_dict"]

    # Collect all group codebooks: vq_layer.layers.{i}._codebook.embed
    codebooks = []
    i = 0
    while True:
        key = f"vq_layer.layers.{i}._codebook.embed"
        if key not in state:
            break
        codebooks.append(state[key].squeeze(0))  # (num_codes, dim)
        i += 1

    if not codebooks:
        raise ValueError(f"Could not find VQ-BeT codebooks. Keys: {list(state.keys())[:20]}")

    num_groups = len(codebooks)
    num_codes, dim = codebooks[0].shape
    stacked = torch.stack(codebooks)  # (groups, num_codes, dim)
    # Average across groups for each code index
    averaged = stacked.mean(dim=0)  # (num_codes, dim)

    print(f"  VQ-BeT: {num_groups} groups, {num_codes} codes, {dim}-d per group → averaged ({num_codes}, {dim})")
    return averaged.float()


EXTRACTORS = {
    "quest": extract_quest_codebook,
    "vq_bet": extract_vqbet_codebook,
}


# ── PCA-aligned projection ──────────────────────────────────────────

def pca_aligned_projection(
    codebook: torch.Tensor,
    text_embeddings: torch.Tensor,
    add_residual_noise: bool = False,
) -> torch.Tensor:
    """Project codebook vectors into text embedding space via PCA alignment.

    Args:
        codebook: (K, d_c) codebook vectors
        text_embeddings: (N, d_e) existing text token embeddings
        add_residual_noise: if True, add noise in the d_e - d_c residual dims

    Returns:
        (K, d_e) projected codebook vectors
    """
    K, d_c = codebook.shape
    N, d_e = text_embeddings.shape
    assert d_c <= d_e, f"Codebook dim ({d_c}) must be <= embedding dim ({d_e})"

    # Step 1: Codebook PCA
    mu_c = codebook.mean(dim=0)
    C_centered = codebook - mu_c
    U_c, sigma_c, Vt_c = torch.linalg.svd(C_centered, full_matrices=False)
    V_c = Vt_c.T  # (d_c, d_c)

    # Clamp small singular values to avoid division by zero
    sigma_c = sigma_c.clamp(min=1e-8)

    print(f"  Codebook PCA: d_c={d_c}, singular values={sigma_c[:5].tolist()}")

    # Step 2: Text embedding PCA
    mu_t = text_embeddings.mean(dim=0)
    T_centered = text_embeddings - mu_t
    # Only need top d_c components
    U_t, sigma_t, Vt_t = torch.linalg.svd(T_centered, full_matrices=False)
    V_t = Vt_t.T  # (d_e, d_e) but we only use first d_c columns

    print(f"  Text PCA: d_e={d_e}, top-{d_c} singular values={sigma_t[:d_c].tolist()}")

    # Step 3: Aligned projection
    # Project codebook to its PCA coordinates
    Z = C_centered @ V_c  # (K, d_c)

    # Rescale to match text variance in each PC direction
    scale = sigma_t[:d_c] / sigma_c  # (d_c,)
    Z_scaled = Z * scale.unsqueeze(0)  # (K, d_c)

    # Rotate to text PCA basis (only first d_c directions)
    C_proj = Z_scaled @ V_t[:, :d_c].T  # (K, d_e)

    # Step 4: Add residual noise if requested
    if add_residual_noise and d_c < d_e:
        residual_dims = d_e - d_c
        # Scale noise by text std in residual directions
        residual_std = sigma_t[d_c:d_c + residual_dims] / np.sqrt(N)
        noise = torch.randn(K, residual_dims) * residual_std.unsqueeze(0)
        # Project noise into residual text PCA directions
        residual_noise = noise @ V_t[:, d_c:d_c + residual_dims].T
        C_proj = C_proj + residual_noise

    # Step 5: Norm matching
    # The PCA alignment preserves covariance per-dimension, but the overall
    # norm distribution may differ (especially when d_c << d_e and the codebook
    # is a regular grid). Rescale so projected norms match text norm distribution.
    text_norms = text_embeddings.norm(dim=-1)
    proj_norms = C_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Map projected norm percentiles to text norm percentiles
    target_mean_norm = text_norms.mean()
    target_std_norm = text_norms.std()
    proj_mean_norm = proj_norms.mean()
    proj_std_norm = proj_norms.std().clamp(min=1e-8)
    # Affine transform: match mean and std of norms
    normalized_norms = (proj_norms - proj_mean_norm) / proj_std_norm
    target_norms = normalized_norms * target_std_norm + target_mean_norm
    C_proj = C_proj * (target_norms / proj_norms)

    # Add text mean
    C_proj = C_proj + mu_t

    return C_proj


# ── Verification ─────────────────────────────────────────────────────

def verify_projection(codebook, projected, text_embeddings):
    """Print verification metrics for the projection."""
    K = codebook.shape[0]

    # Norm comparison
    proj_norms = projected.norm(dim=-1)
    text_norms = text_embeddings.norm(dim=-1)
    print(f"\n  Verification:")
    print(f"    Projected norms: mean={proj_norms.mean():.4f}, "
          f"std={proj_norms.std():.4f}, range=[{proj_norms.min():.4f}, {proj_norms.max():.4f}]")
    print(f"    Text norms:      mean={text_norms.mean():.4f}, "
          f"std={text_norms.std():.4f}, range=[{text_norms.min():.4f}, {text_norms.max():.4f}]")

    # Pairwise cosine similarity
    proj_normed = projected / proj_norms.unsqueeze(-1)
    proj_sim = proj_normed @ proj_normed.T
    proj_sim.fill_diagonal_(0)
    mean_sim = proj_sim.sum() / (K * (K - 1))
    print(f"    Projected pairwise cosine: mean={mean_sim:.4f}")

    text_sample = text_embeddings[:1000]
    t_normed = text_sample / text_sample.norm(dim=-1, keepdim=True)
    t_sim = t_normed @ t_normed.T
    t_sim.fill_diagonal_(0)
    n = text_sample.shape[0]
    t_mean = t_sim.sum() / (n * (n - 1))
    print(f"    Text pairwise cosine (first 1000): mean={t_mean:.4f}")

    # Distance preservation (Spearman correlation)
    cb_dists = torch.cdist(codebook.float(), codebook.float()).view(-1)
    proj_dists = torch.cdist(projected.float(), projected.float()).view(-1)

    # Spearman rank correlation
    def spearman(x, y):
        rx = x.argsort().argsort().float()
        ry = y.argsort().argsort().float()
        n = len(x)
        return 1 - 6 * ((rx - ry) ** 2).sum() / (n * (n ** 2 - 1))

    rho = spearman(cb_dists, proj_dists)
    print(f"    Distance preservation (Spearman ρ): {rho:.4f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Project tokenizer codebook into VLA embedding space")
    parser.add_argument("--tokenizer_type", type=str, required=True, choices=["quest", "vq_bet"])
    parser.add_argument("--tokenizer_ckpt", type=str, required=True, help="Path to tokenizer full.pth")
    parser.add_argument("--base_vlm_ckpt", type=str, required=True, help="Path to base VLM checkpoint .pt")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--add_noise", action="store_true", help="Add noise in residual dimensions")
    parser.add_argument("--action_token_start", type=int, default=151665,
                        help="Start index of action tokens in embedding table")
    args = parser.parse_args()

    # 1. Extract codebook
    print(f"Extracting {args.tokenizer_type} codebook from {args.tokenizer_ckpt}")
    codebook = EXTRACTORS[args.tokenizer_type](args.tokenizer_ckpt)
    K, d_c = codebook.shape
    print(f"  Codebook: {K} vectors, {d_c} dims")

    # 2. Load base VLM embedding table
    print(f"Loading base VLM from {args.base_vlm_ckpt}")
    ckpt = torch.load(args.base_vlm_ckpt, map_location="cpu")
    embed_weight = ckpt["model"]["llm_backbone"]["llm.model.embed_tokens.weight"]
    print(f"  Embedding table: {embed_weight.shape}")

    # Text embeddings = everything before the action tokens
    text_embeddings = embed_weight[:args.action_token_start].float()
    print(f"  Text embeddings: {text_embeddings.shape}")

    # 3. Project codebook
    print("Projecting codebook to text embedding space...")
    projected = pca_aligned_projection(
        codebook, text_embeddings, add_residual_noise=args.add_noise)
    print(f"  Projected: {projected.shape}")

    # 4. Verify
    verify_projection(codebook, projected, text_embeddings)

    # 5. Replace action token embeddings in checkpoint
    action_end = args.action_token_start + K
    print(f"\n  Replacing embedding rows [{args.action_token_start}:{action_end}]")
    new_embed = embed_weight.clone()
    new_embed[args.action_token_start:action_end] = projected.to(embed_weight.dtype)

    new_ckpt = deepcopy(ckpt)
    new_ckpt["model"]["llm_backbone"]["llm.model.embed_tokens.weight"] = new_embed

    # 6. Save
    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"\n  Saved initialized checkpoint to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
