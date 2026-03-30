# VLA Action Token Embedding Initialization from Tokenizer Codebook

## Problem

MiniVLA adds 256 extra tokens to the Qwen2.5-0.5B vocabulary for action tokenization.
These tokens are initialized via `resize_token_embeddings`, which sets all 256 embeddings
to the **mean of existing text embeddings** — making them nearly identical (cosine sim ~0.99).

After 50K steps of policy training, the action token embeddings barely differentiate
(cosine sim drops to ~0.94). A verb probe on these embeddings gets ~2% MF1 (random).
Meanwhile, the tokenizer's encoder latents achieve 28–43% MF1 on the same verb probe.

## Goal

Initialize the VLA's 256 action token embeddings from the tokenizer's **codebook vectors**,
projected into the LLM embedding space with statistics that match existing text embeddings.
This gives the LLM geometrically meaningful action tokens from the start — codes that
produce similar actions will have similar embeddings, and the overall distribution is
"in range" for the LLM's trained layers.

## Method: PCA-Aligned Distribution Matching

### Inputs

- **Codebook vectors** `C ∈ R^{K × d_c}`: one vector per discrete code.
  - QueST (FSQ): enumerate all FSQ grid points → quantized latent vectors. K = codebook_size
    (e.g., 256 for [4,4,4,4]), d_c = number of FSQ dimensions (e.g., 4).
  - VQ-BeT (ResidualVQ): extract learned codebook embeddings. K = num_codes per group,
    d_c = latent_dim per group.
- **Text embeddings** `T ∈ R^{N × d_e}`: the LLM's existing token embeddings.
  N = 151,665 (base Qwen2.5 tokens before extra tokens), d_e = 896.

### Step 1: Codebook PCA

Center and decompose the codebook vectors:

```
μ_c = mean(C, dim=0)                          # (d_c,)
C_centered = C - μ_c                          # (K, d_c)
U_c, σ_c, V_c^T = SVD(C_centered)            # V_c: (d_c, d_c), σ_c: (d_c,)
```

V_c columns are the codebook's principal directions. σ_c are the singular values
(proportional to std in each direction).

### Step 2: Text Embedding PCA

Center and decompose the text embeddings:

```
μ_t = mean(T, dim=0)                          # (d_e,)
T_centered = T - μ_t                          # (N, d_e)
U_t, σ_t, V_t^T = SVD(T_centered)            # V_t: (d_e, d_e), σ_t: (d_e,)
```

V_t columns are the text embedding's principal directions.

### Step 3: Aligned Projection

Map codebook vectors into the text embedding space by aligning principal components:

1. **Project to codebook PCA coordinates**: `Z = C_centered @ V_c`  — shape (K, d_c)
2. **Rescale each PC** to match text variance: `Z_scaled = Z @ diag(σ_t[:d_c] / σ_c)`
   - The i-th codebook PC direction gets scaled by (text_σ_i / codebook_σ_i)
   - This ensures the projected codebook has the same variance as text embeddings
     in each of the top d_c principal directions
3. **Rotate to text PCA basis**: `C_proj = Z_scaled @ V_t[:, :d_c]^T`  — shape (K, d_e)
4. **Add text mean**: `C_proj = C_proj + μ_t`

The final projected codebook `C_proj ∈ R^{K × d_e}` has:
- **Same mean** as text embeddings (μ_t)
- **Same covariance** as text embeddings in the top d_c directions
- **Preserved internal geometry**: pairwise angles between codebook vectors are
  maintained (only rotation + per-axis scaling applied)

### Step 4: Handle Dimensionality Gap

Since d_c < d_e, the projection only occupies d_c of d_e dimensions.
The remaining d_e - d_c dimensions are set to μ_t (zero in the centered space).

Option: add small Gaussian noise in the remaining dimensions, scaled to
match the text variance in those directions. Start without noise (simpler)
and add if needed.

### Step 5: Write Initialized Checkpoint

1. Load the base VLM checkpoint (pre-policy-training)
2. Replace embedding rows at indices 151,665–151,920 with C_proj
3. Save as a new checkpoint
4. Fine-tune policy from this checkpoint

## Codebook Extraction

### QueST (FSQ)

FSQ quantizes each latent dimension independently to L levels.
For levels [4,4,4,4], the grid has 4^4 = 256 points.
Each grid point IS the codebook vector — it's the quantized latent value.

FSQ level L maps to values: `{-1, -1+2/(L-1), ..., 1-2/(L-1), 1}` for odd L,
or the "implicit" grid from `round(x * (L-1)/2) / ((L-1)/2)`.

Enumerate all K grid points → C ∈ R^{K × d_c} where d_c = len(fsq_levels).

Note: d_c is small (4 for [4,4,4,4]). The PCA alignment will map these 4
dimensions into the most important 4 directions of the 896-d text space.

### VQ-BeT (ResidualVQ)

```python
codebook = model.vqvae.vq_layer._codebook.embed  # (groups, num_codes, dim_per_group)
```

For group-offset encoding: each group has its own codebook. Project each group's
codebook independently, OR concatenate group codebooks and project jointly
(depends on how the policy maps codes → token IDs).

## Verification

After projection, verify:
1. **Norm distribution**: action token norms match text token norms
2. **Pairwise cosine similarity**: action tokens have low cosine sim (~0.05–0.10 like text),
   not 0.95
3. **Distance preservation**: Spearman correlation between codebook pairwise distances and
   projected pairwise distances ≈ 1.0
4. **Verb probe**: run verb probe on projected vectors — should be at least as good as
   probing raw codebook vectors

## Usage

```bash
python vla_embed_init/project_codebook.py \
    --tokenizer_type quest \
    --tokenizer_ckpt checkpoints/calvin_sweep/tokenizers/quest_16_4444_4/full.pth \
    --base_vlm_ckpt <base_vlm>/checkpoints/latest-checkpoint.pt \
    --output_ckpt checkpoints/calvin_sweep/policy/minivla_quest_16_4444_4_cbinit/base_init.pt
```
