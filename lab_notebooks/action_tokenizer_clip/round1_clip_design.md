# Round 1: CLIP-Style Contrastive Action–Language Alignment

**Date**: 2026-03-15
**Goal**: Design a contrastive learning approach (CLIP-style) to align quantized action latents with language instructions, replacing the classification head.

## Motivation

The current verb-decodable tokenizer uses a classification head over quantized action latents to predict one of 20 verb classes. This approach has three fundamental limitations:

1. **Single-verb assumption**: Each episode is labeled with exactly one verb. Instructions like "pick up the block and place it on the shelf" contain two verbs, but the model must choose one. CALVIN instructions are pre-filtered to single-verb sentences, but this limits generalization.
2. **Loss of context**: "push the red block left" and "push the blue block right" both map to class `push`. The classification head discards the object, direction, and spatial context — all of which shape the action trajectory.
3. **Semantic blindness**: "turn the knob" and "rotate the knob" are forced into separate classes despite being semantically identical. The model learns an artificial distinction that doesn't exist in the action space.

A CLIP-style contrastive loss addresses all three:
- The text encoder processes the **full instruction** — multi-verb, contextual, and semantically rich.
- Similarity is computed in a continuous embedding space, so semantically close instructions naturally cluster.
- No fixed class taxonomy — the model learns to associate actions with language directly.

### Data recovery: +1,432 training episodes

The classification approach requires spaCy verb extraction and filters out any instruction that doesn't have exactly one verb. This discards **1,432 of 5,124 episodes (28%)** from the CALVIN D→D training set:

| Category | Episodes | Unique instructions | Examples |
|----------|----------|---------------------|----------|
| 0 verbs (spaCy parse failures) | 337 | 25 | "push left the pink block", "in the slider grasp the red block" |
| 2+ verbs (multi-step) | 1,095 | 86 | "grasp the blue block and lift it up", "grasp the drawer handle and open it" |
| 3+ verbs | 23 | — | (subset of above) |
| **Total filtered out** | **1,432** | **111** | |

The 0-verb cases aren't actually verbless — spaCy fails on unusual word order ("push left the X") and prepositional fronting ("in the slider grasp the X"). The 2+ verb cases are legitimate multi-step instructions that describe the full action sequence.

With CLIP, no verb extraction or filtering is needed — the text encoder processes raw instruction strings directly:

| | Classification approach | CLIP approach |
|---|---|---|
| Train episodes | 3,692 (after filtering) | **5,124** (all raw) |
| Unique instructions | 254 | **389** |
| Val episodes | 674 | **1,011** |

### Architecture overview

```
Action branch:   actions → VQ-VAE encoder → quantize (STE) → Transformer → project → L2-norm → a_emb (D)
Language branch: instruction → Text Encoder → project → L2-norm → t_emb (D)

Loss: InfoNCE(a_emb, t_emb) + recon_loss + vq_loss
```

Within a batch of N (action, instruction) pairs, the contrastive loss treats the diagonal as positives and all off-diagonal pairs as negatives. A learnable temperature parameter scales the logits.

---

## Design Decision 1: Text Encoder

The key question: does **vision-text contrastive pretraining** (CLIP) produce text representations that transfer well to **action-text alignment**, compared to a text encoder pretrained only on language (MLM)?

Three conditions to compare:

### Condition A: Frozen LAION CLIP text encoder (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`)

| Property | Value |
|----------|-------|
| Architecture | Causal (masked) self-attention transformer |
| Layers | 12 |
| Hidden dim | 512 |
| FFN dim | 2,048 |
| Attention heads | 8 |
| Vocab size | 49,408 (BPE) |
| Max seq length | 77 |
| Output dim | 512 (after text_projection) |
| Parameters | 63.2M (encoder) + 262K (projection) |
| Activation | quick_gelu |
| Pretraining | Contrastive on LAION-2B (2 billion image-text pairs) |
| Pooling | EOS token position (pooler_output) |

We use LAION CLIP over OpenAI CLIP because it was trained on 5× more data (2B vs 400M pairs) and is the same architecture (non-distilled, trained from scratch). All non-distilled CLIP text encoders share this same "Base" architecture — 63.2M params is the smallest available. MetaCLIP (Meta) uses the same architecture trained on 400M curated pairs.

**Pros**: Trained with contrastive objective on the largest public image-text dataset. The embedding space is shaped for cross-modal matching — semantically similar instructions will be close. Frozen = 63M params don't cause overfitting; the text encoder is just a fixed feature extractor.

**Cons**: Frozen = can't adapt to CALVIN-specific action-language distinctions. Trained for image-text alignment, not action-text. "push" and "slide" may be close in CLIP space even though they correspond to different kinematics.

**Implementation**: Use `CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")`, extract `model.text_model` and `model.text_projection`. Freeze all parameters. Tokenizer: `CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")` (same BPE vocab).

### Condition B: LAION CLIP + LoRA

Same LAION CLIP text encoder, but with LoRA adapters on the attention layers to allow lightweight adaptation to action-text alignment.

Full fine-tuning 63M params on 5,124 samples would massively overfit. LoRA (Hu et al., 2022) adds low-rank adapters to the query and value projections, keeping the vast majority of parameters frozen:

| LoRA rank | Trainable params | % of 63.2M |
|-----------|-----------------|-------------|
| r=4 | 98K | 0.16% |
| r=8 | 197K | 0.31% |
| r=16 | 393K | 0.62% |

**Start with r=8** (197K trainable params). This allows the text encoder to learn CALVIN-specific distinctions (e.g., separate "push" from "slide" if their action patterns differ) without risk of memorizing the ~389 unique instructions.

**What this tests vs Condition A**: Does adapting the text encoder to our action domain help, or are CLIP's frozen representations already good enough? If LoRA significantly improves over frozen, it means CLIP's image-text embedding space needs adjustment for action-text alignment.

**Implementation note**: `peft` is not installed in the mmml env. We can either `pip install peft` or implement LoRA manually — it's just two small matrices per attention layer:
```
W' = W + BA    where B: (512, r), A: (r, 512), r << 512
```

### Condition C: Frozen GPT-2 (`gpt2`)

A causal LM pretrained only on text, for comparison against CLIP's contrastively-trained text encoder.

| Property | LAION CLIP text | GPT-2 |
|----------|-----------------|-------|
| Layers | 12 | 12 |
| Hidden dim | 512 | 768 |
| FFN dim | 2,048 | 3,072 |
| Attention heads | 8 | 12 |
| Vocab size | 49,408 (BPE) | 50,257 (BPE) |
| Max seq length | 77 | 1,024 |
| Output dim | 512 | 768 |
| Parameters | 63.2M | 124.4M |
| Activation | quick_gelu | gelu |
| Pretraining | Contrastive (LAION-2B, 2B image-text) | Autoregressive LM (WebText, 40GB web pages) |
| Attention type | Causal | Causal |
| Pooling | EOS token | Last token |

GPT-2 matches CLIP on the two most important axes: **same layer count** (12) and **same attention type** (causal). It has a wider hidden dim (768 vs 512), making it 2× the params — but since both are frozen this doesn't affect overfitting risk. Both use BPE tokenization with similar vocab sizes.

**What this comparison tests**: Both CLIP and GPT-2 are 12-layer causal transformers. The key difference is pretraining objective: CLIP learned from **vision-text contrastive alignment** (2B image-text pairs), while GPT-2 learned **next-token prediction** on text alone (WebText). If Condition A >> C, it means vision-text alignment transfers to action-text alignment — CLIP's embedding space is shaped for cross-modal matching in a way that autoregressive LM representations are not.

**Pooling note**: GPT-2 has no built-in [EOS] pooling like CLIP. We take the last-token representation (analogous to CLIP's EOS pooling for causal models). The text projection is Linear(768→D).

### Condition D: GPT-2 + LoRA

Same GPT-2 text encoder with LoRA adapters, mirroring Condition B.

| LoRA rank | Trainable params | % of 124.4M |
|-----------|-----------------|-------------|
| r=4 | 147K | 0.12% |
| r=8 | 295K | 0.24% |
| r=16 | 590K | 0.47% |

(GPT-2's q/v projections are 768×768, so LoRA params are slightly larger than CLIP's 512×512.)

**What this enables**: A full 2×2 comparison:

|  | Frozen | + LoRA |
|--|--------|--------|
| **CLIP** (vision-text contrastive) | A | B |
| **GPT-2** (autoregressive LM) | C | D |

- A vs B: does LoRA adaptation help CLIP?
- C vs D: does LoRA adaptation help GPT-2?
- A vs C: does contrastive pretraining help (frozen)?
- B vs D: does contrastive pretraining help (with adaptation)?

If CLIP+LoRA (B) ≈ GPT-2+LoRA (D), it means adaptation washes out the pretraining advantage — both converge when given task-specific signal. If B >> D, contrastive pretraining provides a lasting advantage even after adaptation.

### Initial plan

Start with **Condition A** (frozen LAION CLIP) as the baseline — simplest setup. Then run the full 2×2 grid (A/B/C/D) to disentangle the effects of contrastive pretraining and LoRA adaptation.

---

## Design Decision 2: Action Tokenizer (VQ-VAE Architecture)

### Current: Custom ActionVQVAE

Our scratch-built VQ-VAE is a minimal 2-layer MLP:

```
Encoder: Linear(28→128) → ReLU → Linear(128→64)     [chunk_size=4, action_dim=7]
Codebook: nn.Embedding(512, 64)                       [single VQ, Euclidean distance]
Decoder: Linear(64→128) → ReLU → Linear(128→28)
Total: 56.7K parameters
```

**How it works**:
1. A trajectory of T≈61 timesteps is split into non-overlapping 4-step chunks → ~15 chunks
2. Each chunk (4×7=28-d vector) is encoded to a 64-d latent
3. Nearest codebook entry is found (Euclidean distance)
4. Straight-through estimator passes gradients through quantization
5. Decoder reconstructs the 28-d chunk from the quantized latent

**Is this standard?** Partially. Standard VQ-VAEs (van den Oord et al., 2017) use convolutional encoders/decoders for images and audio. For sequential data like actions, the standard approach would be 1D causal convolutions (like SoundStream/EnCodec for audio). Our MLP-based design:
- **Ignores temporal structure within chunks**: The 4 timesteps are flattened to a 28-d vector, so the encoder sees no ordering. A permutation of the 4 timesteps would give the same encoding.
- **Ignores cross-chunk context**: Each chunk is encoded independently. Chunk i knows nothing about chunk i-1 or i+1.
- **Has no normalization**: No LayerNorm, GroupNorm, or BatchNorm. Standard VQ-VAEs use GroupNorm.
- **Uses ReLU**: Standard modern VQ-VAEs use SiLU/Swish.

These are reasonable simplifications for a 28-d input, but they limit the model's capacity to capture temporal patterns.

### VQ-VLA Architecture

The pretrained VQ-VLA (from VQ-VLA paper) is a much more sophisticated design:

```
Encoder:
  CausalConv2d(in→128, k=3)
  DownBlock(128→128, 4 ResNet layers) → stride-2 downsample
  DownBlock(128→256, 4 ResNet layers) → stride-2 downsample
  DownBlock(256→256, 4 ResNet layers)
  DownBlock(256→512, 4 ResNet layers)
  4 × CausalResnetBlock2D(512→512)
  GroupNorm → SiLU → CausalConv2d(512→128)

Quantizer: Residual VQ (4 stages × 256 codes × 128-d)

Decoder: (symmetric to encoder, with UpBlocks)

Total: ~3–4.5M parameters
```

**Key differences from our ActionVQVAE:**

| Aspect | ActionVQVAE | VQ-VLA |
|--------|-------------|--------|
| Encoder | 2-layer MLP | 16+ layer CausalConvNet |
| Temporal modeling | None (flattened chunks) | Causal convolutions |
| Quantization | Single VQ (512 codes, 64-d) | Residual VQ (4×256 codes, 128-d) |
| Tokens per trajectory | ~15 | ~48 |
| Input preprocessing | Raw 7-d actions | Action-type PE (21-d) + temporal PE |
| Normalization | None | GroupNorm(32) |
| Activation | ReLU | SiLU |
| Parameters | 56.7K | ~3–4.5M |
| Codebook capacity | 512 entries | 256^4 ≈ 4.3B combinations |

**Residual VQ** is the biggest architectural difference. Instead of one codebook, VQ-VLA uses 4 sequential quantizers. The first quantizer captures the coarsest structure; each subsequent quantizer encodes the residual (what the previous ones missed). This is the same technique used in audio codecs (SoundStream, EnCodec). It gives much finer reconstruction without exponentially growing the codebook.

### Sizing a smaller VQ-VLA

The biggest param lever is channel width (params scale as O(channels²)). We explored a range of configurations:

| Config | Stages | ResNet/stage | Channels | Total |
|--------|--------|-------------|----------|-------|
| ActionVQVAE (MLP) | — | — | 128 hidden | **0.06M** |
| 2 stages, 2 resnet | 2 | 2 | 32→64 | **0.9M** |
| 3 stages, 1 resnet | 3 | 1 | 32→64→128 | **2.3M** |
| **3 stages, 2 resnet** | **3** | **2** | **32→64→128** | **3.7M** |
| 4 stages, 2 resnet | 4 | 2 | 32→64→64→128 | **4.2M** |
| Small (4 stages, 2 resnet) | 4 | 2 | 64→128→128→256 | **15.8M** |
| Full VQ-VLA | 4 | 4 | 128→256→256→512 | **113M** |

### Three VQ-VAE conditions

#### VQ-VAE Option 1: Tiny VQ-VLA (trained from scratch) — **3.7M params**

```
Encoder:
  CausalConv2d(1→32, k=3)
  DownBlock(32→32, 2 ResNet layers) → stride-2 downsample
  DownBlock(32→64, 2 ResNet layers) → stride-2 downsample
  DownBlock(64→128, 2 ResNet layers)
  2 × CausalResnetBlock2D(128→128)
  GroupNorm → SiLU → CausalConv2d(128→64)

Quantizer: Residual VQ (2 stages × 256 codes × 64-d)

Decoder: (symmetric, with UpBlocks)

Total: ~3.7M parameters
```

Preserves the key VQ-VLA innovations (causal convolutions, ResidualVQ, GroupNorm, SiLU) at a scale appropriate for 5.1K episodes. 3 down/up stages with 2 ResNet blocks each gives enough depth for temporal modeling without being oversized.

- Tokens per trajectory: 12 windows × 2 codes = **24 tokens** (vs 48 for full VQ-VLA, vs 15 for ActionVQVAE)
- Trained from scratch jointly with the CLIP contrastive loss

#### VQ-VAE Option 2: Full VQ-VLA + LoRA — **113M frozen + ~300-600K trainable**

Use the pretrained full VQ-VLA (113M params), freeze everything, and add LoRA adapters to adapt it to our contrastive objective.

| Strategy | Trainable params | % of 113M |
|----------|-----------------|-----------|
| Codebooks only | 131K | 0.12% |
| LoRA r=4 on ResNet convs | 283K | 0.25% |
| LoRA r=8 on ResNet convs | 565K | 0.50% |
| Codebooks + LoRA r=4 | 414K | 0.37% |

**Pros**: Starts from a pretrained action tokenizer trained on diverse robot datasets (Open X-Embodiment, LIBERO, RH20T, etc.). The encoder already knows how to compress action sequences. LoRA adapts it to our contrastive objective without catastrophic forgetting.

**Cons**: 113M params is large even if mostly frozen — slower forward pass, more GPU memory. LoRA on Conv2d is less standard than on Linear layers (though it works the same way). The pretrained VQ-VLA was trained for reconstruction only — its codebook may not preserve verb-discriminative features.

**Implementation**: Freeze all VQ-VLA parameters, add LoRA to the 3×3 Conv2d layers in ResNet blocks. Fine-tune codebooks are also trainable. Start with r=4 (414K total trainable).

### Decision

**Primary**: Tiny VQ-VLA (Option 1) — trained from scratch with our contrastive loss. The entire action branch learns end-to-end to produce representations useful for language alignment.

**Ablation**: Full VQ-VLA + LoRA (Option 2) — tests whether starting from a pretrained action tokenizer and adapting with LoRA beats training from scratch.

---

## Design Decision 3: Projection Dimension

The projection dimension D is the shared space where action and text embeddings are compared. Both branches project to D dimensions and L2-normalize before computing cosine similarity.

### Considerations

**Small D (64–128)**:
- Forces a highly compressed representation — good for generalization, bad if the space is too small to capture all necessary distinctions.
- CALVIN has ~254 unique instructions mapping to 20 verb classes. A 64-d space can easily separate 35 points (need at most 34 dimensions for 35-point linear separability). So D=64 might suffice for this dataset.
- Matches our VQ-VAE latent dimension (64) and transformer d_model (128), keeping the architecture balanced.

**Large D (256–512)**:
- More room for fine-grained distinctions. Useful if instructions carry rich contextual information beyond just the verb.
- But with ~3,300 training samples and ~254 unique instructions, a large D risks overfitting — the model can find trivial solutions in the high-dimensional space.
- CLIP uses 512 because it operates over millions of image-text pairs with enormous visual and linguistic diversity.

**What CLIP papers do**: CLIP ViT-B/32 uses D=512. CLIP ViT-L uses D=768. But these train on 400M pairs. Smaller contrastive models (e.g., for audio-text) often use D=128 or D=256.

### Decision

**D=128** as default. Rationale:
- Matches our action transformer d_model, so the action branch projection is a simple linear layer (no dimension change needed).
- For frozen CLIP (512-d output), the text projection is Linear(512→128). For frozen GPT-2 (768-d output), it's Linear(768→128). Both compress to the same shared space.
- Large enough to separate 254 instructions with margin; small enough to avoid overfitting with 5.1K samples.

We can sweep D ∈ {64, 128, 256} as a hyperparameter if needed.

---

## Design Decision 4: Training Loss

### Combined loss

```
L_total = L_recon + L_vq + lambda_clip * L_contrastive
```

- **L_recon** (MSE): Ensures the VQ-VAE still reconstructs actions faithfully. Without this, the codebook could collapse to only encode verb-relevant features and lose fine-grained action information needed for downstream VLA.
- **L_vq** (codebook + commitment): Standard VQ training objective. Keeps codebook entries well-utilized.
- **L_contrastive** (symmetric InfoNCE): Aligns action and text embeddings.

### InfoNCE details

For a batch of N pairs {(a_i, t_i)}:

```
sim(i,j) = (a_i^T t_j) / tau          # cosine similarity scaled by temperature

L_a2t = -1/N * sum_i log( exp(sim(i,i)) / sum_j exp(sim(i,j)) )    # action→text
L_t2a = -1/N * sum_i log( exp(sim(i,i)) / sum_j exp(sim(i,j)) )    # text→action

L_contrastive = (L_a2t + L_t2a) / 2
```

**Temperature (tau)**: Learnable scalar, initialized to 0.07 (CLIP default), clamped to [0.01, 1.0]. Lower tau = sharper distribution = harder negatives.

### Handling duplicate instructions

CALVIN has many episodes with identical instructions. In a batch of 64, you might have 5 episodes all labeled "push the red block to the left". These are all valid positives for each other, but vanilla InfoNCE treats them as negatives.

**Options**:
1. **Ignore it**: Standard CLIP also has near-duplicate captions. The model still learns — it just gets a softer gradient for duplicates since their similarity will be high regardless.
2. **Multi-positive InfoNCE**: Modify the loss to treat all episodes with the same instruction as positives. More correct but adds complexity.
3. **Instruction-level batching**: Ensure each batch has at most 1 episode per unique instruction. Maximizes negative diversity but limits batch size to ~35.

**Decision**: Start with option 1 (ignore it). CLIP worked fine with caption duplicates at scale. If we observe that training loss plateaus early, revisit with option 2.

---

## Implementation Plan

### Files to create/modify

1. **`tokenization/clip_action_language.py`** (new) — main module:
   - `ActionLanguageCLIP` class: wraps VQ-VAE + action transformer + text encoder + projections
   - `CalvinCLIPDataset`: loads (action_chunks, instruction_string, n_chunks) tuples
   - `fit_clip_tokenizer()`: training loop with combined loss
   - `encode_actions()` / `encode_text()`: inference-time embedding functions

2. **`run_clip_tokenizer.sh`** (new) — SLURM submission script

### Evaluation metrics

**Primary metric: Continuous action L1** (same as `lab_notebooks/action_tokenizer_cls_head/round2_openvla_finetune.md`).
Fine-tune OpenVLA-mini with our CLIP-aligned tokenizer, then teacher-force on CALVIN val:
decode predicted tokens → continuous actions → mean |pred − GT| in original action space.

Baseline results from cls_head experiments:

| Condition | Action L1 ↓ |
|-----------|------------|
| bin (standard ActionTokenizer) | **0.162** |
| vq_vanilla (λ=0) | 0.232 |
| vq_verb λ=0.1 | 0.186 |
| vq_verb λ=0.5 | 0.196 |
| CLIP (this work) | ? |

**Training diagnostics** (evaluated on tokenizer, no VLA needed):
- Val contrastive loss (is the alignment learning?)
- Val reconstruction MSE (is the VQ-VAE still reconstructing well?)
- Embedding visualization: t-SNE/UMAP of action embeddings colored by verb class

Note: retrieval R@1 is not a good metric here — action trajectories can't distinguish object identity (e.g., "push the red block" vs "push the blue block" have identical actions), so exact instruction retrieval would be unfairly penalized.

### Hyperparameter starting point

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| VQ-VAE | Tiny VQ-VLA, 3 stages 32→64→128, 2 ResidualVQ, 3.7M | From scratch |
| Action transformer | 2 layers, d_model=128, 4 heads | Same as verb-decodable |
| Text encoder | frozen LAION CLIP B/32, 512-d | Condition A first |
| Projection dim D | 128 | Matches d_model |
| Temperature init | 0.07 | CLIP default |
| lambda_clip | 1.0 | Equal weight to contrastive |
| Batch size | 64 | Reasonable for InfoNCE (64 negatives) |
| Learning rate | 1e-3 | Adam, for action branch |
| Epochs | 200 | Same as VQ-VAE training |
| Data filtering | None (use all 5,124 episodes) | No verb extraction needed |

---

## Experiment Plan

Two phases: (1) train CLIP-aligned tokenizers, (2) fine-tune OpenVLA-mini and evaluate.

### Phase 1: Train CLIP-aligned tokenizers (~12h each)

Each job trains a Tiny VQ-VLA from scratch with contrastive + reconstruction loss.
Monitor val contrastive loss and val reconstruction MSE during training.

**Wave 1** — Submit all 4 text encoder conditions in parallel:

| Job | Text encoder | VQ-VAE | Purpose |
|-----|-------------|--------|---------|
| 1.1 | CLIP frozen (A) | Tiny VQ-VLA | Baseline — does vision-text alignment transfer to action-text? |
| 1.2 | CLIP + LoRA r=8 (B) | Tiny VQ-VLA | Does adapting CLIP help? (A vs B) |
| 1.3 | GPT-2 frozen (C) | Tiny VQ-VLA | Does contrastive pretraining matter? (A vs C) |
| 1.4 | GPT-2 + LoRA r=8 (D) | Tiny VQ-VLA | Full 2×2 grid completion |

Submit script: `scripts/clip_tokenizer_submit.sh`

**Wave 2** — VQ-VAE ablation (submit after Wave 1 identifies best text encoder):

| Job | Text encoder | VQ-VAE | Purpose |
|-----|-------------|--------|---------|
| 1.5 | Best from Wave 1 | Full VQ-VLA + LoRA | Does pretrained VQ-VLA beat from-scratch? |

### Phase 2: Fine-tune OpenVLA-mini (~24h each)

Pick the top 2–3 tokenizers from Phase 1 (by val contrastive loss + val recon MSE).
Fine-tune OpenVLA-mini with each tokenizer, then evaluate teacher-forcing action L1 on CALVIN val.

| Job | Tokenizer | Purpose |
|-----|-----------|---------|
| 2.1 | Best overall from Phase 1 | Primary CLIP result |
| 2.2 | Second-best from Phase 1 | Runner-up comparison |
| 2.3 | vq_vanilla (λ=0, no CLIP) | Control — same Tiny VQ-VLA arch, recon-only, no contrastive |

Job 2.3 is critical: it isolates the effect of CLIP loss from the VQ-VLA architecture change.
Without it, improvements could be from the better VQ-VAE rather than contrastive alignment.

### Decision gates

- **After Wave 1**: Compare the 2×2 grid on val contrastive loss and val recon MSE. Pick best text encoder for Wave 2 and Phase 2.
- **After Phase 2**: Compare action L1 against cls_head baselines (0.162–0.232). If CLIP tokenizer beats vq_verb (0.186), strong positive result.

### Total compute estimate

- Phase 1: 5 jobs × ~12h = 60 GPU-hours
- Phase 2: 3 jobs × ~24h = 72 GPU-hours
- Total: ~132 GPU-hours (5–7 days wall time with parallel jobs)

---

## Open Questions

1. **Batch size sensitivity**: InfoNCE benefits from large batches (more negatives). With 5,124 training samples, batch_size=64 gives 80 batches/epoch. Should we use a memory bank or momentum encoder (MoCo-style) to increase effective negatives?

2. **Instruction diversity**: CALVIN has 389 unique instructions across 5,124 episodes. Some verbs have many phrasings ("pick up" has 35), others have few. Should we augment instructions (paraphrasing, template variations)?

3. **Rollout evaluation**: The gold standard for VLA is rollout task success rate (SR1–SR5). This requires a CALVIN simulator and is expensive. Plan to run after action L1 shows promising results.
