# Round 1: Verb RSM & Log-Likelihood Across LMs and VLMs

**Date**: 2026-03-18
**Goal**: Test whether static vision grounding (from VLM training) improves verb representation alignment with motor similarity, or whether it's just text familiarity with manipulation language.

## Motivation

CLIP and SigLIP show nonzero RSM correlation with motor similarity, while pure LLMs show ~0. But is this because:
- (a) Visual grounding teaches physical verb similarity, or
- (b) Manipulation-related text is more common in VLM training corpora?

We can disentangle by measuring **log-likelihood** of CALVIN instructions under autoregressive models. If a VLM has similar log-likelihood to its LM backbone but higher RSM correlation, the visual grounding is doing the work.

## Models

| Model | Type | Params | Visual training | LL measurable |
|-------|------|--------|:-:|:-:|
| GPT-2 | LM | 117M | no | yes |
| Qwen2.5-0.5B | LM | 0.5B | no | token embed only |
| Vicuna-7B-v1.5 | LM (chat) | 7B | no | yes |
| LLaVA-1.5-7B | VLM | 7B | yes (CLIP ViT-L + Vicuna) | yes |
| Qwen-VL-Chat | VLM | 9.6B | yes (ViT-bigG) | yes |
| CLIP ViT-B/32 | contrastive VLM | 151M | yes | no (not autoregressive) |
| SigLIP-SO400M | contrastive VLM | 400M | yes | no (not autoregressive) |

Key controlled pair: **Vicuna vs LLaVA** — same LM backbone, only difference is visual training.

## Verb Embeddings (RSM)

Embedding extraction method:
- Token embed models (GPT-2, Qwen2.5-0.5B): mean-pool input token embeddings
- Autoregressive models (Vicuna, LLaVA, Qwen-VL): mean-pool last hidden state
- Contrastive models (CLIP, SigLIP): text encoder output

20 CALVIN verbs (sparse set, excluding "left").

### RSM Inter-Model Spearman Correlations

```
                      gpt2  qwen2.5-0.5b  clip-vit-b-32  siglip-so400m  llava-1.5-7b  qwen-vl-chat  vicuna-7b
gpt2                 1.000         0.321          0.264          0.095        -0.150        -0.187     -0.018
qwen2.5-0.5b         0.321         1.000          0.319          0.099         0.392         0.449      0.491
clip-vit-b-32         0.264         0.319          1.000          0.085         0.042         0.291      0.165
siglip-so400m         0.095         0.099          0.085          1.000        -0.087         0.011     -0.053
llava-1.5-7b         -0.150         0.392          0.042         -0.087         1.000         0.589      0.878
qwen-vl-chat         -0.187         0.449          0.291          0.011         0.589         1.000      0.595
vicuna-7b            -0.018         0.491          0.165         -0.053         0.878         0.595      1.000
```

Mean off-diagonal cosine similarity (within each model's RSM):
- GPT-2: token embed (low-d, high sim expected)
- Vicuna-7B: 0.7741
- LLaVA-1.5-7B: 0.7543
- Qwen-VL-Chat: 0.8003

### Key RSM observations

1. **Vicuna ≈ LLaVA (rho=0.878)**: Visual training barely changed verb similarity structure at the last hidden state level
2. **CLIP is uncorrelated with LLaVA (0.042)**: Contrastive vs autoregressive architectures produce very different similarity structures
3. **SigLIP is uncorrelated with everything**: Unique similarity structure, possibly due to sigmoid loss
4. **Qwen-VL moderately correlates with CLIP (0.291)**: More than LLaVA does — possibly because Qwen-VL's ViT-bigG is larger

## Log-Likelihood on CALVIN Instructions

259 unique CALVIN instructions. Computed per-token log-likelihood under each autoregressive model.

| Model | Visual grounding | Mean LL | Perplexity | Avg tokens |
|-------|:-:|:-:|:-:|:-:|
| GPT-2 (117M) | no | -5.44 | 229.9 | 6.3 |
| Vicuna-7B | no | -5.05 | 155.7 | 7.6 |
| **LLaVA-1.5-7B** | **yes** | **-5.01** | **149.6** | 7.6 |
| Qwen-VL-Chat | yes | -5.94 | 381.6 | 6.2 |

### Key log-likelihood observations

1. **Vicuna ≈ LLaVA on perplexity** (155.7 vs 149.6): CALVIN text is equally in-distribution for both. Visual training did not make LLaVA more familiar with manipulation language.
2. **Qwen-VL has highest perplexity** (381.6): CALVIN English text is actually *less* in-distribution for Qwen-VL, likely due to Chinese-centric pretraining and different tokenizer.
3. Token count difference (6.2 vs 7.6) reflects different tokenizers — Qwen uses fewer tokens for the same text.

## Interpretation

The Vicuna-LLaVA controlled pair gives the cleanest test:
- **Same perplexity** → same text familiarity with CALVIN instructions
- **Same RSM structure (rho=0.878)** → visual training did NOT change verb representations in the LM backbone

This means for autoregressive VLMs, the text-only verb embeddings from the LM backbone don't benefit from visual grounding. The visual information is likely encoded in the cross-modal interaction (when images are present), not baked into the text representations.

**Contrast with CLIP/SigLIP**: These contrastive models show different (potentially more motor-aligned) verb similarity because they were trained with a fundamentally different objective — aligning text with images — which restructures the text embedding space itself.

## TODO

- [ ] Compute motor similarity RSM from action trajectories (action-only classifier confusion matrix or trajectory distance)
- [ ] Correlate each model's verb RSM with motor RSM (the actual hypothesis test)
- [ ] Visualize RSMs as heatmaps side by side

## Scripts

- `analysis/compute_verb_rsm.py` — verb embedding extraction + RSM computation for all 7 models
- `analysis/compute_loglikelihood.py` — per-instruction log-likelihood for autoregressive models
- Results: `results/verb_rsms.npz`, `results/verb_embeds.npz`, `results/instruction_loglikelihoods.json`
