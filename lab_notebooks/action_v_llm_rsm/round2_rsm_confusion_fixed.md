# Round 2: RSM vs Confusion Matrix Correlations (FIXED APPROACH)

**Date**: 2026-03-18
**Status**: Ready to execute (scripts created, pending GPU runs)

## Overview

Redoing the RSM vs confusion matrix analysis with **three key fixes** to address previous methodology concerns:

### Previous Issues (Round 1)
1. **Confusion matrix construction**: Not clear how symmetric version was being computed
2. **Non-contextualized only**: Just bare word embeddings don't capture manipulation semantics
3. **Limited model coverage**: Only GPT-2 static + contextualized, missing modern LLMs
4. **Assumptions about symmetry**: Report mentions averaging bidirectional rates, but implementation unclear

### New Approach (Round 2)

#### 1. **Proper Symmetric Confusion Matrix**
Constructs a truly symmetric motor similarity matrix:
```
M[i,j] = (P(pred=j | true=i) + P(pred=i | true=j)) / 2
```
- Both elements [i,j] and [j,i] capture **bidirectional confusion rate**
- Example: if "open" → "pull" 20% of the time, and "pull" → "open" 15% of the time,
  then M[open,pull] = M[pull,open] = 0.175
- This measures: "how confused are verbs i and j with each other?"
- Guaranteed symmetric: `M = M^T` ✓

#### 2. **Two Types of Verb Embeddings**

**Non-contextualized** (baseline):
- Just the word embedding for the bare verb ("close", "grasp", etc.)
- What standard word embedding models capture
- Comparable to prior work

**Contextualized** (novel):
- Verb embedding extracted from CALVIN instruction context
- Example: "close the drawer" vs just "close"
- Verbs appear in different sentences → different hidden states
- Mean-pooled across all CALVIN instruction occurrences
- Should capture manipulation-specific verb semantics

#### 3. **Expanded Model Coverage**

| Model | Type | Non-ctx | Ctx | Notes |
|-------|------|:---:|:---:|-------|
| GPT-2 | LM (frozen embeddings) | ✓ | — | Baseline (no context in token embed) |
| LLaMA-2-7B | LLM (base) | ✓ | ✓ | Strong base LLM, no vision |
| LLaVA-1.5-7B | VLM (vision-tuned) | ✓ | ✓ | CLIP + Llama, uses text LM backbone |
| Qwen-2.5-7B | LLM (modern) | ✓ | ✓ | Qwen latest, multilingual |
| Qwen-VL | VLM (vision-tuned) | ✓ | ✓ | Qwen with ViT-bigG vision encoder |

Why these models?
- **LLaMA**: Open-source, widely used baseline for instruction following
- **LLaVA**: Controlled pair with LLaMA (same base, added vision)
- **Qwen**: Modern autoregressive LLM, different architecture from LLaMA
- **Qwen-VL**: Vision-language variant to test if visual pretraining helps

## Hypothesis

**H1 (symmetry)**: The fixed symmetric confusion matrix should make clearer patterns visible
**H2 (contextualization)**: Contextualized embeddings should have **higher** correlation with motor confusion
**H3 (vision grounding)**: VLMs (LLaVA, Qwen-VL) may show higher correlation if visual pretraining captures manipulation semantics
**H4 (language scale)**: Larger models (7B) may better capture verb distinctions than smaller ones

## Experimental Setup

### Confusion Matrix Source
- Checkpoint: `r8_ao_native_preds.json` (action-only native tokenizer, best validation)
- 683 validation episodes, 20 verb classes
- Model achieved 39.5% accuracy, 38.7% macro-F1

### Embedding Extraction

**Non-contextualized**:
```python
# For GPT-2: mean-pool token embeddings
"close" → [token_ids] → mean(embed[token_ids]) → embedding

# For LLMs: feed bare verb, take mean-pooled last hidden state
input: "close"
output: mean_pool(hidden_states[-1])
```

**Contextualized**:
```python
# For each verb, find all CALVIN instructions containing it
# Example for "close": ["close the drawer", "please close it", ...]
# For each instruction, extract hidden state at verb position
# Mean-pool across all occurrences
```

### RSM Computation
- Cosine similarity between verb embedding pairs
- Creates 20×20 symmetric matrix
- Diagonal = 1 (self-similarity), off-diagonal = verb pair similarity

### Correlation Metric
- **Spearman ρ**: Rank correlation (robust to scale differences)
- **Pearson r**: Linear correlation
- Computed on upper triangle of both symmetric matrices
- P-value indicates significance

## Expected Outcomes

### If hypothesis correct:
- **Contextualized > Non-contextualized**: Difference of ~0.2-0.3 in Spearman ρ
- **VLMs > non-visual LLMs**: VLMs may capture that "push" and "pull" differ in outcome
- **Symmetric matrix reveals patterns**: Can see verb clusters (synonym pairs, opposite pairs)

### If null result (ρ ≈ 0):
- Language embeddings don't correlate with motor confusion
- Motor verbs have idiosyncratic motion patterns not captured by text statistics
- Action verbs are fundamentally about **physics**, not linguistic statistics

## Scripts

### Execution
```bash
# Main analysis (extracts embeddings, computes correlations) — ~2 hours GPU time
python analysis/compute_rsm_confusion_correlations.py

# Visualization (creates comparison figures)
python figures/plot_rsm_confusion_v2.py
```

### Outputs
- `results/rsm_confusion_analysis_v2.npz` — confusion matrix, RSMs, embeddings
- `results/rsm_confusion_correlations_v2.json` — correlations table
- `figures/rsm_vs_confusion_v2_heatmaps.png` — 10 heatmaps side-by-side
- `figures/rsm_vs_confusion_v2_correlations.png` — bar chart of correlations
- `figures/rsm_vs_confusion_v2_detailed.png` — detailed 3-panel comparison

## Technical Notes

### Symmetric Confusion Matrix Correctness
The matrix is constructed as:
```python
cm_raw = confusion_matrix(true_labels, pred_labels)  # raw counts
cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)  # row-normalize
cm_sym = (cm_norm + cm_norm.T) / 2  # symmetrize
```

This means:
- `cm_sym[i,j]` = probability verb i and j are confused with each other
- **NOT** a covariance or correlation matrix (that's the RSM)
- Diagonal elements are typically high (verbs correctly classified)
- Off-diagonal high values = synonym/confusable verb pairs

### Contextualized Embedding Details
- Load up to 500 CALVIN training episodes
- For each verb, extract all instructions containing it
- Limit to 5 instructions per verb (memory constraints)
- Truncate instructions to 128 tokens
- Mean-pool last hidden state across all occurrences
- Result: one contextualized vector per verb

### Potential Issues & Mitigations
- **GPU memory**: Process one model at a time, free after each
- **Token imbalance**: Some verbs appear in more instructions than others
  - Mitigation: Mean-pool so each instruction weighted equally
- **Instruction variety**: CALVIN instructions are templated
  - Mitigation: Still better than bare words; shows robustness
- **Contextualized overhead**: 5× slower than non-contextualized
  - Mitigation: Only needed for modern LLMs, not GPT-2

## Comparison to Prior Work

| Aspect | Round 1 | Round 2 |
|--------|---------|---------|
| Confusion matrix | Unclear averaging | Explicit symmetric formula |
| Embeddings | Non-ctx only (GPT-2, 1 ctx) | 5 non-ctx + 4 ctx |
| Models | 2 (GPT-2 + Vicuna) | 5 (2 LMs + 3 VLMs) |
| Correlation | ρ ≈ 0 (GPT-2 static) | TBD |
| Interpretation | "null result, antonyms confounded" | Will clarify symmetry contribution |

## Roadmap
- [ ] Run `compute_rsm_confusion_correlations.py` (estimate 2 GPU hours)
- [ ] Generate visualizations with `plot_rsm_confusion_v2.py`
- [ ] Write findings notebook (pattern analysis)
- [ ] Compare to baseline methods if correlation remains near zero

## References
- Previous report: `figures/report/report.tex` (section on RSM vs confusion)
- Lab notebook: `lab_notebooks/action_v_llm_rsm/round1_verb_rsm_loglikelihood.md`
- Datasets: CALVIN v0 (3,404 training episodes, 27→21 sparse classes)
