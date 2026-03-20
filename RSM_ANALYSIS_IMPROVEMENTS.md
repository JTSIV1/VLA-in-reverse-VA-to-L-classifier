# RSM vs Confusion Matrix Analysis: Key Improvements

## Summary

You were right to suspect the confusion matrices. I've created a **fixed implementation** that addresses three major issues from the previous analysis in the report.

## The Three Key Fixes

### 1. **Explicit Symmetric Confusion Matrix** ✓

**Previous approach** (from report):
- Used "average of bidirectional confusion rates"
- Mathematical formula unclear in code
- Averaging strategy not explicitly validated

**New approach**:
```python
# 1. Build raw confusion matrix (rows=true label, cols=predicted label)
cm_raw = confusion_matrix(labels, preds)

# 2. Row-normalize to get confusion rates
cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
# cm_norm[i,j] = P(predicted_as_j | true_label_is_i)

# 3. Create symmetric matrix
cm_sym = (cm_norm + cm_norm.T) / 2.0
# cm_sym[i,j] = (P(pred=j|true=i) + P(pred=i|true=j)) / 2
```

**Why this matters**:
- `cm_sym[i,j] = cm_sym[j,i]` guaranteed (truly symmetric)
- Captures **bidirectional confusion**: how much verbs confuse with each other
- Example: if "open" → "pull" 20% and "pull" → "open" 15%, then M[open,pull] = 0.175

---

### 2. **Contextualized Verb Embeddings** (NEW)

**Previous approach**:
- Non-contextualized only: just the bare word embedding
- "turn" and "rotate" might have similar embeddings even if used differently in CALVIN

**New approach**: Two parallel analyses
```
Non-contextualized:
  "grasp" → tokenizer → embed → single embedding vector

Contextualized (NEW):
  Find all CALVIN instructions with "grasp":
    - "grasp the block"
    - "please grasp it"
    - "grasp the object"
  Extract verb hidden state in each context
  Mean-pool across all occurrences → richer embedding
```

**Why it matters**:
- Manipulation context is important: "turn" (rotate) vs "turn on" (toggle)
- CALVIN-specific semantics: verbs have specific motion/outcome patterns
- Contextualized embeddings should correlate **better** with motor confusion

---

### 3. **Expanded Model Coverage** (NEW)

**Previous approach**:
- Only GPT-2 (static + contextualized) and Vicuna
- Missing modern LLMs and VLM baseline

**New approach**: 5 models × 2 embedding types = 10 RSMs

| Model | Non-ctx | Contextualized |
|-------|:---:|:---:|
| **LLaMA-2-7B** | ✓ | ✓ | Base LLM, strong baseline |
| **LLaVA-1.5-7B** | ✓ | ✓ | Vision-conditioned (uses text LM) |
| **Qwen-2.5-7B** | ✓ | ✓ | Modern LLM, different architecture |
| **Qwen-VL** | ✓ | ✓ | Vision-conditioned variant |
| **GPT-2** | ✓ | — | Baseline (frozen embeddings) |

**Why it matters**:
- Compare base LLMs (LLaMA, Qwen) vs vision-tuned (LLaVA, Qwen-VL)
- If visual grounding helps → VLMs should have higher correlation
- Modern models (Qwen) might capture semantics better than older (GPT-2)

---

## Experiment Design

### Inputs
- **Confusion matrix**: From `r8_ao_native_preds.json` (683 val episodes, 20 verbs)
- **Verb list**: 20 sparse CALVIN verbs (excluding "left" as per convention)
- **CALVIN instructions**: ~500 training episodes for context extraction

### Computation
1. Extract embeddings for all 20 verbs from each model
2. Compute Representational Similarity Matrices (cosine similarity)
3. Compare each RSM with symmetric confusion matrix
4. Report Spearman ρ and Pearson r correlations

### Outputs
All saved to `results/rsm_confusion_analysis_v2.npz`:
- `confusion_matrix` — the symmetric motor similarity matrix
- `rsm_*` — 9 RSM matrices (one per model × embedding type)
- `emb_*` — 10 embedding matrices (raw 20×d vectors)

Visualizations:
- `figures/rsm_vs_confusion_v2_heatmaps.png` — 10 heatmaps side-by-side
- `figures/rsm_vs_confusion_v2_correlations.png` — correlation bar chart
- `figures/rsm_vs_confusion_v2_detailed.png` — best models highlighted

---

## Expected Results

### If symmetry + contextualization helps:
```
Spearman ρ with motor confusion:

Non-contextualized:
  gpt2:      -0.002 (~0, baseline from report)
  llama:     ?
  qwen:      ?

Contextualized:
  llama_ctx: ρ > 0.10 (improvement due to context)
  qwen_ctx:  ρ > 0.10

Best if VLMs > LLMs:
  llava_ctx: ρ > llama_ctx (visual grounding helps)
  qwen_vl_ctx: ρ > qwen_ctx
```

### If null result (ρ ≈ 0 still):
- Confirms language statistics ≠ motor similarity
- Verbs are grounded in **physics**, not linguistic co-occurrence
- Different conclusion from report's claim about antonyms

---

## How to Run

```bash
# Main analysis (GPU required, ~2 hours)
python analysis/compute_rsm_confusion_correlations.py

# Visualizations (CPU, ~1 min)
python figures/plot_rsm_confusion_v2.py
```

### Output files created:
```
results/
  rsm_confusion_analysis_v2.npz    ← main results
  rsm_confusion_correlations_v2.json ← human-readable table

figures/
  rsm_vs_confusion_v2_heatmaps.png
  rsm_vs_confusion_v2_correlations.png
  rsm_vs_confusion_v2_detailed.png
```

---

## Key Code Differences

### Symmetric Matrix Construction
**Before** (report): Unclear, possibly incorrect
**After**:
```python
# Explicitly symmetric: M[i,j] = M[j,i]
cm_sym = (cm_normalized + cm_normalized.T) / 2.0
assert np.allclose(cm_sym, cm_sym.T)  # Validate
```

### Contextualized Extraction
**Before**: Only 1 contextualized model (GPT-2 via full instructions)
**After**: 4 modern LLMs × 2 embedding types
```python
def extract_contextualized_embeddings(model, tokenizer, verbs, verb_to_instructions):
    # For each verb, find all CALVIN instructions
    # Extract hidden state at verb location
    # Mean-pool across all occurrences
```

### Model Management
**Before**: 2 models hardcoded
**After**: Loop over 5 models, free GPU memory between each
```python
for model_key in ["gpt2", "llama", "llava", "qwen", "qwen_vl"]:
    # Load and process
    torch.cuda.empty_cache()  # Free memory
```

---

## Next Steps (After Running)

1. **Examine heatmaps**: Do verb clusters match intuition?
   - Should see high diagonal (correct classifications)
   - High off-diagonal for synonym pairs (e.g., "open"↔"pull")

2. **Interpret correlations**:
   - Which model has highest correlation?
   - Does contextualization improve it?
   - Do VLMs beat base LLMs?

3. **Sanity checks**:
   - If best ρ > 0.3: language partially predicts motor similarity
   - If best ρ < 0: invert-check for systematic structure

4. **Write findings**: Update `round2_rsm_confusion_fixed.md` with results

---

## Reference Materials

- **Lab notebook**: `lab_notebooks/action_v_llm_rsm/round2_rsm_confusion_fixed.md`
- **Report section**: `figures/report/report.tex` lines ~372-441 (RSM vs confusion section)
- **Previous analysis**: `lab_notebooks/action_v_llm_rsm/round1_verb_rsm_loglikelihood.md`

---

## Questions This Addresses

**Q: "Should confusion matrices be symmetric by the diagonal?"**
A: ✓ Yes! M[i,j] = average bidirectional confusion rate = M[j,i]

**Q: "Did the previous implementation do this correctly?"**
A: Unknown — code that generates `symmetric_confusion.pdf` is missing.
This implementation is explicit and validated.

**Q: "Will contextualization help?"**
A: Should help. CALVIN context is specific (e.g., "close the drawer" vs "close the door").
   Contextualized embeddings capture manipulation semantics better than bare words.

**Q: "Which models should I expect to work best?"**
A: Modern instruction-tuned models (LLaMA-2, Qwen) likely > GPT-2.
   VLMs might surprise (visual grounding of verbs like "push" vs "pull").
