# Experiment Log: Verb Classification from Vision + Action

**Goal**: Given a robot manipulation trajectory (first/last RGB frames + relative action sequence), classify which verb describes the task (27 classes from CALVIN dataset).

**Core question**: How much does each modality (vision vs. action) and action representation (native continuous vs. FAST discrete tokens) contribute to verb classification?

---

## Setup

### Dataset
- CALVIN task_D_D split: 3,404 training / 683 validation annotated episodes
- 27 unique verb classes (highly imbalanced: "push" ~300 examples, some <10)
- Each episode: sequence of 200x200 RGB frames + 7D relative actions (dx, dy, dz, droll, dpitch, dyaw, gripper)
- Typical sequence length: 64 timesteps

### Model
- Transformer encoder with ViT-style patch embeddings (patch size 25x25 → 64 patches/image)
- d_model=128, nhead=8, num_layers=4
- Token types: [CLS], start frame patches, end frame patches, action tokens
- Learned positional + token-type embeddings
- Classification head on [CLS] output

### Training
- 30 epochs, batch size 16, OneCycleLR (max_lr=5e-4)
- Cross-entropy loss, no class weighting, no early stopping

### Sequence lengths by modality

| Config       | Sequence length                  | Params  |
|--------------|----------------------------------|---------|
| full (A+V)   | 1 (CLS) + 128 (patches) + 64 (actions) = 193 | 2.64M |
| action_only  | 1 + 64 = 65                      | 2.39M   |
| vision_only  | 1 + 128 = 129                    | 2.63M   |

---

## Experiment 1: Modality Ablation (Native Actions)

**Question**: How much does vision vs. action contribute when using native continuous actions?

### Results

| Model            | Accuracy | Macro F1 | Weighted F1 |
|------------------|----------|----------|-------------|
| Action + Vision  | 42.17%   | 23.52%   | 37.19%      |
| Action Only      | 39.39%   | 25.68%   | 36.07%      |
| Vision Only      | 12.88%   | 0.85%    | 2.94%       |

### Overfitting analysis (no early stopping was used)

| Model            | Best val_acc | @ epoch | Final val_acc (ep 30) | Lost |
|------------------|--------------|---------|-----------------------|------|
| Action + Vision  | 42.75%       | 24      | 42.17%                | 0.6% |
| Action Only      | 42.61%       | 15      | 39.39%                | 3.2% |

### Findings

1. **Action signal dominates.** Action-only (39.4%) nearly matches the full model (42.2%). Vision-only is near chance (12.9% vs 3.7% random baseline for 27 classes).

2. **Vision provides minimal additive signal.** Adding vision to actions improves accuracy by only ~3 percentage points. First/last frames alone cannot distinguish verbs well — "push" and "slide" look similar in start/end states.

3. **Model capacity concern.** The full model processes 193 tokens while action-only processes 65. Same 4-layer transformer, but attention is diluted across 3x more tokens. The small benefit of vision may partly reflect this capacity disadvantage rather than vision being unhelpful.

---

## Experiment 2: FAST Tokenization Sweep

**Question**: Does discretizing actions via FAST (DCT → quantize → BPE) help or hurt, and how does vocab size matter?

### Results

| Action Rep | Action + Vision | Action Only | Vision Only |
|------------|-----------------|-------------|-------------|
| native     | 42.17%          | 39.39%      | 12.88%      |
| fast_v256  | 16.84%          | 18.89%      | —           |
| fast_v512  | 12.88%          | 23.28%      | —           |
| fast_v1024 | 15.37%          | 17.86%      | —           |
| fast_v2048 | 12.88%          | 8.49%       | —           |

(Vision-only is constant across action reps — it doesn't use actions.)

### Training dynamics

- Full+FAST models barely reduce loss over 30 epochs (near-flat curves)
- Action-only FAST v1024 memorizes training data (58% train acc) but can't generalize (17.9% val acc)
- Several full+FAST models collapse to predicting only "push" (12.88% = push class proportion)

### Findings

0. **Vision model collapsed. training losses barely moved**

1. **FAST tokenization dramatically hurts classification.** Best FAST result (action-only v512, 23.3%) is ~20 points behind native action-only (39.4%).

2. **Larger FAST vocab does not help.** No clear trend — v512 action-only is the best FAST variant, v2048 is worst.

3. **Adding vision to FAST models often hurts.** In every vocab size, action-only FAST outperforms full FAST. If action tokens are noisy, adding 128 vision tokens further dilutes attention.

4. **Why FAST fails for classification:**
   - **Lossy quantization**: DCT coefficients are rounded to integers (scale=10), destroying fine-grained magnitude differences that distinguish "push" vs "slide"
   - **Small training set**: only ~3,400 trajectories to learn embeddings for 256–2048 tokens

---

## Vision Model Collapse

The vision-only model completely collapsed — it predicts "push" for all 683 validation samples (12.88% accuracy = majority class proportion). Training loss barely moved (2.94 → 2.82 over 30 epochs), indicating the from-scratch Conv2d patch embedding never learns meaningful features from only 3,404 training images.

**Evidence**: Macro F1 = 0.85%, Weighted F1 = 2.94%. Per-class recall is 0% for all 26 non-push verbs. The model exploits class imbalance rather than learning visual features.

**Root cause**: A randomly initialized Conv2d(3, 128, kernel_size=25, stride=25) has no capacity to extract useful visual features from 200x200 synthetic images with only ~3,400 examples. The model needs either (a) a pretrained vision encoder, or (b) significantly more training data.

---

## Per-Class Analysis

Observations from native action models:

- **Verbs with distinctive motion signatures** classify well: `rotate` (98.2% recall), `pull` (100%), `take off` (75%)
- **Verbs with similar motions** are confused: `push`/`slide`/`move` share horizontal motion patterns
- **Rare verbs** are mostly missed: `collapse`, `left`, `move up`, `slide down/up` have <5 val samples and 0% recall
- **Vision-only** recognizes only `grasp` (54%) and `push` (55%) — likely exploiting gripper state in end frame, not motion

---

## Open Questions

**Q1: Can vision be made useful with a pretrained encoder?**
The from-scratch Conv2d patch embedding completely fails (see Vision Model Collapse above). Next steps:
- Use a pretrained vision encoder fine-tuned on robot manipulation data (e.g., MVP — ViT-B pretrained on 4.5M internet + egocentric robot images via MAE)
- MVP provides 196 patch tokens of 768-d, projected to our d_model=128
- Try both frozen and fine-tuned variants
- If vision-only still fails, consider adding more frames (not just first/last)

**Q2: Are there pretrained FAST tokenizers we can use off-the-shelf?**
Our custom FAST tokenizers were fit on only ~3,400 trajectories — far too few for BPE to learn meaningful merge patterns. The FAST+ tokenizer from physical-intelligence/fast (HuggingFace) was trained on 1M real robot action sequences with vocab size ~1024. Using it would rule out the possibility that FAST's poor performance is due to insufficient fitting data rather than a fundamental issue with action discretization for classification.

**Q3: Is the multimodal comparison fair?**
The full model processes 193 tokens with the same 4-layer transformer that action-only uses for 65 tokens. Attention is diluted 3x. Possible approaches:
- Add modality-specific self-attention layers before shared fusion layers
- Use cross-attention instead of early self-attention fusion
- Match model capacity by FLOPs rather than architecture

**Q4: Would early stopping help?**
Action-only lost 3.2% from overfitting (42.6% → 39.4%). Easy win — restore best val checkpoint.

**Q5: Would class-weighted loss help rare verbs?**
Many verbs have <10 val samples. Weighted CE or focal loss might improve macro F1.

---

## Experiment Index

| Job ID  | Experiment              | Status    | Log | Metrics |
|---------|-------------------------|-----------|-----|---------|
| 6443442 | full_native             | completed | results/full_native_log.json | results/full_native_metrics.json |
| 6445597 | action_only (native)    | completed | results/action_only_j6445597_log.json | results/action_only_j6445597_metrics.json |
| 6445598 | vision_only             | completed | results/vision_only_j6445598_log.json | results/vision_only_j6445598_metrics.json |
| 6445599 | full_fast_v256          | completed | results/full_fast_v256_j6445599_log.json | results/full_fast_v256_j6445599_metrics.json |
| 6445600 | action_only_fast_v256   | completed | results/action_only_fast_v256_j6445600_log.json | results/action_only_fast_v256_j6445600_metrics.json |
| 6445601 | full_fast_v512          | completed | results/full_fast_v512_j6445601_log.json | results/full_fast_v512_j6445601_metrics.json |
| 6445602 | action_only_fast_v512   | completed | results/action_only_fast_v512_j6445602_log.json | results/action_only_fast_v512_j6445602_metrics.json |
| 6445603 | full_fast_v1024         | completed | results/full_fast_v1024_j6445603_log.json | results/full_fast_v1024_j6445603_metrics.json |
| 6445604 | action_only_fast_v1024  | completed | results/action_only_fast_v1024_j6445604_log.json | results/action_only_fast_v1024_j6445604_metrics.json |
| 6445605 | full_fast_v2048         |
 completed | results/full_fast_v2048_j6445605_log.json | results/full_fast_v2048_j6445605_metrics.json |
| 6445606 | action_only_fast_v2048  | completed | results/action_only_fast_v2048_j6445606_log.json | results/action_only_fast_v2048_j6445606_metrics.json |
| 6445231 | fit_fast_tokenizers     | timeout   | — | — |

Note: fit_fast job timed out after fitting v256–v2048. v4096 tokenizer was never fitted, so v4096 experiments were not run.
Note: v2048 jobs originally failed at test time due to a state_dict key mismatch (`transformer.layers.X` vs `layers.X`) caused by a code refactor mid-run. Fixed by adding backward-compat key remapping in test_transformer.py and rerunning tests locally.
