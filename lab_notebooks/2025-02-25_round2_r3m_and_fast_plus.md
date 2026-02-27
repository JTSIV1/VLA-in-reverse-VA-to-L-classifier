# Round 2: Pretrained Encoders (R3M Vision + FAST+ Action)

**Date**: 2025-02-25
**Motivation**: Round 1 showed (a) vision-only collapsed completely with from-scratch Conv2d, (b) custom-fitted FAST tokenizers (3,400 trajectories) hurt badly. This round tests whether pretrained encoders fix both issues.

---

## Changes from Round 1

### Vision: R3M (Radosavovic et al., 2022)
- **Encoder**: R3M ResNet-50, pretrained on Ego4D human video data using time-contrastive + language-aligned objectives
- **Output**: 2048-d global feature per image, projected to d_model=128 via learned linear layer
- **Key difference**: Each image becomes 1 token (vs 64 patches in round 1), so vision-only has just 3 tokens total: [CLS] + start_frame + end_frame
- **Input**: 224x224 (upscaled from 200x200 CALVIN native)
- **Variants**: frozen (only train projection + transformer) and fine-tuned (train everything)

### Action: Pretrained FAST+ (Physical Intelligence, 2025)
- **Tokenizer**: Universal BPE trained on 1M real robot action sequences (vs our 3,400)
- **Source**: `physical-intelligence/fast` on HuggingFace
- **Params**: scale=10, vocab_size=2048, min_token=-354
- **Token length**: ~130 tokens per trajectory (vs ~64 native timesteps), so max_seq_len raised to 192

### Additional features added this round
- **Multi-frame support** (`--num_frames`): Sample N uniformly-spaced frames instead of just first+last. With R3M, each frame adds only 1 token.
- **Weighted cross-entropy** (`--weighted_loss`): Inverse-frequency class weighting to combat imbalanced verb distribution.
- **Sparse class filtering** (`--min_class_count N`): Drop verb classes with fewer than N training samples. With threshold 30: 27→21 classes, keeping 97% of data (3309/3404 train, 665/683 val). Dropped: collapse, go, move up, slide down, slide up, unstack.
- **Best-val checkpoint saving**: Saves `_best.pth` whenever val accuracy improves. Avoids overfitting losses (FAST+ lost 3.8pp, action_only lost 3.2pp in round 1).
- **Bug fix**: R3M finetune initially had a bug where `torch.no_grad()` was always used in the forward pass, preventing gradient flow. Fixed for job 6452738+.

---

## Experiments

| Job ID  | Experiment            | Modality    | Vision Encoder   | Action Rep       | Notes |
|---------|-----------------------|-------------|------------------|------------------|-------|
| 6452034 | vision_r3m_frozen     | vision_only | R3M (frozen)     | —                | 2 frames |
| 6452738 | vision_r3m_finetune   | vision_only | R3M (finetune)   | —                | 2 frames, bug-fixed |
| 6452036 | ao_fast_pretrained    | action_only | —                | FAST+ pretrained | max_seq_len=192 |
| 6452975 | vision_r3m_8f         | vision_only | R3M (frozen)     | —                | 8 frames |
| 6453120 | vision_r3m_ft_weighted| vision_only | R3M (finetune)   | —                | 2 frames + weighted CE |
| 6457792 | vision_r3m_ft_8f      | vision_only | R3M (finetune)   | —                | 8 frames |
| 6457793 | vision_r3m_ft_sparse  | vision_only | R3M (finetune)   | —                | 2 frames, min_class_count=30 |
| 6457794 | ao_native_sparse      | action_only | —                | native           | min_class_count=30 |
| 6457795 | vision_r3m_ft_8f_sparse| vision_only | R3M (finetune)  | —                | 8 frames, min_class_count=30 |
| 6457851 | ao_native_weighted    | action_only | —                | native           | 27cls, weighted CE |
| 6457852 | ao_native_sparse_wt   | action_only | —                | native           | 21cls, weighted CE |
| 6457855 | vision_r3m_ft_sparse_wt| vision_only | R3M (finetune)  | —                | 2 frames, 21cls, weighted CE |

Training config: 30 epochs, batch_size=16, lr=5e-4 (OneCycleLR), d_model=128, 4 layers, 8 heads. All new jobs include best-val checkpoint saving.

### Round 1 baselines for comparison

| Model               | Accuracy | Macro F1 | Weighted F1 |
|----------------------|----------|----------|-------------|
| vision_only (patch)  | 12.88%   | 0.85%    | 2.94%       |
| action_only (native) | 39.39%   | 25.68%   | 36.07%      |
| best custom FAST     | 23.28%   | —        | —           |

---

## Results: Vision-Only Experiments

All vision models use R3M ResNet-50 encoder with d_model=128 Transformer classifier.

| # | Experiment              | #Cls | Frames | Wt CE | Acc    | MacF1  | WtdF1  | Active | BestAcc | BestMF1 | BestWF1 |
|---|-------------------------|------|--------|-------|--------|--------|--------|--------|---------|---------|---------|
| 1 | Patch from-scratch (R1) | 27   | 2      | no    | 12.9%  | 0.8%   | 2.9%   | 1/27   | —       | —       | —       |
| 2 | R3M frozen 2f           | 27   | 2      | no    | 12.9%  | 0.8%   | 2.9%   | 1/27   | —       | —       | —       |
| 3 | R3M frozen 8f           | 27   | 8      | no    | 12.9%  | 0.8%   | 2.9%   | 1/27   | —       | —       | —       |
| 4 | R3M finetune 2f         | 27   | 2      | no    | 22.4%  | 2.8%   | 9.2%   | 3/27   | —       | —       | —       |
| 5 | R3M ft weighted CE      | 27   | 2      | yes   | 8.6%   | 0.6%   | 1.4%   | 1/27   | —       | —       | —       |
| 6 | R3M ft 8f               | 27   | 8      | no    | 20.9%  | 3.7%   | 12.1%  | 4/27   | 23.1%   | 4.3%    | 13.8%   |
| 7 | R3M ft 2f sparse        | 21   | 2      | no    | 22.0%  | 3.4%   | 10.3%  | 3/21   | 23.3%   | 4.6%    | 13.0%   |
| 8 | R3M ft 8f sparse        | 21   | 8      | no    | 15.8%  | 2.3%   | 7.0%   | 2/21   | 19.2%   | 2.9%    | 8.7%    |
| 9 | R3M ft sparse+wt CE     | 21   | 2      | yes   | 13.2%  | 1.1%   | 3.1%   | 1/21   | 15.6%   | 1.3%    | 4.2%    |

**Column definitions:**
- **#Cls**: Number of verb classes (27 = all, 21 = after dropping classes with <30 training samples)
- **Active**: Number of classes the model actually predicts with recall > 0 out of total classes
- **BestAcc/BestMF1/BestWF1**: Metrics on best-validation-accuracy checkpoint (only available for later experiments that save `_best.pth`)

### Per-class collapse analysis

The best vision model (R3M ft 2f, 22.4% acc) only predicts 3 classes: `push` (93.2% recall), `pick up` (90.9%), and `grasp` (1.0%). All other 24 classes have 0% recall. The model learns to predict the 2-3 most frequent classes and ignores everything else.

Weighted CE doesn't fix this — it just shifts which single class the model collapses to. Comparing R3M ft 2f (unweighted) vs R3M ft weighted CE:
- Unweighted: predicts `push` (93.2%) + `pick up` (90.9%) → 22.4% acc
- Weighted: predicts only `slide` (100%) → 8.6% acc
The heavy inverse-frequency weights destabilize the few classes the model can learn, forcing collapse to a different single class.

### Training dynamics

- **Frozen R3M** (#2, #3): Training loss flat (~2.82) for 30 epochs. Collapsed identically with 2 or 8 frames. R3M's global features encode scene semantics but not verb-discriminative differences.
- **R3M finetune** (#4, #6, #7): Starts learning around epoch 22 when OneCycleLR enters cosine decay. Train/val track closely (~23.5%/22.4%), so underfitting rather than overfitting — the bottleneck is model capacity, not data.
- **8 frames + finetune** (#6): Slightly improves macro F1 (3.7% vs 2.8%) and activates 4 vs 3 classes on 27-class. But with sparse filtering (#8 vs #7), 8 frames actually hurts — possibly overfitting with more parameters but fewer classes.
- **Weighted CE** (#5, #9): Consistently makes things worse for vision. The model doesn't have enough capacity to learn rare classes, so upweighting them just destabilizes training.

### Key findings

1. **Fine-tuning R3M is essential.** Frozen R3M = collapsed regardless of frame count. Fine-tuning reaches 22-23% — a 10pp improvement, but still only 3-4 active classes.
2. **More frames has limited impact.** 8 frames marginally helps macro F1 with finetuning (+0.9pp), but doesn't unlock new classes.
3. **Sparse filtering doesn't help vision.** Removing rare classes reduces #Cls from 27→21 but doesn't improve metrics — the model is only predicting 3 classes anyway, and those classes remain after filtering.
4. **Weighted CE consistently hurts vision.** Without capacity to learn rare classes, heavy weights just destabilize the few classes the model manages to learn.
5. **Best vision result: ~23% accuracy, ~4.6% macro F1** — far below action-only performance (~39-43% accuracy, ~25-33% macro F1).

---

## Results: Action-Only Experiments

### Native action representation

| Experiment              | #Cls | Acc    | MacF1  | Active  | BestAcc | BestMF1 |
|-------------------------|------|--------|--------|---------|---------|---------|
| AO native (R1)          | 27   | 39.4%  | 25.7%  | 20/27   | —       | —       |
| AO native sparse        | 21   | 40.0%  | 32.0%  | —       | 43.0%   | 32.5%   |
| AO native weighted      | 27   | 33.7%  | 29.7%  | 24/27   | 35.1%   | 23.7%   |
| **AO native sparse+wt** | **21** | **37.9%** | **37.4%** | **21/21** | **39.5%** | **38.7%** |

**Key finding**: Sparse filtering + weighted CE activates **all 21 classes** (21/21) with **38.7% macro F1** — the best action-only result. Weighted CE alone (27cls) activates 24/27 classes but accuracy drops. The combination of removing rare classes and upweighting remaining ones is optimal.

### FAST tokenization: scale × vocab sweep

Reconstruction MSE is nearly identical across all configurations (~0.331-0.333) because DCT quantization is nearly lossless for 7D actions. The key variable is **compression rate** (tokens per trajectory):

| Scale | Vocab | MSE    | Tok/traj | Compress | Acc    | MacF1  | Active | BestAcc | BestMF1 |
|-------|-------|--------|----------|----------|--------|--------|--------|---------|---------|
| 1     | 256   | 0.3331 | 26.7     | 16.0x    | 28.7%  | 13.7%  | 14/27  | **30.9%** | 9.9%  |
| 5     | 256   | 0.3327 | 94.8     | 4.5x     | 24.3%  | 8.0%   | 8/27   | 25.5%   | 8.6%    |
| 10    | 256   | —      | —        | —        | 23.3%  | —      | —      | —       | —       |
| 10    | 512   | 0.3318 | 137.7    | 3.1x     | 23.1%  | 7.5%   | 9/27   | 24.9%   | 7.9%    |
| 10    | 1024  | —      | —        | —        | 23.3%  | —      | —      | —       | —       |
| 20    | 768   | 0.3315 | 240.1    | 1.8x     | 21.1%  | 5.6%   | 6/27   | 22.5%   | 5.9%    |
| 50    | 1536  | 0.3312 | 254.9    | 1.7x     | 26.8%  | 7.7%   | 8/27   | 28.0%   | 7.0%    |
| FAST+ | 2048  | 0.3318 | 127.7    | 3.3x     | 21.2%  | 8.5%   | 11/27  | 23.4%   | 5.2%    |

**Rows without MSE/tokens**: Round 1 results (scale=10, v256/v1024) that didn't record these.

**Key finding**: **Scale=1 (coarsest quantization) wins clearly** — 30.9% best accuracy, 14 active classes. Lower scale → shorter token sequences → easier to learn from 3,400 samples. But all FAST configs are still far below native actions (39.4%).

A **full 5×3 grid sweep** (scales 1/5/10/20/50 × vocabs 256/512/1024) is now being fitted (job 6458014). 10 new combinations will fill in the missing cells.

![FAST scale sweep](../figures/fast_scale_sweep.png)

---

## Open Questions

### Why is vision-only so bad?

The start/end frames should in principle contain rich information about what happened — object displacements, gripper state changes, and scene rearrangements. Several factors likely contribute:

1. **R3M global pooling destroys spatial information.** R3M outputs a single 2048-d vector per frame, collapsing all spatial layout. To distinguish "push left" from "slide left," you need to detect precise object displacement between frames (e.g., block at (x1,y1) → (x2,y2)). A global feature vector loses this spatial correspondence.

2. **Many CALVIN verbs differ in *how* the motion happens, not just the outcome.** "Push" vs "slide" can produce identical object displacements — the difference is contact dynamics (gripper pressing vs object gliding on surface). Start/end frames only show endpoints, not the trajectory between them.

3. **2 frames = endpoints only.** The action sequence provides 64 timesteps × 7D = 448 continuous values describing the full trajectory. Even 8 frames provide sparse snapshots. Actions are temporally dense; vision is temporally sparse.

4. **The 1-token-per-frame bottleneck.** With R3M, each frame is compressed to a single token. The transformer has only 3 tokens (CLS + 2 frames) for 2-frame models, or 9 tokens for 8-frame. Compare to action-only which has 64+ tokens. The vision transformer has far less information to attend over.

### Potential next steps for vision

1. **Use R3M patch-level features instead of global pooling.** R3M's ResNet produces a 7×7 spatial feature map before global average pooling. Using these 49 patch tokens per frame (instead of 1 global token) would preserve spatial layout and let the transformer attend to specific regions across frames. This would give 2×49+1 = 99 tokens for 2-frame, comparable to action-only's 64 tokens.

2. **Try a different visual backbone.** R3M was trained for state representation (Ego4D), not for detecting fine-grained spatial changes between frames. Alternatives:
   - **DINOv2**: Self-supervised ViT with strong spatial features and built-in patch tokens
   - **VideoMAE / TimeSformer**: Video-native models that jointly encode temporal+spatial structure
   - **Optical flow features**: Explicitly encode pixel-level displacement between frames

3. **Dense frame sampling.** Instead of 2 or 8 frames, sample all available frames (the CALVIN trajectories are short). With R3M global features at 1 token/frame, even 32 frames only adds 32 tokens.

4. **Two-stream approach.** Separate appearance (single frame) from motion (frame differences or optical flow). The verb classification likely depends more on motion than static appearance.

### Overall research direction

This is not a competition between vision-only and action-only. The goal is to:
1. Get the best possible results for each modality independently
2. Then build a multimodal model that combines both and compare to the unimodal baselines

The large gap between vision (~23%) and action (~43%) suggests that either (a) the visual representation needs significant improvement, or (b) the information truly isn't there in just 2 frames for many verbs. Resolving this question is important before proceeding to multimodal fusion — if vision adds no signal beyond what actions provide, the multimodal model may not improve over action-only.

---

## Experiment Index

| Job ID  | Experiment             | Status    | Log | Metrics |
|---------|------------------------|-----------|-----|---------|
| 6452034 | vision_r3m_frozen      | completed | results/vision_r3m_frozen_j6452034_log.json | results/vision_r3m_frozen_j6452034_metrics.json |
| 6452035 | vision_r3m_finetune (bugged) | completed | results/vision_r3m_finetune_j6452035_log.json | results/vision_r3m_finetune_j6452035_metrics.json |
| 6452738 | vision_r3m_finetune (fixed)  | test rerun | results/vision_r3m_finetune_j6452738_log.json | results/vision_r3m_finetune_j6452035_metrics.json |
| 6452036 | ao_fast_pretrained     | completed | results/ao_fast_pretrained_j6452036_log.json | results/ao_fast_pretrained_j6452036_metrics.json |
| 6452975 | vision_r3m_8f          | completed | results/vision_r3m_8f_j6452975_log.json | results/vision_r3m_8f_j6452975_metrics.json |
| 6453120 | vision_r3m_ft_weighted | completed | results/vision_r3m_ft_weighted_j6453120_log.json | results/vision_r3m_ft_weighted_j6453120_metrics.json |
| 6457792 | vision_r3m_ft_8f       | completed | results/vision_r3m_ft_8f_j6457792_log.json | results/vision_r3m_ft_8f_j6457792_metrics.json |
| 6457793 | vision_r3m_ft_sparse   | completed | results/vision_r3m_ft_sparse_j6457793_log.json | results/vision_r3m_ft_sparse_j6457793_metrics.json |
| 6457794 | ao_native_sparse       | completed | results/ao_native_sparse_j6457794_log.json | results/ao_native_sparse_j6457794_metrics.json |
| 6457795 | vision_r3m_ft_8f_sparse| completed | results/vision_r3m_ft_8f_sparse_j6457795_log.json | results/vision_r3m_ft_8f_sparse_j6457795_metrics.json |
| 6457851 | ao_native_weighted     | completed | results/ao_native_weighted_j6457851_log.json | results/ao_native_weighted_j6457851_metrics.json |
| 6457852 | ao_native_sparse_wt    | completed | results/ao_native_sparse_weighted_j6457852_log.json | results/ao_native_sparse_weighted_j6457852_metrics.json |
| 6457855 | vision_r3m_ft_sparse_wt| completed | results/vision_r3m_ft_sparse_wt_j6457855_log.json | results/vision_r3m_ft_sparse_wt_j6457855_metrics.json |
| 6457871 | ao_fast_s1_v256        | completed | results/ao_fast_s1_v256_j6457871_log.json | results/ao_fast_s1_v256_j6457871_metrics.json |
| 6457872 | ao_fast_s5_v256        | completed | results/ao_fast_s5_v256_j6457872_log.json | results/ao_fast_s5_v256_j6457872_metrics.json |
| 6457873 | ao_fast_s10_v512       | completed | results/ao_fast_s10_v512_j6457873_log.json | results/ao_fast_s10_v512_j6457873_metrics.json |
| 6457874 | ao_fast_s20_v768       | completed | results/ao_fast_s20_v768_j6457874_log.json | results/ao_fast_s20_v768_j6457874_metrics.json |
| 6457875 | ao_fast_s50_v1536      | completed | results/ao_fast_s50_v1536_j6457875_log.json | results/ao_fast_s50_v1536_j6457875_metrics.json |
| 6457876 | ao_fast_pretrained_v2  | completed | results/ao_fast_pretrained_v2_j6457876_log.json | results/ao_fast_pretrained_v2_j6457876_metrics.json |
| 6458014 | fit_fast_full_sweep    | running   | logs/fit_fast_full_sweep-6458014.out | — |

Note: Job 6452738 trained correctly but test_transformer.py failed due to architecture change (type_img_start/end → frame_pos/type_img). Test was rerun locally with backward-compat key remapping. Metrics saved under j6452035 filename for consistency.
