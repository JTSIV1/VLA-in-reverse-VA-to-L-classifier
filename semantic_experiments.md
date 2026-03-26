python tokenization/train_tokenizer.py \
  --tokenizer vq_bet \
  --data_dir ../task_D_D/training \
  --aux_head verb \
  --loss_function semantic \
  --semantic_temp 0.1 \
  --aux_lambda 0.1 \
  --batch_size 256 \
  --num_workers 6 \
  --tag vqbet_semantic01_c5e16g4

torchrun --standalone --nnodes=1 --nproc-per-node=1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-bridge" \
  --vla.base_vlm "/home/ec2-user/11777/models/prism-qwen25" \
  --expected_world_size 1 \
  --data_root_dir "/home/ec2-user/11777/calvin_rlds" \
  --run_root_dir "/home/ec2-user/11777/runs/calvind_scratch" \
  --tokenizer_checkpoint "/home/ec2-user/11777/VLA-in-reverse-VA-to-L-classifier/checkpoints/vq_bet_verb0.1_vqbet_semantic01_c5e16g4/tokenizer_weights.pth"

python tokenization/train_tokenizer.py \
  --tokenizer quest \
  --data_dir ../task_D_D/training \
  --aux_head verb \
  --loss_function semantic \
  --semantic_temp 0.1 \
  --aux_lambda 0.1 \
  --batch_size 256 \
  --num_workers 6 \
  --tag quest_semantic01_h16d2



# Semantic Tokenizers (Verb $\lambda=0.1$): VQ-BeT & QueST Experiment Documentation

## Overview

This experiment evaluates the training of two standalone action tokenizers—Vector-Quantized Behavior Transformer (VQ-BeT) and QueST—equipped with an auxiliary semantic verb classification head. The goal is to successfully compress continuous robot actions into discrete tokens while loosely structuring the latent space around semantic action groupings, without degrading the primary reconstruction objective.

**Run Names**: 
* `vq_bet_verb0.1_vqbet_semantic01_c5e16g4`
* `quest_verb0.1_quest_semantic01_h16d2`

**Dataset**: task_D_D (3356 train / 666 val episodes)
**Verb classes**: 20 classes 
**Date**: March 2026

---

## 1. Tokenizer Architecture Summaries

### VQ-BeT (Vector-Quantized Behavior Transformer)
* **Architecture**: MLP encoder → ResidualVQ → MLP decoder
* **Hyperparameters (`c5e16g4`)**: chunk_size=5, num_codes=16, vq_groups=4
* **Tokens per chunk**: 4 (one per VQ group), codebook combinations = 65K ($16^4$)
* **Reference**: Behavior Generation with Latent Actions (Lee et al., 2024)

### QueST 
* **Architecture**: Transformer-based state-to-action quantizer
* **Hyperparameters (`h16d2`)**: Codebook size/heads=16, depth=2
* **Tokens per chunk**: Mapped via multi-head quantization. 

---

## 2. Auxiliary Head

### Verb Classification Head
* **Mechanism**: Linear probe on mean-pooled encoder latents → 20-class verb prediction.
* **Weighting**: Auxiliary weight `aux_lambda` ($\lambda$) = 0.1.
* **Objective**: Guide the codebook to map physically distinct but semantically similar actions (e.g., different ways of "grasping") closer together in the latent space.

---

## 3. File Locations

### Scripts
| Script | Path | Description |
|--------|------|-------------|
| Tokenizer training | `tokenization/train_tokenizer.py` | Main training loop for the tokenizer + aux head |
| VQ-BeT model | `tokenization/vqbet_tokenizer.py` | VQBeTTokenizer class definition |
| QueST model | `tokenization/quest_tokenizer.py` | QueST Tokenizer class definition |

### Data
| Resource | Path |
|----------|------|
| CALVIN D training | `../task_D_D/training/` (Cache: `_action_cache.npz`, 142.8K steps) |
| CALVIN D validation | `../task_D_D/validation/` (Cache: `_action_cache.npz`, 26.7K steps) |

---

## 4. Training Results

**Training Duration**: ~100 Epochs
**Learning Rate**: Cosine decay (1e-4 → 0.0)

### VQ-BeT Performance (`c5e16g4`)
| Metric | Epoch 1 | Epoch 100 | Best Overall |
| :--- | :--- | :--- | :--- |
| **Train Recon Loss** | 0.1814 | 0.0048 | - |
| **Val Recon Loss** | 0.1741 | 0.0052 | **~0.0052** |
| **Train Verb Acc** | 2.00% | 28.07% | - |
| **Val Verb Acc** | 3.15% | 27.78% | **28.38%** |

### QueST Performance (`h16d2`)
| Metric | Epoch 1 | Epoch 98 | Best Overall |
| :--- | :--- | :--- | :--- |
| **Train Recon Loss** | 1.8853 | 0.0279 | - |
| **Val Recon Loss** | 0.3123 | 0.0185 | **0.0183** *(Ep 94)* |
| **Train Verb Acc** | 3.13% | 48.15% | - |
| **Val Verb Acc** | 2.55% | 38.59% | **42.34%** *(Ep 72)* |

---

## 5. Analysis & Takeaways

### 1. The Reconstruction vs. Semantics Trade-off
Adding the auxiliary verb head ($\lambda=0.1$) exposed a clear architectural divergence between VQ-BeT and QueST. 
* **VQ-BeT** prioritized physical fidelity, driving validation reconstruction loss down to an exceptional **0.0052**, but struggled to push semantic verb accuracy past **28.38%**.
* **QueST** adapted much better to the semantic grouping, reaching a peak validation verb accuracy of **42.34%** (more than double VQ-BeT's semantic performance). However, this came at the cost of physical fidelity, as its validation reconstruction loss bottomed out around **0.0183** (roughly 3.5x higher than VQ-BeT).

### 2. Semantic Latent Space Structuring
Given 20 verb classes, a random guess baseline is 5%. Both models successfully structured their latent spaces beyond random chance. QueST's ability to consistently hold ~40% accuracy in later epochs proves its multi-head quantization strategy is highly receptive to auxiliary semantic gradients, making it the superior choice if downstream language grounding is the strict priority.

### 3. Overfitting Dynamics
VQ-BeT's validation metrics tightly tracked its training metrics, showing no signs of overfitting. QueST, however, began to show a slight generalization gap in the later epochs—Train Verb Acc continued climbing to 48.15% by Epoch 98, while Val Verb Acc peaked at Epoch 72 (42.34%) and slowly degraded to ~38%, indicating that early stopping (around Epoch 70-75) is optimal for the QueST tokenizer.

---

## 6. Next Steps
Both tokenizers have converged and offer distinct advantages. The weights from the best epochs (Epoch 100 for VQ-BeT, Epoch 72 for QueST) should now be frozen to act as the discrete action spaces for downstream policy training (OpenVLA-mini / Transformer VLA fine-tuning) to see whether superior physical reconstruction (VQ-BeT) or superior semantic grouping (QueST) yields a better final robotic policy.