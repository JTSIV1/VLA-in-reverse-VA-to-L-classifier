# Round 5: MiniVLA Fine-Tuning on CALVIN D

**Date**: 2026-03-16
**Goal**: Test whether verb-decodable action tokenization improves language grounding
in MiniVLA (Qwen2.5 0.5B), a much smaller VLA than OpenVLA-7B.

---

## Motivation

Rounds 1–3 fine-tuned **OpenVLA-7B** (Llama-2 7B backbone) on CALVIN via LoRA.
Round 4 switched to **RDT2** (Qwen2.5-VL-7B) for temporal token structure.

This round tests the verb-decodable tokenizer on **MiniVLA** (Qwen2.5 0.5B backbone,
~14× smaller than OpenVLA-7B). Key questions:

1. Does verb-decodable tokenization still improve language grounding in a much
   smaller model with less capacity to learn implicit verb structure?
2. How does MiniVLA compare to OpenVLA-7B on CALVIN?
3. Is the 0.5B model sufficient for the CALVIN task complexity?

**Hypothesis**: The verb-decodable tokenizer should provide a *larger* relative
benefit for MiniVLA than for OpenVLA-7B, because the smaller model has less
capacity to learn verb semantics from scratch — the tokenizer provides a stronger
inductive bias.

---

## Architecture: MiniVLA

- **Backbone**: Qwen2.5 0.5B (vs. Llama-2 7B in OpenVLA)
- **Vision**: DINOSigLIP-224px (same as OpenVLA)
- **Training**: FSDP full training from pretrained base VLM (NOT LoRA)
- **Action tokens**: 256 extra tokens added to Qwen vocab (`use_extra=True`)
- **Codebase**: `/data/user_data/wenjiel2/Code/openvla-mini` (Stanford-ILIAD/openvla-mini)
- **Training script**: `vla-scripts/train.py` (FSDP, not finetune.py/LoRA)
- **Base VLM**: `prism-qwen25-extra-dinosiglip-224px+0_5b` (HuggingFace, public)

### Key differences from Round 2 (OpenVLA-7B)

| | Round 2 (OpenVLA-7B) | Round 5 (MiniVLA 0.5B) |
|---|---|---|
| Model size | 7B params | 0.5B params |
| LLM backbone | Llama-2 7B | Qwen2.5 0.5B |
| Training method | LoRA (r=32) via finetune.py | Full FSDP via train.py |
| Action token mapping | Last 256 vocab tokens | 256 extra tokens (`<extra_0>` .. `<extra_255>`) |
| Batch size | 8 (×2 grad_accum = 16 effective) | 32 (no grad_accum, FSDP constraint) |

---

## Experiment Conditions

| Condition | Action Tokenizer | VQ lambda | Tokens/step |
|-----------|-----------------|-----------|-------------|
| vq_vanilla | `calvin_vq_vanilla_extra_action_tokenizer` | 0 | 4 per 5-step chunk |
| vq_verb λ=0.1 | `calvin_vq_verb01_extra_action_tokenizer` | 0.1 | 4 per 5-step chunk |
| vq_verb λ=0.5 | `calvin_vq_verb_extra_action_tokenizer` | 0.5 | 4 per 5-step chunk |

---

## Implementation

### Files modified in openvla-mini

1. **`prismatic/vla/action_tokenizer.py`** — Added `_calvin_vq_factory()` lazy import
   and three new `ACTION_TOKENIZERS` entries:
   - `calvin_vq_vanilla_extra_action_tokenizer`
   - `calvin_vq_verb_extra_action_tokenizer` (λ=0.5)
   - `calvin_vq_verb01_extra_action_tokenizer` (λ=0.1)

2. **`prismatic/conf/vla.py`** — Added 4 VLA config dataclasses:
   - `Exp_Qwen25_DinoSigLIP_224px_0_5B_Calvin_D_Bin` (base class, not submitted)
   - `Exp_Qwen25_DinoSigLIP_224px_0_5B_Calvin_D_VQ_Vanilla`
   - `Exp_Qwen25_DinoSigLIP_224px_0_5B_Calvin_D_VQ_Verb` (λ=0.5)
   - `Exp_Qwen25_DinoSigLIP_224px_0_5B_Calvin_D_VQ_Verb01` (λ=0.1)
   - Plus `VLARegistry` entries

### VLA Config Parameters

| Parameter | Value |
|-----------|-------|
| base_vlm | `prism-qwen25-extra-dinosiglip-224px+0_5b` |
| data_mix | `calvin_dataset` (D split) |
| expected_world_size | 1 |
| global_batch_size | 32 |
| per_device_batch_size | 32 |
| max_steps | 50,000 |
| learning_rate | 2e-5 |
| lr_scheduler | constant |
| weight_decay | 0.0 |
| max_grad_norm | 1.0 |
| image_aug | True |
| shuffle_buffer_size | 50,000 |
| save_interval | 5,000 |
| train_strategy | fsdp-full-shard |

### SLURM Scripts

- `openvla_experiment/scripts/train_minivla_calvin_vq_vanilla.sh`
- `openvla_experiment/scripts/train_minivla_calvin_vq_verb.sh` (λ=0.5)
- `openvla_experiment/scripts/train_minivla_calvin_vq_verb01.sh` (λ=0.1)

All: 1× GPU, 64G mem, 30h wall time, WANDB_MODE=offline

---

## Data Pipeline

- CALVIN D split RLDS: `/data/user_data/wenjiel2/datasets/calvin_rlds/calvin_dataset/1.0.0/`
- 5124 train episodes, 1011 val episodes
- Transform: `calvin_dataset_transform` (already registered in openvla-mini)
- VQ tokenizer checkpoints:
  - vanilla: `checkpoints/vqvla_ft_vanilla/vqvla_weights.pth`
  - verb λ=0.1: `checkpoints/vqvla_ft_verb_l0.1/vqvla_weights.pth`
  - verb λ=0.5: `checkpoints/vqvla_ft_verb_l0.5/vqvla_weights.pth`

---

## Jobs

| Job ID | Condition | Status | Notes |
|--------|-----------|--------|-------|
| 6613086 | vq_vanilla | SUBMITTED | |
| 6613087 | vq_verb λ=0.1 | SUBMITTED | |
| 6613088 | vq_verb λ=0.5 | SUBMITTED | |

---

## Evaluation Plan

Same pipeline as Round 2:

### Level 1–2: Tokenizer verb probes
Already done in Round 1 — same VQ-VLA tokenizer checkpoints.

### Level 3: LLM action token embedding probe
Train classifier on MiniVLA's action token embeddings (from Qwen2.5 0.5B).
Key question: does the smaller embedding space cluster by verb differently?

### Level 4: Continuous L1 loss
Teacher-forcing on val, decode → continuous actions → L1 vs GT.

### Attention analysis
Action token → verb token attention contrast (same as Round 3).

---

## Results

### Training

| Condition | Train Loss | Val Loss | L1 ↓ | Speed |
|-----------|-----------|---------|------|-------|
| vq_vanilla | — | — | — | — |
| vq_verb λ=0.1 | — | — | — | — |
| vq_verb λ=0.5 | — | — | — | — |

### Verb Probes (Level 3 — LLM embeddings)

| Condition | L3 Linear ↑ | L3 Transformer ↑ |
|-----------|------------|------------------|
| vq_vanilla | — | — |
| vq_verb λ=0.1 | — | — |
| vq_verb λ=0.5 | — | — |

### Cross-Model Comparison (Round 2 vs Round 5)

| Condition | OpenVLA-7B L1 ↓ | MiniVLA 0.5B L1 ↓ |
|-----------|----------------|-------------------|
| vq_vanilla | 0.232 | — |
| vq_verb λ=0.1 | 0.186 | — |
| vq_verb λ=0.5 | 0.196 | — |

---

## Technical Notes

### train.py vs finetune.py

MiniVLA cannot use `finetune.py` because:
1. No HuggingFace-format checkpoint exists for MiniVLA (README: "Converting and
   deploying MiniVLA models and VQ / multi image is not supported yet!")
2. `finetune.py` loads via `AutoModelForVision2Seq.from_pretrained()` which requires
   HF format
3. `train.py` loads via `load()` from model registry (Prismatic format)

### No gradient accumulation in FSDP

`train.py` asserts `grad_accumulation_steps == 1`, so `global_batch_size` must equal
`per_device_batch_size × expected_world_size`. With 1 GPU: both set to 32.

### Action token mapping (extra tokens)

Qwen2.5 adds 256 extra tokens (`<extra_0>` .. `<extra_255>`) to its vocab.
MiniVLA uses these for action bins (via `use_extra=True`) instead of the last 256
regular vocab tokens. This avoids overwriting meaningful language tokens.

### HF token

Base VLM is public (`Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b`).
Scripts create empty `.hf_token` file to satisfy `train.py`'s reader.

---

## Next Steps

- [ ] Submit vq_vanilla job
- [ ] Submit vq_verb λ=0.1 job
- [ ] Submit vq_verb λ=0.5 job
- [ ] Fill in training results
- [ ] Level 3 verb probes (LLM embeddings)
- [ ] Level 4 continuous L1 eval
- [ ] Attention analysis
- [ ] Cross-model comparison with OpenVLA-7B (Round 2)
