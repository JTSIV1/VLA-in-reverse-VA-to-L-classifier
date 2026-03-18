# Round 4: RDT2-VQ Fine-Tuning on CALVIN

**Date**: 2026-03-16
**Goal**: Test whether verb-decodable action tokenization improves language grounding
in a VLA where action tokens have **temporal structure** and participate in autoregressive
cross-attention with language tokens.

---

## Motivation

Rounds 1–3 used OpenVLA-mini, where action tokens are only output targets. Although
teacher-forcing lets action token k attend to tokens 1..k-1 + language + vision,
OpenVLA's VQ tokens are **residual refinements** (4 tokens per 5-step window, each
refining the previous level's quantization error) — they carry no temporal progression.

RDT2-VQ (Qwen2.5-VL-7B + RVQ action tokenizer) changes this:
- CNN encoder with 8× temporal compression: 24-step chunk → 3 temporal slots
- Each slot produces tokens from multiple RVQ codebook levels (pos, rot, grip)
- **Later tokens encode later parts of the trajectory**, so autoregressive prediction
  is genuinely "predict what happens next in time"
- When predicting token 10 (e.g., first rot token at temporal slot 2), the model has
  already processed tokens 1–9 (all pos tokens across 3 temporal slots) as input context,
  with their LLM embeddings attending to the verb in the instruction

**Key hypothesis**: Verb-decodable tokenization should have a stronger effect on
language grounding when action tokens carry temporal structure, because the LLM must
use verb semantics to predict the temporal evolution of the action sequence.

---

## Architecture: RDT2-VQ

- **Backbone**: Qwen2.5-VL-7B-Instruct (autoregressive VLM)
- **Action tokenizer**: MultiVQVAE (CNN encoder + ResidualVQ, separate pos/rot/grip)
- **Sequence**: `[image patches] [instruction] [action_token_1] ... [action_token_N]`
- **Training**: standard causal LM loss on action tokens (teacher-forcing)
- **Fine-tuning**: LoRA (r=8, targets: q/k/v/o_proj + gate/up/down_proj)
- **Codebase**: `/data/user_data/wenjiel2/Code/RDT2` (cloned from `thu-ml/RDT2`)

### Action token mapping

Action token IDs are mapped to the LLM vocabulary tail:
```python
action_input_ids = processor.tokenizer.vocab_size - (action_tokens + 1)
```
These embeddings are looked up from the LLM's embedding table and participate in
self-attention during autoregressive generation.

---

## CalvinMultiVQVAE (CALVIN-adapted RVQ tokenizer)

CALVIN actions: 7-dim = delta_xyz(3D) + delta_euler(3D) + gripper(1D)

| Component | Input dims | RVQ codebooks | × temporal slots | = tokens |
|-----------|-----------|---------------|-----------------|----------|
| **pos** (delta_xyz) | 3D | 2 | × 3 | 6 |
| **rot** (delta_euler) | 3D | 2 | × 3 | 6 |
| **grip** (gripper) | 1D | 1 | × 3 | 3 |
| **Total** | 7D | 5 | | **15** |

- action_horizon = 24 steps, CNN 8× compression → 3 temporal slots
- num_embeddings = 1024 per codebook
- embedding_dim = 32, cnn hidden/output = 64
- Script: `RDT2/vqvae/models/calvin_multivqvae.py`

### Verb-decodable variant (VerbDecodableCalvinMultiVQVAE)

Joint training: `L = recon + vq + λ * verb_CE`
- Verb classifier: small transformer over concatenated z_q from all sub-VQVAEs
  at each temporal slot → [CLS] + 3 temporal tokens → 2-layer transformer → classify
- Operates on z_q (differentiable via straight-through), same as Round 1
- 21 sparse CALVIN verb classes, weighted CE

---

## Experiment Conditions

| Condition | Tokenizer | Verb CE | Tokens | Status |
|-----------|-----------|---------|--------|--------|
| rvq_vanilla | CalvinMultiVQVAE | No | 15 | Pending |
| rvq_verb λ=0.5 | CalvinMultiVQVAE + verb CE | Yes | 15 | Pending |

---

## Data Pipeline

### CALVIN → WebDataset conversion

RDT2 expects WebDataset shards (tar files) with per-sample:
- `{idx}.image.jpg` — RGB frame (200×200 rgb_static → resized for Qwen2.5-VL)
- `{idx}.action.npy` — continuous actions (24×7, float32)
- `{idx}.action_token.npy` — pre-tokenized action IDs (15 or 16, int16)
- `{idx}.meta.json` — metadata with instruction key

Plus `instructions.json` — instruction lookup table.

Each training sample: one image frame + next 24 steps of actions + language annotation.
Samples drawn from CALVIN D-train episodes (5124 episodes, ~60 steps avg).

### Normalizer

RDT2 uses `LinearNormalizer` to map actions to [-1, 1] before VQ encoding.
Need to fit on CALVIN training action statistics.

---

## Evaluation Plan

### Levels 1–2: Tokenizer verb decodability (same as Round 1)

- **L1 — z_q latent probe**: classify verb from quantized latent vectors
- **L2 — token ID probe**: classify verb from discrete token IDs

### Level 3: LLM action token embedding probe

Valid for RDT2-VQ because action tokens are embedded via the LLM look-up table
and participate in the forward pass during teacher-forcing. With LoRA, the
embedding table is frozen (pretrained Qwen2.5-VL weights), so L3 tests whether
the *pretrained* embeddings at action token positions carry verb structure
based on the tokenizer's code assignments.

**Note**: This is weaker than a model where embeddings are trainable. If we
switch to full fine-tuning, the embeddings would be updated and L3 becomes
more meaningful.

### Level 4: Continuous L1 loss

Teacher-forcing on CALVIN val, decode predicted tokens → continuous actions → L1 vs GT.
Cross-tokenizer comparable (all conditions compared in continuous action space).

### Attention analysis

Action token → verb token attention contrast (same methodology as Round 3).
Key comparison: does verb CE shift attention toward the verb more strongly
when action tokens have temporal structure (rvq) vs residual structure (vqvla)?

---

## Implementation Steps

- [x] Clone RDT2 repo (`/data/user_data/wenjiel2/Code/RDT2`)
- [x] Write CalvinMultiVQVAE + VerbDecodableCalvinMultiVQVAE
- [ ] Train CalvinMultiVQVAE: vanilla (recon only)
- [ ] Train CalvinMultiVQVAE: verb-decodable (λ=0.5)
- [ ] Fit LinearNormalizer on CALVIN training actions
- [ ] Convert CALVIN → WebDataset shards (one set per tokenizer condition)
- [ ] Write CALVIN dataset config (configs/datasets/calvin.yaml)
- [ ] Fine-tune RDT2-VQ on CALVIN: 2 conditions (LoRA, L40S)
- [ ] Evaluation: L1/L2 verb probes
- [ ] Evaluation: L3 LLM embedding probe
- [ ] Evaluation: L4 continuous L1 loss
- [ ] Evaluation: attention analysis (action→verb contrast)

---

## Jobs

| Job ID | Task | Status | Notes |
|--------|------|--------|-------|
| 6612372 | CalvinMultiVQVAE vanilla training | **RUNNING** | ep10: recon=1.15, vq=0.005, cb: 97/96/32% |
| ~~6612373~~ | CalvinMultiVQVAE verb λ=0.5 | CANCELLED | NaN — double VQ forward bug |
| ~~6612599~~ | CalvinMultiVQVAE verb λ=0.5 | CANCELLED | NaN — classifier params destabilize VQ |
| 6612964 | CalvinMultiVQVAE verb λ=0.5 | **RUNNING** | separate optims + 20-epoch warmup |
| — | CALVIN WebDataset build (rvq_vanilla) | Pending | |
| — | CALVIN WebDataset build (rvq_verb) | Pending | |
| — | RDT2-VQ finetune: rvq_vanilla | Pending | |
| — | RDT2-VQ finetune: rvq_verb | Pending | |

---

## Results

### CalvinMultiVQVAE Training

| Condition | Recon Loss | VQ Loss | Train Verb Acc | Codebook Usage |
|-----------|-----------|---------|----------------|----------------|
| rvq_vanilla | — | — | N/A | — |
| rvq_verb λ=0.5 | — | — | — | — |

### Verb Probe (Levels 1–2)

| Condition | L1 z_q ↑ | L2 token IDs ↑ | Quant. loss (L1−L2) ↓ |
|-----------|----------|----------------|----------------------|
| rvq_vanilla | — | — | — |
| rvq_verb λ=0.5 | — | — | — |

### RDT2-VQ Fine-Tuning

| Condition | L3 LLM embed ↑ | L4 action L1 ↓ | Verb attn contrast ↑ |
|-----------|----------------|----------------|----------------------|
| rvq_vanilla | — | — | — |
| rvq_verb λ=0.5 | — | — | — |

---

## Technical Notes

### RDT2 VQ requires distributed init

RDT2's `VectorQuantizer` uses `dist.all_gather` and `dist.all_reduce` for EMA
codebook updates. Single-GPU training requires `torch.distributed.init_process_group`
with world_size=1, or modification to handle non-distributed mode.

### SLURM resources

- VQ-VAE training: 1× GPU (any), ~1–2h
- WebDataset build: CPU, ~30 min
- RDT2-VQ LoRA fine-tune: 1× L40S (≥32GB VRAM), ~10k steps, ~8h
- Evaluation: 1× L40S, ~1–2h per condition
