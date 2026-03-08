# Verb-Decodable Action Tokenizer for OpenVLA-mini

## Hypothesis

If discrete action tokens encode verb semantics (e.g., "push", "grasp", "lift"),
then a VLA that predicts these tokens will better understand and execute language
instructions -- improving task success rate compared to a vanilla reconstruction-only
tokenizer.

## Approach

### Stage 1: Fine-tune VQ-VLA tokenizer with verb decodability loss

Take the pretrained [VQ-VLA](https://github.com/xiaoxiao0406/VQ-VLA) action
tokenizer (causal conv VAE + 4-group ResidualVQ, trained on 100M+ synthetic
trajectories) and fine-tune it on CALVIN actions with an auxiliary verb
classification loss:

```
L_total = L_recon + 5 * L_vq + lambda * L_verb_CE
```

The verb classifier operates on the quantized latent codes, encouraging the
codebook to capture verb-relevant structure. The tokenizer architecture and
`get_code()` / `draw_code_forward()` / `get_action_from_latent()` interface
remain unchanged, so it drops in as a replacement for the vanilla tokenizer.

- Input: 5-step action windows (T, 7) from CALVIN split D
- Output: 4 discrete codes per window (ResidualVQ with 4 quantizers x 256 codebook)
- Verb labels: 21 sparse classes extracted from CALVIN language annotations

### Stage 2: Fine-tune OpenVLA-mini (MiniVLA) on CALVIN

[OpenVLA-mini](https://github.com/Stanford-ILIAD/openvla-mini) with the
Qwen2.5-0.5B backbone (MiniVLA) takes (image, language_instruction) and
autoregressively predicts discrete action tokens.

Fine-tune MiniVLA on CALVIN split D twice:
- **Baseline**: vanilla VQ-VLA tokenizer (frozen, pretrained)
- **Experimental**: verb-decodable VQ-VLA tokenizer (frozen, from Stage 1)

Both conditions use LoRA fine-tuning with identical hyperparameters.

### Evaluation

**Primary (offline, teacher forcing)**: On held-out CALVIN D validation set,
compare action prediction loss and token accuracy between baseline and
experimental conditions. No simulator needed.

**Secondary (online, optional)**: Deploy in CALVIN simulator, measure task
success rate over 1000 chained tasks.

## Data

All experiments use **CALVIN split D** (`task_D_D`) only:
- ~3,300 annotated trajectories with language instructions, RGB observations,
  and 7-DoF actions
- Train/val split provided by CALVIN (training/ and validation/ subdirectories)
- Both stages train on D-train, evaluate on D-val
- CALVIN data must be converted to **RLDS format** (TFRecord-based) for
  OpenVLA-mini's data pipeline

**Note on data overlap**: Stage 1 (tokenizer) and Stage 2 (VLA) both use
D-train. This is acceptable because the tokenizer is frozen before VLA training
and the VLA never sees verb labels — analogous to how pretrained word embeddings
are trained on the same corpus as downstream models.

## Architecture

```
                          Stage 1                           Stage 2
                    ┌─────────────────┐            ┌──────────────────────┐
                    │   VQ-VLA        │            │   MiniVLA (0.5B)     │
  action (T,7) ──> │ CausalConvVAE   │            │                      │
    5-step windows  │   + ResidualVQ  │──codes──>  │ Qwen2.5-0.5B LLM    │
                    │   + verb head   │  (freeze)  │   + vision encoder   │
                    │                 │            │   + action tokenizer │
                    └─────────────────┘            └──────────────────────┘
                    L = recon + vq                  L = next-token CE
                        + λ·verb_CE                   (action tokens only)
```

### VQ-VLA tokenizer specs
- Encoder: 4-block causal conv (128→256→256→512 channels), SiLU, group norm
- Latent dim: 128
- Quantization: ResidualVQ, 4 quantizers x 256 codebook x 128-d
- Decoder: 4-block causal conv (mirrored)
- Temporal embeddings: action-type PE + time PE
- Pretrained checkpoint: `checkpoints/vqvla_pretrained/action_tokenizer_weight/all_data_vq.pth`
- Config: `vqvla/config/config_action_type_pe_time_pe.json`

### MiniVLA specs
- Vision: DINOv2 + SigLIP fused backbone (224px)
- Language: Qwen2.5-0.5B
- Action decoding: autoregressive over VQ code tokens
- Fine-tuning: LoRA (r=32)

## Directory structure

```
openvla_experiment/
├── README.md                    # This file
├── scripts/
│   ├── finetune_tokenizer.py    # Stage 1: VQ-VLA + verb loss fine-tuning
│   ├── finetune_vla.py          # Stage 2: MiniVLA fine-tuning on CALVIN
│   ├── eval_teacher_forcing.py  # Offline evaluation (action prediction loss)
│   └── submit_experiment.sh     # SLURM submission for full pipeline
├── configs/
│   └── experiment.yaml          # Hyperparameters for both stages
└── data_conversion/
    └── calvin_to_rlds.py        # CALVIN → RLDS format converter
```

### Shared infrastructure (in parent directory)
- `vqvla/` -- VQ-VLA model architecture + configs (vendored from VQ-VLA repo)
- `tokenization/` -- Action tokenizer modules (FAST, VQ-VAE, etc.)
- `config.py` -- CALVIN data paths
- `utils.py` -- Language annotation loading + verb extraction

## Experiment plan

| Step | Script | Compute | Notes |
|------|--------|---------|-------|
| 0. Convert CALVIN D → RLDS | `calvin_to_rlds.py` | CPU, ~30 min | One-time |
| 1a. Fine-tune VQ-VLA (vanilla, control) | `finetune_tokenizer.py` | 1 GPU, ~1-2h | recon + vq only (lambda=0), on D-train |
| 1b. Fine-tune VQ-VLA (verb-decodable) | `finetune_tokenizer.py` | 1 GPU, ~1-2h | recon + vq + lambda*verb_CE, on D-train |
| 2a. Fine-tune MiniVLA (vanilla tokenizer) | `finetune_vla.py` | 1-2 GPU, ~4-8h | LoRA on Qwen2.5-0.5B |
| 2b. Fine-tune MiniVLA (verb tokenizer) | `finetune_vla.py` | 1-2 GPU, ~4-8h | Same hyperparams |
| 3. Evaluate (teacher forcing) | `eval_teacher_forcing.py` | 1 GPU, ~30 min | Compare loss + token acc |

## Dependencies

- OpenVLA-mini: `pip install -e .` from cloned repo
- VQ-Bet (for ResidualVQ): `pip install git+https://github.com/jayLEE0301/vq_bet_official`
- CALVIN environment (for online eval only): `pip install calvin-env`
- Standard: PyTorch 2.x, transformers, timm, tensorflow-datasets (for RLDS)

## References

- VQ-VLA: Wang et al., "VQ-VLA: Improving Vision-Language-Action Models via Scaling
  Vector-Quantized Action Tokenizers", ICCV 2025. [arXiv:2507.01016](https://arxiv.org/abs/2507.01016)
- OpenVLA: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024.
  [GitHub](https://github.com/Stanford-ILIAD/openvla-mini)
- CALVIN: Mees et al., "CALVIN: A Benchmark for Language-Conditioned Policy Learning
  for Long-Horizon Robot Manipulation Tasks", RA-L 2022.
