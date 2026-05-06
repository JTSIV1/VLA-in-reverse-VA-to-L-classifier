# LATTiCE on LIBERO-Para

**Goal**: behavioral synonym-robustness signal for LATTiCE — measure success-rate
delta between paraphrased and original instructions, vanilla OAT vs LATTiCE OAT,
on LIBERO-Goal-Para.

**Why**: BridgeV2 TF-L1 was a null result for synonym robustness — TF eval is too
strong-supervised. LIBERO-Para tests rollout success, where verb-token drift
compounds. Existing VLAs lose 22.8–51.9 pp under paraphrase
([Kim et al. 2026](https://arxiv.org/abs/2603.28301)). Plenty of headroom.

## Scope (narrowed per 2026-05-05 instruction)

- **One config**: OAT `16_855_4` (chunk_size=16, fsq_levels=[8,5,5], num_registers=4)
  — same as our existing HERO so the BridgeV2 ↔ LIBERO comparison is apples-to-apples.
- **Two policies only**:
  1. *Vanilla OAT*  — `aux=none`, random embedding init.
  2. *LATTiCE*       — `aux=vlm_clip0.1`, TiCE (`d_fixed=896`).
- **One task suite**: LIBERO-Goal (10 tasks; what the LIBERO-Para paper uses).

This drops Phase 3 from 4 conditions to 2, halving the policy-fine-tune cost.

## Repo layout (cloned to `/data/user_data/wenjiel2/Code/LIBERO-Para`)

| Path | Role |
|---|---|
| `libero/libero/bddl_files/libero_goal/` | 10 original-instruction BDDL envs |
| `libero/libero/bddl_files/libero_para/` | 4092 paraphrased BDDLs (eval-only) |
| `libero/libero/init_files/libero_para/` | Per-task init states (`evalN.pruned_init`) |
| `eval_scripts/eval_template.py` | Abstract eval interface (load_model + predict_action) |
| `eval_scripts/examples/eval_openvla_oft.py` | Reference 553-line eval; handles BDDL parsing, JSON logging, rollout loop |
| `metrics/analyze_results.py` | Reads `logs_para/<model>/seed*/eval*.json`, computes success rate + PRIDE |
| `metrics/PRIDE_metric_playground.ipynb` | PRIDE worked example |

## Phases

### Phase 1 — env setup (~½ day, in flight)
- Conda env `libero-para` (Python 3.10) per `eval_guides/openvla_oft_goal.md`.
- LIBERO base + robosuite 1.4.0 + bddl 1.0.1 + spaCy.
- Smoke test: a single rollout on `libero_goal` task 0 with the OpenVLA-OFT
  HuggingFace ckpt to verify mujoco-egl + env work.

### Phase 2 — tokenizer training on LIBERO-Goal (~1–2 days, ~6 GPU-h)
- Convert LIBERO-Goal demos (HDF5) → CSV `data/libero_goal_episodes.csv`
  in the same shape as `bridge_episodes_filtered.csv` (instruction, verb,
  action_chunk_path).
- Train two tokenizers at `oat_16_855_4`:
  1. `oat_libero_goal/full.pth`              (vanilla, no aux)
  2. `oat_libero_goal_vlm_clip0.1/full.pth`  (Qwen contrastive aux, λ=0.1)

### Phase 3 — policy fine-tune (~3 days, ~20 GPU-h)
- LoRA fine-tune the MiniVLA Qwen2.5-0.5B on LIBERO-Goal demos with each tokenizer.
- Two runs:
  1. `minivla_libero_goal_oat_vanilla`              (vanilla tokenizer + random emb)
  2. `minivla_libero_goal_oat_vlm_clip0.1_fullproj` (LATTiCE = vlm_clip + TiCE)

### Phase 4 — eval (~2 days, ~10–15 GPU-h)
- Adapt `eval_scripts/eval_template.py` → `eval_scripts/examples/eval_minivla_lattice.py`
  — copy the BDDL-parsing + JSON-logging structure from the OpenVLA-OFT script,
  swap the model loader for our minivla load_vla.
- Run rollouts on **both** suites:
  - `libero_goal` (original instructions, 10 tasks × N seeds) — baseline
  - `libero_para` (4092 BDDL files = ~409 paraphrases × 10 tasks) — perturbed
- For each (policy, suite, seed) write `logs_para/<policy>/seed<N>/eval*.json`.

### Phase 5 — analysis (~½ day)
- Run `metrics/analyze_results.py` per policy → success rate + PRIDE.
- Summary plot: 2-bar chart per policy (original SR vs paraphrase SR), and a
  per-paraphrase-distance breakdown.

## Compute budget (1 A6000/L40S)

| Phase | GPU-hours |
|---|---:|
| 2 — tokenizer ×2 | ~6 |
| 3 — LoRA policy ×2 | ~20 |
| 4 — eval ×2 ×2 suites | ~12 |
| **Total** | **~40 GPU-hours** |

## Risks
- **Mujoco-egl setup on Babel** — historically finicky; eat half-day of debug.
- **LIBERO action normalization** vs Bridge — verify our vanilla OAT hits ~95+%
  on `libero_goal` original-instruction eval before trusting paraphrase numbers.
- **Demo count is small** (~50 demos × 10 tasks = ~500 episodes for tokenizer
  training, 10× fewer than Bridge). If verb-MI on the new tokenizer is poor, fall
  back to training the tokenizer on LIBERO-90 (90 task suites, 9000 demos).

## Decision points
- After Phase 1: is mujoco rollout working? If not, bail before tokenizer training.
- After Phase 3: does our vanilla OAT match the paper's OpenVLA-OFT 97.9% on
  original-instruction LIBERO-Goal? If we're 20 pp lower, the comparison isn't
  fair and we'd need to debug data normalization first.

## Progress log

**2026-05-05 (afternoon):**
- ✅ **Phase 1** env install (`libero-para` conda env): LIBERO + robosuite +
  bddl + spaCy + transformers + torch + h5py all imported cleanly.
- ✅ **Phase 1.5** mujoco-egl smoke test on L40S — `libero_goal` task 0
  (`Open the middle layer of the drawer`) loads + steps with random actions.
- ✅ **Phase 2 data prep** — downloaded LIBERO-Goal HDF5s (~2.7 GB, 10 files)
  to `/data/user_data/wenjiel2/datasets/libero_data/libero_goal/`. Conversion
  via `scripts/convert_libero_to_csv.py` produced
  `data/libero_goal_episodes.csv` (500 rows) + per-episode action npys at
  `/data/user_data/wenjiel2/datasets/libero_goal_actions/`.
- Verb distribution: `put` 300, `open` 100, `push` 50, `turn` 50 — coarser than
  Bridge but fine since both target conditions skip the verb-classifier head.
- Pause point before kicking off tokenizer training: still need to add
  `LiberoActionDataset` class so `tokenization/train_tokenizer.py` can read the
  new layout without Bridge-specific path hardcoding. ~30-60 min code change,
  then ready to launch ~3 h × 2 GPU tokenizer training.

**2026-05-05 (evening):**
- ✅ **Phase 2 tokenizer training** — both OAT 16_855_4 tokenizers trained on
  interactive node:
  - Vanilla: `checkpoints/libero_sweep/tokenizers/oat_16_855_4/full.pth`
    (129 epochs, recon 0.098, 16 active codes)
  - LATTiCE (`vlm_clip0.1`):
    `checkpoints/libero_sweep/tokenizers/oat_16_855_4_vlm_clip0.1/full.pth`
    (76 epochs, recon 0.119, 9 active codes, R@5 = 76%, clip val ≈ 1.85)
- ✅ **Verb-probe sanity check** on the trained latents (`action_rep=latent`,
  default 4-layer / d=128 transformer, 100 epochs, 90/10 split, vanilla CE).
  H(Y) ≈ 1.571 bits over {put 300, open 100, push 50, turn 50}.
  | Tokenizer | Best val loss | Best Acc | Best MacroF1 | Verb-MI |
  |---|---|---|---|---|
  | Vanilla OAT 16_855_4 | 0.4302 (ep 98) | 90.0% | 89.5% | 0.951 bits |
  | LATTiCE (vlm_clip0.1) | 0.1755 (ep 86) | 100.0% | 100.0% | 1.318 bits |
  +0.37 bits / +10 pp acc for LATTiCE. Direction matches Bridge but val set is
  only 50 episodes and verbs are trivially correlated with task identity, so
  treat as a structural sanity signal, not a headline result. Behavioral
  (paraphrase rollout) results from Phases 3–4 remain the load-bearing test.
  Logs: `logs/libero_probe/cs_oat_libero_{vanilla,vlm_clip}_latent_f0.log`,
  results: `results/libero_compression_sweep/cs_oat_libero_*_latent_f0.json`.
- ✅ **Phase 3 launch (2026-05-05 evening)** — submitted two policy fine-tune
  jobs (`scripts/submit_libero_goal_oat_fullproj.sh`):
  - `7744812 pol_libero_goal_oat_16_855_4_fullproj` (vanilla OAT)
  - `7744813 pol_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj` (LATTiCE OAT)
  Both: `prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops` config
  (new entry in `openvla-mini/prismatic/conf/vla.py`), 1×L40S, BS=16, 50k steps,
  d_fixed=896, image_sequence_len=2, use_wrist_image=True, 14 h SLURM budget.
  RLDS data downloaded from `openvla/modified_libero_rlds:libero_goal_no_noops`
  (1.6 GB) at `/data/user_data/wenjiel2/datasets/libero_rlds/`.
- Smoke test before submit: 30 steps in ~75 s, loss 17.9 → 2.6, ~0.8 s/step
  steady-state — extrapolates to ~11 h for 50k steps. DynEmbed (dynamic OAT
  encoder embedding) path firing correctly with d_fixed=896, latent_dim=256.

### Phase 3 design note — TiCE latent path (training and rollout)

We initially worried about a train/rollout asymmetry on the TiCE input
embedding. After reading the upstream code, **there isn't one** — both paths
already use `CalvinSweepActionTokenizer.compute_prefix_embeddings`
(`openvla-mini/prismatic/vla/calvin_sweep_action_tokenizer.py:335`):

For each LLM register position `k = 0..K-1`, the 256-d input embedding is
computed by:
1. Quantizing the chunk → discrete tokens (`B, K`).
2. Decoding using **only the first k tokens** (the rest are masked by
   `MaskedNestedDropout`'s `eval_keep_k=k`) → a coarse chunk reconstruction.
3. Re-encoding that coarse chunk → taking the register-k latent.

So position k's embedding depends on tokens 0..k — and on nothing else. At
training, those tokens come from tokenizing the GT chunk (teacher forcing); at
rollout, they come from the policy's own predictions. The
decode-then-re-encode function in steps 2-3 is identical in both cases, so
there is no train-time-only "GT-encoder" leak the way the legacy
`encode_pre_fsq` path had (see
`lab_notebooks/compression_rate_sweep/oat_tice_leakage.md` for that history).

`compute_prefix_embeddings` is the default; the LEAKY `encode_pre_fsq` path
only fires if `OAT_TICE_LEAKY_EMBED=1` is set in the environment (we don't).

**No precompute lookup is required for correctness.** It would only be a speed
optimization at rollout (cache `prefix_tokens → embedding` to skip the
decode-encode each call). Worth revisiting at Phase 4 if rollout latency
becomes a bottleneck, but the prefix space for OAT 16_855_4 is large
(`fsq_levels=[8,5,5]` → 200 codes/register, ~1.6B prefixes for the longest
register), so a sparse cache is more realistic than a dense table.
