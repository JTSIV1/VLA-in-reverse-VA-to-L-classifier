# SimplerEnv Bridge Smoke Eval Handoff

## Goal

Run a minimal closed-loop SimplerEnv evaluation for Bridge-style WidowX tasks
using policy/tokenizer checkpoints from the Bridge sweep. The intended report
metric is final SimplerEnv success rate, with secondary logging for reward,
episode length, truncation, task stats, and action ranges.

The eval code lives in:

```text
policy/scripts/evaluate_simplerenv_lattice.py
scripts/submit_simplerenv_lattice_eval.sh
```

Outputs are written under:

```text
results/simplerenv/
```

The four smoke tasks are:

```text
widowx_spoon_on_towel
widowx_carrot_on_plate
widowx_stack_cube
widowx_put_eggplant_in_basket
```

Each smoke run used 3 seeds per task: `0,1,2`.

## Dynamic OAT TiCE Issue

The original LATTiCE checkpoint uses OAT with a dynamic `ActionEmbeddingWrapper`.
For this checkpoint, action-token embeddings are not a pure token-id lookup.
When an action token appears in the LLM context, the wrapper expects a matching
pre-FSQ latent:

```text
action token embedding = proj(pre_fsq_latent_for_this_action_position)
```

During training or teacher-forced evaluation, those latents are available because
the batch contains the ground-truth continuous action chunk:

```python
latents = action_tokenizer.encode_pre_fsq(raw_actions)
embed_layer.set_current_latents(latents)
```

Closed-loop rollout has the opposite order:

```text
image + instruction -> generated action tokens -> decoded action chunk -> env.step(...)
```

So the model needs pre-FSQ latents while generating action tokens, but those
latents require the continuous action chunk, which is itself the generation
output. The circular dependency is:

```text
need latents to embed generated action tokens
need generated action tokens to decode actions
need decoded/raw actions to compute latents
```

Serial environment rollout does not fix this because the missing latent is
needed inside the same autoregressive token generation call. For example, after
`z1` is generated, it is fed back into the LLM context to generate `z2`; at that
moment the dynamic wrapper needs the latent for the action-token position.

## OAT Prefix Re-encode Approximation

The evaluator implements the closest no-retraining bridge we currently have for
dynamic OAT:

```text
generated token prefix -> OAT detokenize -> approximate action chunk
approximate action chunk -> OAT encode_pre_fsq -> refreshed dynamic TiCE latents
```

This is done one action token at a time using manual greedy decoding. The first
token has no prefix, so it uses a zero latent only for that initial empty-prefix
case. After each generated prefix, the prefix is decoded and re-encoded, and the
resulting pre-FSQ latents are passed into `ActionEmbeddingWrapper` before the
next token.

This is more faithful than a fixed zero-latent rollout, but it is still not the
same distribution as teacher forcing. Training used latents from ground-truth
future action chunks; rollout uses latents reconstructed from the model's own
partial generated token prefix.

## OAT Checkpoint Evaluated

Tokenizer:

```text
/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/bridge_sweep/tokenizers/oat_16_855_4/vlm_clip0.1_pre_fsq/full.pth
```

Policy:

```text
/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/bridge_sweep/policy/oat_16_855_4/vlm_clip0.1_pre_fsq_fullproj/checkpoints/step-050000-epoch-00-loss=0.2729.pt
```

SLURM job:

```text
7697225
```

Output:

```text
results/simplerenv/lattice_oat_clip_pfsq_smoke/
```

Validation:

```text
tokenizer_type: oat
embedding_class: ActionEmbeddingWrapper
uses_dynamic_wrapper: true
dynamic_latent_dim: 256
n_action_tokens: 4
chunk_size: 16
generation_mode: manual_greedy_prefix_reencode
prefix_latent_refreshes: 4
decoded_shape: [16, 7]
```

Result:

| Task | Success | Mean Steps | Mean Reward | Truncation |
|---|---:|---:|---:|---:|
| `widowx_spoon_on_towel` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_carrot_on_plate` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_stack_cube` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_put_eggplant_in_basket` | 0/3 | 120 | 0.0 | 1.0 |
| Overall | 0/12 | 75 | 0.0 | 1.0 |

Behavioral diagnosis: actions were very small, gripper was mostly open, and
SimplerEnv task stats stayed false (`is_src_obj_grasped=false`,
`moved_correct_obj=false`, `src_on_target=false`). The model was running and
producing action chunks, but the robot did not meaningfully interact with the
objects.

## Static VQ-BeT Alternative

To avoid dynamic OAT latents entirely, we tried a Bridge VQ-BeT checkpoint.
VQ-BeT tokenization is rollout-compatible because decoding uses a static code
sequence rather than per-step pre-FSQ latents.

Tokenizer:

```text
/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/bridge_sweep/tokenizers/vq_bet_5_16_2_512/vlm_clip0.1/full.pth
```

Policy:

```text
/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/checkpoints/bridge_sweep/policy/vq_bet_5_16_2_512/vlm_clip0.1/checkpoints/step-040000-epoch-00-loss=0.1899.pt
```

Dry-load job:

```text
7697481
```

Full smoke job:

```text
7697554
```

Output:

```text
results/simplerenv/lattice_vqbet_static_vlm_clip_smoke/
```

Validation:

```text
tokenizer_type: vq_bet
uses_dynamic_wrapper: false
dynamic_latent_dim: 0
embedding_class: Embedding
has_action_wrapper: false
n_action_tokens: 2
chunk_size: 5
generation_mode: manual_greedy
prefix_latent_refreshes: 0
decoded_shape: [5, 7]
```

Important wrinkle: this checkpoint does not reconstruct an
`ActionEmbeddingWrapper`; it loads with plain learned LLM embeddings. That still
avoids the dynamic latent problem, but it means this is not a static codebook
TiCE-wrapper checkpoint. The evaluator used `--no_require_wrapper` for this run.

Result:

| Task | Success | Mean Steps | Mean Reward | Truncation |
|---|---:|---:|---:|---:|
| `widowx_spoon_on_towel` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_carrot_on_plate` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_stack_cube` | 0/3 | 60 | 0.0 | 1.0 |
| `widowx_put_eggplant_in_basket` | 0/3 | 120 | 0.0 | 1.0 |
| Overall | 0/12 | 75 | 0.0 | 1.0 |

Behavioral diagnosis: same broad failure mode as OAT. The first spoon episodes
repeated the same action tokens, decoded to tiny motion, and kept the gripper
near/open (`~0.99`). SimplerEnv again reported no grasp, no correct-object
movement, and no success.

## Implementation Notes

- `evaluate_simplerenv_lattice.py` supports both dynamic OAT and non-dynamic
  VQ-BeT paths.
- Dynamic OAT uses manual greedy token generation with prefix decode/re-encode
  latent refresh.
- Non-dynamic policies use manual greedy token generation without latent refresh.
- `--no_require_wrapper` allows checkpoints that load with plain `Embedding`.
- `submit_simplerenv_lattice_eval.sh` submits with `general --gres=gpu:1
  --mem=64G` and now sets `PYTHONPATH` before checking for `simpler_env`.
- SimplerEnv was installed into `/tmp/${USER}/SimplerEnv-OpenVLA`.
- The job pins `numpy==1.24.4` after SimplerEnv/ManiSkill installation because
  their install flow can upgrade to numpy 2.x, which breaks the older
  TensorFlow/transformers stack imported in this environment.
- Videos were skipped in both runs because `mediapy` is not installed.

## Current Interpretation

The zero success rate is probably not just the dynamic latent issue, because the
VQ-BeT non-dynamic checkpoint fails in the same way. The strongest shared clue is
action collapse: tiny Cartesian/rotation deltas and an open gripper. Plausible
causes include:

- action-interface mismatch between Bridge training actions and SimplerEnv
  control expectations;
- gripper sign or binarization mismatch;
- prompt/image preprocessing mismatch relative to training;
- greedy decoding collapsing to high-prior/no-op action tokens;
- checkpoint/task distribution mismatch despite using Bridge-like SimplerEnv
  tasks.

The next high-value diagnostic is not another full 12-rollout smoke run. It
should be a 1-task, 1-seed ablation that logs actual motion and varies:

```text
execute_k: 1 vs 4 vs full chunk
gripper sign / binarization
action_unnorm_mode: none vs stats
sampling or temperature vs greedy
```

If a known-good OpenVLA/MiniVLA baseline checkpoint exists for these SimplerEnv
Bridge tasks, running it through the same harness would also isolate whether the
remaining issue is the policy checkpoint or our env/action plumbing.
