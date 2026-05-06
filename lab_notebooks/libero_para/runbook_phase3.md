# Phase 3 — LIBERO-Goal policy fine-tune runbook

Self-contained recipe for submitting two LoRA fine-tunes on a fresh cluster:

| Run | Tokenizer | Aux loss in tokenizer | What we're testing |
|-----|-----------|------------------------|---------------------|
| `minivla_libero_goal_oat_16_855_4_fullproj`              | OAT 16_855_4 | none           | vanilla baseline |
| `minivla_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj`  | OAT 16_855_4 | `--aux_head clip --aux_lambda 0.1 --text_type vlm` (= LATTiCE) | language-tied codebook |

Both runs use the same VLM backbone, dataset, and policy config — the only difference is which tokenizer checkpoint the policy embeds against.

Each run = 50k steps with frozen vision backbone, BS=16. Walltime depends on GPU class:

| GPU | Approx walltime | Why |
|---|---|---|
| H100 | ~5–6 h | best |
| A100 80 GB | ~6–8 h | |
| A100 40 GB | ~7–9 h | |
| L40S | ~11 h | what we used; tight on memory at FP32 weights |

**Use the best GPU your cluster has** — there's no advantage to running on slower hardware. Plan a 14 h SLURM walltime budget regardless.

---

## 0. Compute / hardware

- 1 GPU per run, **as fast as your cluster allows** (H100 > A100 80 GB > A100 40 GB > A6000 > L40S). Any 40+ GB GPU works memory-wise.
- ~96 GB host RAM, 8 CPU cores.
- ~50 GB scratch per run for checkpoints.

## 1. Repos

Clone the two repos and check out the LIBERO-Para branch:

```bash
export WORK=$HOME/work          # adjust
mkdir -p $WORK && cd $WORK

git clone <this-repo-url> VLA-in-reverse-VA-to-L-classifier
git clone https://github.com/Stanford-ILIAD/openvla-mini.git
```

Apply the LIBERO-Goal VLA config patch in the `openvla-mini` checkout. The change is two small additions to `prismatic/conf/vla.py` (a new `Exp_Qwen25_DinoSigLIP_224px_0_5B_LiberoGoal` dataclass and one line in `VLARegistry`) — see [openvla-mini-vla.py.patch](#patch-openvla-mini-vla-py) at the bottom of this doc.

## 2. Conda env

Use the `mmml` env (Python 3.9, PyTorch 2.x, TF for RLDS, transformers, draccus, dill, vector_quantize_pytorch, robosuite, mujoco, libero, h5py, einops, timm, wandb, huggingface_hub).

If starting fresh on the new cluster:

```bash
conda create -n mmml python=3.9 -y
conda activate mmml
# from this repo's root:
pip install -r requirements.txt
# additional minivla deps (not in our requirements.txt because we never need
# to import them outside the policy training pipeline):
pip install draccus dill vector_quantize_pytorch
```

Sanity check from the cluster's GPU node:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# should print: True 1
```

## 3. Data + base VLM (one-time downloads)

Set a data root and grab the three artifacts.

```bash
export DATA_ROOT=/scratch/$USER/data    # adjust
mkdir -p $DATA_ROOT
```

### 3a. LIBERO-Goal RLDS (1.6 GB extracted)

```bash
huggingface-cli download openvla/modified_libero_rlds \
    --repo-type dataset \
    --local-dir $DATA_ROOT/libero_rlds \
    --include 'libero_goal_no_noops/*' 'README.md'
```

Final layout: `$DATA_ROOT/libero_rlds/libero_goal_no_noops/1.0.0/...`. The minivla mixture name `libero_goal_no_noops` is registered in `prismatic/vla/datasets/rlds/oxe/{configs.py,mixtures.py,transforms.py}`.

### 3b. Base VLM (~1.5 GB)

```bash
huggingface-cli download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --local-dir $DATA_ROOT/base_vlm
export BASE_VLM=$DATA_ROOT/base_vlm
```

### 3c. Tokenizer checkpoints (the input to Phase 3)

You have two options:

**Option A — use the tokenizer ckpts from this project (recommended, identical to ours):**

```bash
# Copy from our cluster (or attach the two .pth files):
#   checkpoints/libero_sweep/tokenizers/oat_16_855_4/full.pth                 (~30 MB)
#   checkpoints/libero_sweep/tokenizers/oat_16_855_4_vlm_clip0.1/full.pth     (~30 MB)
mkdir -p $WORK/VLA-in-reverse-VA-to-L-classifier/checkpoints/libero_sweep/tokenizers/{oat_16_855_4,oat_16_855_4_vlm_clip0.1}
# scp the two full.pth files into those two folders.
```

**Option B — retrain tokenizers locally** (~3 h × 2 on 1×L40S; only do this if you can't transfer the ckpts). Steps:

```bash
# 3c.1 — get the LIBERO-Goal HDF5 demos (raw, ~2.7 GB)
huggingface-cli download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --local-dir $DATA_ROOT/libero_data \
    --include 'libero_goal/*'

# 3c.2 — build the per-episode action npy + CSV
cd $WORK/VLA-in-reverse-VA-to-L-classifier
python scripts/convert_libero_to_csv.py \
    --hdf5_dir $DATA_ROOT/libero_data/libero_goal \
    --out_csv  data/libero_goal_episodes.csv \
    --out_dir  $DATA_ROOT/libero_goal_actions

# 3c.3 — pre-compute Qwen2.5 text embeddings (used by --aux_head clip)
python scripts/precompute_vlm_text_embeddings.py \
    --csv data/libero_goal_episodes.csv \
    --output tokenization/vlm_text_embeddings_libero.pt

# 3c.4 — train vanilla OAT
python tokenization/train_tokenizer.py \
    --tokenizer oat \
    --dataset libero_goal \
    --libero_csv         data/libero_goal_episodes.csv \
    --libero_actions_dir $DATA_ROOT/libero_goal_actions \
    --epochs 200 --batch_size 64 \
    --set horizon=16 fsq_levels=[8,5,5] num_registers=4 \
    --tag libero_oat_16_855_4 \
    --save_dir checkpoints/libero_sweep/tokenizers

# 3c.5 — train LATTiCE (vlm_clip0.1) OAT
python tokenization/train_tokenizer.py \
    --tokenizer oat \
    --dataset libero_goal \
    --libero_csv         data/libero_goal_episodes.csv \
    --libero_actions_dir $DATA_ROOT/libero_goal_actions \
    --epochs 200 --batch_size 64 \
    --set horizon=16 fsq_levels=[8,5,5] num_registers=4 \
    --aux_head clip --aux_lambda 0.1 \
    --text_type vlm \
    --text_model tokenization/vlm_text_embeddings_libero.pt \
    --tag libero_oat_16_855_4 \
    --save_dir checkpoints/libero_sweep/tokenizers
```

After Option B, the two ckpts will be at `checkpoints/libero_sweep/tokenizers/oat_16_855_4/full.pth` and `checkpoints/libero_sweep/tokenizers/oat_16_855_4_vlm_clip0.1/full.pth` — same as Option A.

## 4. (Optional) Cluster overrides

If you set `WORK` and `DATA_ROOT` as in §3 above, you don't need to edit anything — the submitter [scripts/submit_libero_phase3.sh](../../scripts/submit_libero_phase3.sh) reads them as env vars with sensible defaults. Just run it (§6).

For non-default cluster directives (different partition name, GPU constraint, excluded nodes), pass them as env vars:

```bash
export SLURM_PARTITION=gpu          # default: general
export SLURM_CONSTRAINT=h100|a100   # default: unset; uncomment to pin to a GPU class
export SLURM_EXCLUDE=node-bad-1     # default: unset
export SLURM_TIME=14:00:00          # default: 14:00:00
```

(See the header of `submit_libero_phase3.sh` for the full list of overridable env vars — you can also override individual paths like `BASE_VLM=...` if your layout differs from the runbook.)

If you'd rather inspect/customize each job script directly instead of using the wrapper, there are also two standalone sbatch files: [scripts/libero_phase3_vanilla.sbatch](../../scripts/libero_phase3_vanilla.sbatch) and [scripts/libero_phase3_lattice.sbatch](../../scripts/libero_phase3_lattice.sbatch). Both have an "EDIT ME" path block near the top.

## 5. Smoke test (recommended, ~2 min)

Before burning ~22 GPU-h, verify the full pipeline works on this cluster with a 30-step run. From an interactive GPU node:

```bash
conda activate mmml
cd $WORK/openvla-mini
export PRISMATIC_DATA_ROOT=$DATA_ROOT/libero_rlds
export WANDB_MODE=offline

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops \
    --vla.base_vlm $BASE_VLM \
    --data_root_dir $DATA_ROOT/libero_rlds \
    --run_root_dir /tmp/smoke_libero \
    --run_id smoke_test \
    --image_aug True \
    --save_interval 100000 \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 4 \
    --vla.per_device_batch_size 4 \
    --vla.freeze_vision_backbone True \
    --vla.max_steps 30 \
    --vla.d_fixed 896 \
    --vla.action_tokenizer "sweep:oat:$WORK/VLA-in-reverse-VA-to-L-classifier/checkpoints/libero_sweep/tokenizers/oat_16_855_4/full.pth"
```

Expected: 30 steps in ~75 s, loss starts ~17, drops to ~3 by step 30, ~0.8 s/step steady-state. If you see this, the real submission will work. (Then `rm -rf /tmp/smoke_libero` — checkpoint is ~4 GB.)

## 6. Submit and walk away

```bash
cd $WORK/VLA-in-reverse-VA-to-L-classifier
bash scripts/submit_libero_phase3.sh
```

That's it. The wrapper will:

1. Verify your `OPENVLA_DIR`, `RLDS_DIR`, `BASE_VLM`, `TOK_DIR/{oat_16_855_4,oat_16_855_4_vlm_clip0.1}/full.pth`, and conda env are all in place — fails fast with a clear error if anything is missing.
2. Generate two on-the-fly sbatch scripts and submit them.
3. Sleep 5 s and check each job's state via `sacct` — if the cluster QOS instant-rejects the submission (we've seen this on Babel when over `MaxJobsPU`), it prints a clear "FAILED immediately" message instead of silently looking submitted.
4. Print `Submitted: ... [<jobid>, PENDING]` for each job, and `Done. Submitted 2 / 2 Phase 3 jobs.`

Each job's outputs (ckpts + SLURM logs) are written world-readable (`umask 022` + a final `chmod -R o+rX`) so the project owner can pick them up without any further action from you.

Logs land at `$PROJECT_DIR/logs/<jobid>_pol_libero_goal_oat_16_855_4*.{out,err}` and ckpts at `$POLICY_DIR/minivla_libero_goal_oat_16_855_4_*/checkpoints/`.

## 7. Outputs

Each policy run lands at:

```
$POLICY_DIR/minivla_libero_goal_oat_16_855_4_fullproj/
$POLICY_DIR/minivla_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj/
```

Inside each: `checkpoints/step-*.pt` (final + every 5k), `config.{json,yaml}`, `dataset_statistics.json`, and a JSONL training log. Final ckpts are ~4 GB each.

### Making the outputs readable to the project owner

Both sbatch files set `umask 022` and run a final `chmod -R o+rX` over the run dir + log files, so the project owner can pick up the checkpoints without further action. If the run is still in progress and you'd like the partial output readable now, run:

```bash
chmod -R o+rX $POLICY_DIR/minivla_libero_goal_oat_16_855_4_fullproj \
              $POLICY_DIR/minivla_libero_goal_oat_16_855_4_vlm_clip0.1_fullproj
chmod o+r $PROJECT_DIR/logs/<jobid>_pol_libero_goal_oat_16_855_4*.{out,err}
```

(Adjust `<jobid>` for the SLURM job IDs.)

## 8. Sanity checks while training

After the first job has been running ~15 min:

```bash
tail -F $PROJECT_DIR/logs/<jobid>_pol_libero_goal_oat_16_855_4_fullproj.out
```

Healthy signs:
- `Mixture libero_goal_no_noops` line appears once at startup, weight `1.000000`.
- `[DynEmbed]` log line confirms the d_fixed=896 OAT projection path is firing.
- `Loss` ticks down from ~17 to <3 within the first ~200 steps.
- Each step takes ~0.8 s once warmed up.

If loss plateaus above 5 past step ~2k, or step time exceeds 2 s, something is off (probably dataloader). Compare against the smoke run.

---

## Patch: `openvla-mini/prismatic/conf/vla.py`

Add this dataclass right after `Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge`:

```python
@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_LiberoGoal(Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-goal-no-noops"
    data_mix: str = "libero_goal_no_noops"
    image_sequence_len: int = 2
    use_wrist_image: bool = True
```

And add this line to the `VLARegistry` enum (right after `QWEN25_DINOSIGLIP_224PX_0_5B_BRIDGE`):

```python
QWEN25_DINOSIGLIP_224PX_0_5B_LIBERO_GOAL = Exp_Qwen25_DinoSigLIP_224px_0_5B_LiberoGoal
```

That's all. The dataset config (`libero_goal_no_noops`), the mixture entry, and the dataset transform are already in the upstream `openvla-mini` repo (`prismatic/vla/datasets/rlds/oxe/{configs.py,mixtures.py,transforms.py}`).
