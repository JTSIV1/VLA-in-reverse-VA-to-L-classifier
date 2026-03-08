"""
evaluate_openvla_rollout.py

Rollout evaluation of fine-tuned OpenVLA models on the CALVIN benchmark.

Implements the standard CALVIN evaluation protocol:
  - 1000 language-conditioned sequences (configurable via --num_sequences)
  - Each sequence has up to 5 chained subtasks
  - Each subtask runs for up to EP_LEN=360 environment steps
  - Reports SR1–SR5 (success rate for completing ≥k consecutive subtasks)
    and average successful sequence length

Supports two action tokenizer conditions:
  - bin:        generate 7 tokens → unnormalize → 1 env step
  - vq_vanilla / vq_verb / vq_verb01:
                generate 4 tokens → VQ decoder → 5 env steps (buffered)

Usage:
  conda run -n calvin_venv python -m openvla_experiment.scripts.evaluate_openvla_rollout \\
      --condition bin \\
      --checkpoint_dir runs/openvla/openvla-7b+calvin_dataset+...--calvin_bin--image_aug \\
      --dataset_path /data/user_data/yashagar/task_D_D \\
      --output_dir results/rollout/bin \\
      --num_sequences 1000

  conda run -n calvin_venv python -m openvla_experiment.scripts.evaluate_openvla_rollout \\
      --condition vq_verb \\
      --checkpoint_dir runs/openvla/...--calvin_vq_verb--image_aug \\
      --vqvla_checkpoint_dir checkpoints/vqvla_ft_verb_l0.5 \\
      --dataset_path /data/user_data/yashagar/task_D_D \\
      --output_dir results/rollout/vq_verb \\
      --num_sequences 1000
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
OPENVLA_DIR  = str(Path("/data/user_data/wenjiel2/Code/openvla-mini"))
CALVIN_DIR   = str(Path("/data/user_data/wenjiel2/Code/calvin"))

for p in [PROJECT_ROOT, OPENVLA_DIR,
          str(Path(CALVIN_DIR) / "calvin_models"),
          str(Path(CALVIN_DIR) / "calvin_env")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# CALVIN's calvin_agent.evaluation.utils imports MCIL which needs pytorch_lightning.
# It's installed in mmml; these path inserts make it importable before the lazy imports below.

EP_LEN = 360


# ── OpenVLA CALVIN model wrapper ───────────────────────────────────────────────

class CalvinOpenVLAModel:
    """
    Wraps a fine-tuned OpenVLA model to implement the CalvinBaseModel interface.

    For bin condition:
      step() generates 7 action tokens → unnormalizes → returns 1 (7-dim) action.

    For VQ conditions:
      step() generates 4 code tokens → VQ decoder → 5 buffered actions.
      Successive calls pop from the buffer; only calls the LLM once per 5 env steps.
    """

    def __init__(self, condition, checkpoint_dir, vqvla_checkpoint_dir, device):
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.condition = condition
        self.device    = device

        print(f"Loading processor from {checkpoint_dir} ...")
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)

        print("Loading model ...")
        self.vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()

        # ── Action tokenizer ───────────────────────────────────────────────────
        if condition == "bin":
            from prismatic.vla.action_tokenizer import ActionTokenizer
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
            self._action_buffer   = None   # not used for bin
        elif condition.startswith("vq_"):
            from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer
            self.action_tokenizer = CalvinVQActionTokenizer(
                self.processor.tokenizer,
                vqvla_checkpoint_dir=vqvla_checkpoint_dir,
            )
            self._action_buffer = []       # buffer for decoded 5-step chunks
            self._vq_groups     = self.action_tokenizer.vq_vae.vqvae_groups      # 4
        else:
            raise ValueError(f"Unknown condition: {condition}")

        print("Model ready.")

    def reset(self):
        """Called by CALVIN between episodes."""
        if self._action_buffer is not None:
            self._action_buffer.clear()

    def step(self, obs: dict, goal: str) -> np.ndarray:
        """
        Args:
            obs:  CALVIN observation dict; obs["rgb_obs"]["rgb_static"] is (H, W, 3) uint8
            goal: language instruction string

        Returns:
            action: (7,) float32 — delta_xyz + delta_euler + gripper
        """
        # ── VQ: pop from buffer if actions remain from previous chunk ──────────
        if self.condition.startswith("vq_") and self._action_buffer:
            return self._action_buffer.pop(0)

        # ── Build model input ──────────────────────────────────────────────────
        rgb = obs["rgb_obs"]["rgb_static"]                  # (H, W, 3) uint8
        image = Image.fromarray(rgb).convert("RGB")

        prompt = f"In: {goal}\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        # ── Generate action tokens ─────────────────────────────────────────────
        if self.condition == "bin":
            # generate 7 tokens (one per action dim), then unnormalize
            with torch.no_grad():
                action = self.vla.predict_action(
                    **inputs,
                    unnorm_key="calvin_dataset",
                    do_sample=False,
                )
            return action.astype(np.float32)

        else:
            # VQ: generate n_groups (4) code tokens
            n_tokens = self._vq_groups
            with torch.no_grad():
                generated = self.vla.generate(
                    **inputs,
                    max_new_tokens=n_tokens,
                    do_sample=False,
                )
            # Extract last n_tokens as code token IDs
            code_token_ids = generated[0, -n_tokens:].cpu().numpy()      # (4,)
            # Decode: token_id = tokenizer_len - 1 - code_id → code_id = tokenizer_len - 1 - token_id
            tok_len  = len(self.processor.tokenizer)
            code_ids = tok_len - 1 - code_token_ids                      # (4,) in [0, 255]
            code_ids = np.clip(code_ids, 0, self.action_tokenizer.vq_vae.vqvae_n_embed - 1)

            # VQ decode: code_ids → z_q → 5 continuous actions
            actions_5 = self._vq_decode(code_ids)                        # (5, 7)

            # Buffer steps 1–4, return step 0
            self._action_buffer.extend(actions_5[1:].tolist())
            return actions_5[0].astype(np.float32)

    def _vq_decode(self, code_ids: np.ndarray) -> np.ndarray:
        """Decode VQ code IDs (4,) → continuous actions (5, 7)."""
        import torch as _torch
        codes = _torch.from_numpy(code_ids.astype(np.int64)).unsqueeze(0)  # (1, 4)
        vq_vae = self.action_tokenizer.vq_vae

        with _torch.no_grad():
            z_q    = vq_vae.draw_code_forward(codes)            # (1, latent_dim)
            decoded = vq_vae.get_action_from_latent(z_q)        # (1, 5, 7) or similar
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            if isinstance(decoded, _torch.Tensor):
                actions = decoded.squeeze(0).cpu().numpy()       # (5, 7)
            else:
                actions = np.array(decoded).squeeze(0)
        return actions.astype(np.float32)


# ── CALVIN evaluation loop ─────────────────────────────────────────────────────

def make_env(dataset_path: str):
    from calvin_env.envs.play_table_env import get_env
    val_folder = Path(dataset_path) / "validation"
    # Only request rgb_static and rgb_gripper — excludes tactile camera (tacto broken in mmml)
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    env = get_env(val_folder, obs_space=obs_space, show_gui=False)
    return env


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in range(i, 6))
        step_success.append(n_success / len(results))
    return step_success


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence,
                      val_annotations, ep_len=EP_LEN):
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        lang_annotation = val_annotations[subtask][0]

        obs = env.get_obs()
        start_info = env.get_info()
        model.reset()

        subtask_success = False
        for step in range(ep_len):
            action = model.step(obs, lang_annotation)
            # CALVIN requires gripper ∈ {-1, 1} (binarize the last action dim)
            action = action.copy()
            action[-1] = 1.0 if action[-1] >= 0 else -1.0
            obs, _, _, current_info = env.step(action)

            # Check if current step solves the subtask
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )
            if len(current_task_info) > 0:
                subtask_success = True
                break

        if subtask_success:
            success_counter += 1
        else:
            break

    return success_counter


def run_rollout_eval(
    condition: str,
    checkpoint_dir: str,
    vqvla_checkpoint_dir: str,
    dataset_path: str,
    output_dir: str,
    num_sequences: int,
    ep_len: int,
    device: str,
) -> dict:
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import print_and_save, get_log_dir
    from omegaconf import OmegaConf
    import hydra

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = CalvinOpenVLAModel(
        condition=condition,
        checkpoint_dir=checkpoint_dir,
        vqvla_checkpoint_dir=vqvla_checkpoint_dir,
        device=device,
    )

    # Build CALVIN env
    print("Building CALVIN env ...")
    env = make_env(dataset_path)

    # Load task oracle and validation annotations
    conf_dir = Path(CALVIN_DIR) / "calvin_models" / "calvin_agent" / ".." / ".." / "conf"
    conf_dir = (Path(CALVIN_DIR) / "calvin_models" / "conf").resolve()
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle  = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Sample evaluation sequences
    eval_sequences = get_sequences(num_sequences)

    results = []
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence,
            val_annotations, ep_len=ep_len,
        )
        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == num_sequences:
            sr = count_success(results)
            sr_str = " | ".join([f"SR{k+1}={v*100:.1f}%" for k, v in enumerate(sr)])
            avg = np.mean(results)
            print(f"  [{i+1}/{num_sequences}] avg={avg:.3f} | {sr_str}")

    sr = count_success(results)
    avg_len = float(np.mean(results))

    summary = {
        "condition":      condition,
        "num_sequences":  num_sequences,
        "avg_len":        avg_len,
        "SR1": sr[0], "SR2": sr[1], "SR3": sr[2], "SR4": sr[3], "SR5": sr[4],
    }
    print("\n=== Rollout Results ===")
    print(f"  Condition : {condition}")
    print(f"  Sequences : {num_sequences}")
    print(f"  Avg len   : {avg_len:.3f}")
    for k, v in enumerate(sr, 1):
        print(f"  SR{k}       : {v*100:.2f}%")

    out_path = os.path.join(output_dir, f"rollout_{condition}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sequence": results}, f, indent=2)
    print(f"Saved to {out_path}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CALVIN rollout evaluation for fine-tuned OpenVLA")
    p.add_argument("--condition", required=True,
                   choices=["bin", "vq_vanilla", "vq_verb", "vq_verb01"])
    p.add_argument("--checkpoint_dir", required=True,
                   help="Merged fine-tuned OpenVLA checkpoint directory")
    p.add_argument("--vqvla_checkpoint_dir", default="",
                   help="VQ-VLA tokenizer checkpoint (required for vq_* conditions)")
    p.add_argument("--dataset_path", default="/data/user_data/yashagar/task_D_D",
                   help="Path to CALVIN dataset root (contains training/ and validation/)")
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "results", "rollout"))
    p.add_argument("--num_sequences", type=int, default=1000,
                   help="Number of evaluation sequences (standard=1000)")
    p.add_argument("--ep_len", type=int, default=EP_LEN,
                   help="Max environment steps per subtask")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    run_rollout_eval(
        condition=args.condition,
        checkpoint_dir=args.checkpoint_dir,
        vqvla_checkpoint_dir=args.vqvla_checkpoint_dir,
        dataset_path=args.dataset_path,
        output_dir=os.path.join(args.output_dir, args.condition),
        num_sequences=args.num_sequences,
        ep_len=args.ep_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
