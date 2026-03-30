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
  conda run -n calvin_venv python -m policy.scripts.evaluate_openvla_rollout \\
      --condition bin \\
      --checkpoint_dir runs/openvla/openvla-7b+calvin_dataset+...--calvin_bin--image_aug \\
      --dataset_path /data/user_data/yashagar/task_D_D \\
      --output_dir results/rollout/bin \\
      --num_sequences 1000

  conda run -n calvin_venv python -m policy.scripts.evaluate_openvla_rollout \\
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as C  # noqa: E402

OPENVLA_DIR  = C.OPENVLA_DIR
CALVIN_DIR   = C.CALVIN_DIR

for p in [OPENVLA_DIR,
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

    def __init__(self, condition, checkpoint_dir, vqvla_checkpoint_dir, device,
                 sweep_tokenizer_type="", sweep_checkpoint_path=""):
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
        self._action_buffer = None
        self._n_action_tokens = 7  # default for bin
        self._is_chunk_tokenizer = False

        if sweep_tokenizer_type and sweep_checkpoint_path:
            from prismatic.vla.calvin_sweep_action_tokenizer import CalvinSweepActionTokenizer
            self.action_tokenizer = CalvinSweepActionTokenizer(
                self.processor.tokenizer,
                sweep_tokenizer_type, sweep_checkpoint_path,
                device=device, use_extra=True,
            )
            self._action_buffer = []
            self._n_action_tokens = self.action_tokenizer.n_codes_per_chunk
            self._is_chunk_tokenizer = True
            print(f"Sweep tokenizer: {sweep_tokenizer_type}, "
                  f"{self._n_action_tokens} tokens/chunk, "
                  f"chunk_size={self.action_tokenizer.chunk_size}")
        elif condition == "bin":
            from prismatic.vla.action_tokenizer import ActionTokenizer
            self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        elif condition.startswith("vq_"):
            from prismatic.vla.calvin_vq_action_tokenizer import CalvinVQActionTokenizer
            self.action_tokenizer = CalvinVQActionTokenizer(
                self.processor.tokenizer,
                vqvla_checkpoint_dir=vqvla_checkpoint_dir,
            )
            self._action_buffer = []
            self._n_action_tokens = self.action_tokenizer.vq_vae.vqvae_groups
            self._is_chunk_tokenizer = True
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
        # Pop from buffer if actions remain from previous chunk
        if self._is_chunk_tokenizer and self._action_buffer:
            return np.array(self._action_buffer.pop(0), dtype=np.float32)

        # Build model input
        rgb = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(rgb).convert("RGB")
        prompt = f"In: {goal}\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        if not self._is_chunk_tokenizer:
            # Bin: generate 7 tokens, unnormalize
            with torch.no_grad():
                action = self.vla.predict_action(
                    **inputs,
                    unnorm_key="calvin_dataset",
                    do_sample=False,
                )
            return action.astype(np.float32)

        # Chunk tokenizer (VQ-VLA, VQ-BeT, OAT, QueST): generate n_action_tokens
        n_tokens = self._n_action_tokens
        with torch.no_grad():
            generated = self.vla.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
            )
        code_token_ids = generated[0, -n_tokens:].cpu().numpy()

        # Decode token IDs → full action chunk for rollout
        actions = self.action_tokenizer.decode_full_chunk(code_token_ids)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.atleast_2d(actions).astype(np.float32)  # (chunk_size, 7)

        # Buffer remaining steps, return first
        if len(actions) > 1:
            self._action_buffer.extend(actions[1:].tolist())
        return actions[0]


class CalvinFSDPModel:
    """
    Wraps a native prismatic VLA loaded from an FSDP .pt checkpoint
    (MiniVLA from-scratch training) to implement the CalvinBaseModel interface.
    """

    def __init__(self, fsdp_checkpoint, device,
                 sweep_tokenizer_type="", sweep_checkpoint_path=""):
        from prismatic.models import load_vla
        from transformers import GenerationMixin

        self.device = device

        print("Loading native VLA from FSDP checkpoint: {} ...".format(fsdp_checkpoint))
        self.vla = load_vla(fsdp_checkpoint, load_for_training=False)
        self.vla = self.vla.to(device)
        self.vla.eval()

        self._image_transform = self.vla.vision_backbone.get_image_transform()
        self._tokenizer = self.vla.llm_backbone.tokenizer
        self._GenerationMixin = GenerationMixin  # bypass PrismaticVLM.generate()

        # Action tokenizer setup
        self._action_buffer = None
        self._n_action_tokens = 7
        self._is_chunk_tokenizer = False

        if sweep_tokenizer_type and sweep_checkpoint_path:
            from prismatic.vla.calvin_sweep_action_tokenizer import CalvinSweepActionTokenizer
            self.action_tokenizer = CalvinSweepActionTokenizer(
                self._tokenizer,
                sweep_tokenizer_type, sweep_checkpoint_path,
                device=device, use_extra=True,
            )
            self._action_buffer = []
            self._n_action_tokens = self.action_tokenizer.n_codes_per_chunk
            self._is_chunk_tokenizer = True
            print("Sweep tokenizer: {}, {} tokens/chunk, chunk_size={}".format(
                sweep_tokenizer_type, self._n_action_tokens,
                self.action_tokenizer.chunk_size))
        else:
            # Bin — use the model's built-in action tokenizer
            self.action_tokenizer = self.vla.action_tokenizer

        print("FSDP model ready.")

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
        # Pop from buffer if actions remain from previous chunk
        if self._is_chunk_tokenizer and self._action_buffer:
            return np.array(self._action_buffer.pop(0), dtype=np.float32)

        rgb = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(rgb).convert("RGB")

        if not self._is_chunk_tokenizer:
            # Bin: use native predict_action (handles prompt, tokenization, unnorm)
            with torch.no_grad():
                action = self.vla.predict_action(
                    image, goal,
                    unnorm_key="calvin_dataset",
                    do_sample=False,
                )
            return action.astype(np.float32)

        # Chunk tokenizer: build native inputs and generate n_action_tokens
        prompt_builder = self.vla.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message="What action should the robot take to {}?".format(goal.lower()),
        )
        prompt_text = prompt_builder.get_prompt()

        input_ids = self._tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)

        pixel_values = self._image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}

        n_tokens = self._n_action_tokens
        autocast_dtype = self.vla.llm_backbone.half_precision_dtype
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vla.enable_mixed_precision_training):
            # Bypass PrismaticVLM.generate(image, prompt_text) — call
            # GenerationMixin.generate(input_ids=..., pixel_values=...) directly
            generated = self._GenerationMixin.generate(
                self.vla,
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=n_tokens,
                do_sample=False,
            )
        code_token_ids = generated[0, -n_tokens:].cpu().numpy()

        # Decode token IDs → full action chunk for rollout
        actions = self.action_tokenizer.decode_full_chunk(code_token_ids)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.atleast_2d(actions).astype(np.float32)  # (chunk_size, 7)

        # Buffer remaining steps, return first
        if len(actions) > 1:
            self._action_buffer.extend(actions[1:].tolist())
        return actions[0]


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
    sweep_tokenizer_type: str = "",
    sweep_checkpoint_path: str = "",
    fsdp_checkpoint: str = "",
) -> dict:
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import print_and_save, get_log_dir
    from omegaconf import OmegaConf
    import hydra

    os.makedirs(output_dir, exist_ok=True)

    # Load model — FSDP native or HuggingFace
    if fsdp_checkpoint:
        model = CalvinFSDPModel(
            fsdp_checkpoint=fsdp_checkpoint,
            device=device,
            sweep_tokenizer_type=sweep_tokenizer_type,
            sweep_checkpoint_path=sweep_checkpoint_path,
        )
    else:
        model = CalvinOpenVLAModel(
            condition=condition,
            checkpoint_dir=checkpoint_dir,
            vqvla_checkpoint_dir=vqvla_checkpoint_dir,
            device=device,
            sweep_tokenizer_type=sweep_tokenizer_type,
            sweep_checkpoint_path=sweep_checkpoint_path,
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
    p = argparse.ArgumentParser(description="CALVIN rollout evaluation for fine-tuned OpenVLA / MiniVLA")
    p.add_argument("--condition", required=True,
                   help="Condition label (e.g. bin, vq_verb, vb_c5e16g4, quest_h16f256d2)")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Fine-tuned VLA checkpoint directory")
    p.add_argument("--vqvla_checkpoint_dir", default="",
                   help="VQ-VLA tokenizer checkpoint (for old vq_* conditions)")
    p.add_argument("--sweep_tokenizer_type", default="",
                   help="Sweep tokenizer type: vq_bet, oat, quest")
    p.add_argument("--sweep_checkpoint_path", default="",
                   help="Path to sweep tokenizer checkpoint (full.pth)")
    p.add_argument("--dataset_path", default="/data/user_data/yashagar/task_D_D",
                   help="Path to CALVIN dataset root (contains training/ and validation/)")
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "results", "rollout"))
    p.add_argument("--num_sequences", type=int, default=1000,
                   help="Number of evaluation sequences (standard=1000)")
    p.add_argument("--ep_len", type=int, default=EP_LEN,
                   help="Max environment steps per subtask")
    p.add_argument("--fsdp_checkpoint", default="",
                   help="Path to FSDP .pt checkpoint (native prismatic format, for MiniVLA scratch)")
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
        sweep_tokenizer_type=args.sweep_tokenizer_type,
        sweep_checkpoint_path=args.sweep_checkpoint_path,
        fsdp_checkpoint=args.fsdp_checkpoint,
    )


if __name__ == "__main__":
    main()
