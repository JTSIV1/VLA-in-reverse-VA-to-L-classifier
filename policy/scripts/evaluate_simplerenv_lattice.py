#!/usr/bin/env python3
"""Closed-loop SimplerEnv smoke eval for the Bridge LATTiCE MiniVLA policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as C  # noqa: E402


OPENVLA_DIR = Path(C.OPENVLA_DIR)
if str(OPENVLA_DIR) not in sys.path:
    sys.path.insert(0, str(OPENVLA_DIR))

DEFAULT_POLICY_CKPT = Path(
    "/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/"
    "checkpoints/bridge_sweep/policy/oat_16_855_4/"
    "vlm_clip0.1_pre_fsq_fullproj/checkpoints/"
    "step-050000-epoch-00-loss=0.2729.pt"
)
DEFAULT_TOKENIZER_CKPT = Path(
    "/data/user_data/wenjiel2/Code/VLA-in-reverse-VA-to-L-classifier/"
    "checkpoints/bridge_sweep/tokenizers/oat_16_855_4/"
    "vlm_clip0.1_pre_fsq/full.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "simplerenv" / "lattice_oat_clip_pfsq_smoke"
DEFAULT_TASKS = (
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
)
DEFAULT_SEEDS = (0, 1, 2)


def jsonable(value: Any) -> Any:
    """Convert numpy/torch-ish values into JSON serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "p") and hasattr(value, "q"):
        return {
            "type": value.__class__.__name__,
            "p": jsonable(getattr(value, "p")),
            "q": jsonable(getattr(value, "q")),
        }
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def action_stats(actions: Sequence[np.ndarray]) -> Dict[str, Any]:
    if not actions:
        return {"count": 0}
    arr = np.asarray(actions, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "min": arr.min(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "abs_max": np.abs(arr).max(axis=0).tolist(),
    }


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def import_simpler_env():
    try:
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    except Exception as exc:  # pragma: no cover - depends on external install
        raise RuntimeError(
            "Could not import simpler_env. Install SimplerEnv in this environment, "
            "or run scripts/submit_simplerenv_lattice_eval.sh which can install it "
            "from SIMPLERENV_REPO into SIMPLERENV_DIR."
        ) from exc
    return simpler_env, get_image_from_maniskill2_obs_dict


def load_native_vla(policy_ckpt: Path, device: torch.device):
    from prismatic.models import load_vla

    print(f"[load] policy checkpoint: {policy_ckpt}")
    vla = load_vla(policy_ckpt, load_for_training=False)
    vla = vla.to(device)
    vla.eval()
    return vla


class LatticeBridgePolicy:
    """Prismatic native checkpoint wrapper that emits full OAT action chunks."""

    def __init__(
        self,
        policy_ckpt: Path,
        tokenizer_ckpt: Path,
        device: torch.device,
        unnorm_key: str,
        action_unnorm_mode: str,
        image_preprocess: str,
        require_wrapper: bool = True,
    ) -> None:
        self.device = device
        self.unnorm_key = unnorm_key
        self.action_unnorm_mode = action_unnorm_mode
        self.image_preprocess = image_preprocess
        self.vla = load_native_vla(policy_ckpt, device)
        # Newer transformers GenerationMixin versions expect these cache
        # capability flags on the model. The native Prismatic/OpenVLA class in
        # this checkout predates them, and this eval only needs short greedy
        # action-token generation, so disable cache support explicitly.
        if not hasattr(self.vla, "_supports_cache_class"):
            self.vla._supports_cache_class = False
        if not hasattr(self.vla, "_supports_static_cache"):
            self.vla._supports_static_cache = False
        self.tokenizer = self.vla.llm_backbone.tokenizer
        self.image_transform = self.vla.vision_backbone.get_image_transform()
        self.action_tokenizer = self.vla.action_tokenizer
        self.n_action_tokens = int(getattr(self.action_tokenizer, "n_codes_per_chunk", 0))
        self.chunk_size = int(getattr(self.action_tokenizer, "chunk_size", 1))

        if self.n_action_tokens <= 0:
            raise RuntimeError("Loaded action tokenizer does not expose n_codes_per_chunk.")

        self.embedding = self.vla.llm_backbone.llm.get_input_embeddings()
        self.embedding_class = self.embedding.__class__.__name__
        self.has_action_wrapper = self.embedding_class == "ActionEmbeddingWrapper" or hasattr(self.embedding, "proj")
        self.uses_dynamic_wrapper = bool(getattr(self.embedding, "dynamic", False))
        self.dynamic_latent_dim = int(getattr(getattr(self.embedding, "proj", None), "in_features", 0))
        if require_wrapper and not self.has_action_wrapper:
            raise RuntimeError(
                "Expected ActionEmbeddingWrapper after load, got "
                f"{self.embedding_class}. This usually means wrapper reconstruction failed."
            )
        if self.uses_dynamic_wrapper:
            print(
                "[validate] dynamic ActionEmbeddingWrapper detected; closed-loop "
                "generation will refresh pre-FSQ latents from decoded/generated token prefixes."
            )

        self.tokenizer_ckpt = tokenizer_ckpt
        self.buffer: List[np.ndarray] = []
        self.last_generation: Dict[str, Any] = {}

    def reset(self) -> None:
        self.buffer.clear()
        self.last_generation = {}

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if self.image_preprocess == "none":
            return image
        if self.image_preprocess != "simpler_bridge":
            raise ValueError(f"Unknown image_preprocess: {self.image_preprocess}")

        # Match OpenVLA's SimplerEnv Bridge path: JPEG round-trip, resize to
        # Bridge base size 128, then to the model's 224px input size.
        import io

        pil = Image.fromarray(image).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        pil = pil.resize((128, 128), resample=Image.Resampling.LANCZOS)
        pil = pil.resize((224, 224), resample=Image.Resampling.LANCZOS)
        return np.asarray(pil, dtype=np.uint8)

    def _build_inputs(self, image: np.ndarray, instruction: str) -> Tuple[torch.Tensor, Any]:
        image = self._preprocess_image(image)
        pil = Image.fromarray(image).convert("RGB")
        prompt_builder = self.vla.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()
        input_ids = self.tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)

        pixel_values = self.image_transform(pil)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)}")
        return input_ids, pixel_values

    def _maybe_stats_unnorm(self, actions: np.ndarray) -> np.ndarray:
        if self.action_unnorm_mode == "none":
            return actions
        if self.action_unnorm_mode != "stats":
            raise ValueError(f"Unknown action_unnorm_mode: {self.action_unnorm_mode}")

        stats = self.vla.get_action_stats(self.unnorm_key)
        mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        return np.where(mask, 0.5 * (actions + 1.0) * (high - low) + low, actions)

    def _local_codes_from_token_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        token_ids_np = np.asarray(token_ids, dtype=np.int64)
        codes = self.action_tokenizer.tokenizer_len - 1 - token_ids_np
        return np.clip(codes, 0, self.action_tokenizer.n_bins - 1).astype(np.int64)

    @torch.no_grad()
    def _prefix_latents_from_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Decode a generated action-token prefix, then re-encode to pre-FSQ latents.

        OAT supports partial detokenization by padding shorter token sequences and
        decoding with eval_keep_k equal to the prefix length. This gives a
        self-consistent approximate action chunk for the prefix we have generated
        so far; encoding that chunk yields the 256-d dynamic TiCE latents needed
        when those prefix tokens are fed back into the LLM context.
        """
        if not token_ids:
            return torch.zeros(
                1,
                self.n_action_tokens,
                self.dynamic_latent_dim,
                dtype=torch.float32,
                device=self.device,
            )

        codes = self._local_codes_from_token_ids(token_ids)
        codes_t = torch.as_tensor(codes[None, :], dtype=torch.long, device=torch.device("cpu"))
        approx_actions = self.action_tokenizer.model.detokenize(codes_t)
        if isinstance(approx_actions, np.ndarray):
            approx_actions_t = torch.from_numpy(approx_actions).float()
        else:
            approx_actions_t = approx_actions.detach().float().cpu()
        latents = self.action_tokenizer.encode_pre_fsq(approx_actions_t)
        return latents.to(self.device)

    @torch.inference_mode()
    def _manual_greedy_action_tokens(self, input_ids: torch.Tensor, pixel_values: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        generated: List[int] = []
        cur_input_ids = input_ids
        lo = int(getattr(self.embedding, "action_lo", self.action_tokenizer.tokenizer_len - self.action_tokenizer.n_bins))
        hi = int(getattr(self.embedding, "action_hi", self.action_tokenizer.tokenizer_len))
        prefix_refreshes = 0

        autocast_dtype = self.vla.llm_backbone.half_precision_dtype
        use_cuda_autocast = self.device.type == "cuda" and bool(self.vla.enable_mixed_precision_training)

        for _ in range(self.n_action_tokens):
            if self.uses_dynamic_wrapper:
                latents = self._prefix_latents_from_token_ids(generated)
                self.embedding.set_current_latents(latents)
                prefix_refreshes += 1

            attention_mask = torch.ones_like(cur_input_ids, dtype=torch.long, device=self.device)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_cuda_autocast):
                output = self.vla(
                    input_ids=cur_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=None,
                    use_cache=False,
                    return_dict=True,
                )

            next_logits = output.logits[:, -1, :]
            action_logits = next_logits[:, lo:hi]
            next_token = int(action_logits.argmax(dim=-1).item() + lo)
            generated.append(next_token)
            next_token_t = torch.tensor([[next_token]], dtype=cur_input_ids.dtype, device=self.device)
            cur_input_ids = torch.cat([cur_input_ids, next_token_t], dim=1)

        generation_mode = (
            "manual_greedy_prefix_reencode" if self.uses_dynamic_wrapper else "manual_greedy"
        )
        return np.asarray(generated, dtype=np.int64), {
            "generation_mode": generation_mode,
            "action_token_lo": lo,
            "action_token_hi": hi,
            "prefix_latent_refreshes": prefix_refreshes,
        }

    @torch.inference_mode()
    def generate_chunk(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        input_ids, pixel_values = self._build_inputs(image, instruction)
        token_ids, generation_meta = self._manual_greedy_action_tokens(input_ids, pixel_values)
        decoded = self.action_tokenizer.decode_full_chunk(token_ids)
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.detach().cpu().numpy()
        decoded = np.atleast_2d(decoded).astype(np.float32)
        actions = self._maybe_stats_unnorm(decoded).astype(np.float32)
        meta = {
            "generated_token_count": int(len(token_ids)),
            "generated_token_ids": token_ids.tolist(),
            "decoded_shape": list(decoded.shape),
            "raw_decoded_stats": action_stats([a for a in decoded]),
            "unnorm_mode": self.action_unnorm_mode,
            **generation_meta,
        }
        self.last_generation = meta
        return actions, meta

    def step(self, image: np.ndarray, instruction: str) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        if self.buffer:
            return self.buffer.pop(0), {}, False

        actions, meta = self.generate_chunk(image, instruction)
        if len(actions) > 1:
            self.buffer.extend([a.copy() for a in actions[1:]])
        return actions[0], meta, True

    def dry_validate(self) -> Dict[str, Any]:
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        actions, meta = self.generate_chunk(black, "put the object on the towel")
        if meta["generated_token_count"] != self.n_action_tokens:
            raise RuntimeError(
                f"Generated {meta['generated_token_count']} action tokens, expected {self.n_action_tokens}."
            )
        if tuple(actions.shape) != (self.chunk_size, 7):
            raise RuntimeError(f"Decoded chunk shape {actions.shape}, expected {(self.chunk_size, 7)}.")
        return {
            "embedding_class": self.embedding_class,
            "has_action_wrapper": self.has_action_wrapper,
            "uses_dynamic_wrapper": self.uses_dynamic_wrapper,
            "dynamic_latent_dim": self.dynamic_latent_dim,
            "tokenizer_class": self.action_tokenizer.__class__.__name__,
            "tokenizer_type": getattr(self.action_tokenizer, "tokenizer_type", None),
            "n_action_tokens": self.n_action_tokens,
            "chunk_size": self.chunk_size,
            "image_preprocess": self.image_preprocess,
            "tokenizer_checkpoint": str(self.tokenizer_ckpt),
            "tokenizer_checkpoint_resolved": str(self.tokenizer_ckpt.resolve()),
            "dry_generation": meta,
        }


def reset_env(env: Any, seed: int) -> Tuple[Any, Dict[str, Any]]:
    try:
        out = env.reset(seed=seed)
        seed_supported = True
    except TypeError:
        out = env.reset()
        seed_supported = False

    if isinstance(out, tuple) and len(out) == 2:
        obs, reset_info = out
    else:
        obs, reset_info = out, {}
    reset_info = dict(reset_info or {})
    reset_info["seed_requested"] = seed
    reset_info["seed_supported"] = seed_supported
    return obs, reset_info


def env_attr(env: Any, name: str, default: Any = None) -> Any:
    if hasattr(env, name):
        return getattr(env, name)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, name):
        return getattr(unwrapped, name)
    return default


def get_env_instruction(env: Any) -> str:
    fn = env_attr(env, "get_language_instruction")
    if not callable(fn):
        raise AttributeError("Environment does not expose get_language_instruction on env or env.unwrapped.")
    return str(fn())


def get_env_is_final_subtask(env: Any) -> bool:
    fn = env_attr(env, "is_final_subtask")
    return bool(fn()) if callable(fn) else True


def clip_action(action: np.ndarray, action_space: Any) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return action
    return np.clip(action, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))


def transform_action(
    action: np.ndarray,
    action_scale: float,
    gripper_mode: str,
    rotation_mode: str,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.size >= 6:
        action[:6] *= float(action_scale)
        if rotation_mode == "euler":
            pass
        elif rotation_mode == "axis_angle":
            from transforms3d.euler import euler2axangle

            axis, angle = euler2axangle(float(action[3]), float(action[4]), float(action[5]))
            action[3:6] = np.asarray(axis, dtype=np.float32) * float(angle)
        else:
            raise ValueError(f"Unknown rotation_mode: {rotation_mode}")

    if action.size >= 7:
        if gripper_mode == "identity":
            pass
        elif gripper_mode == "flip":
            action[6] = -action[6]
        elif gripper_mode == "normalize01":
            action[6] = np.sign(2.0 * action[6] - 1.0)
        elif gripper_mode == "normalize01_flip":
            action[6] = -np.sign(2.0 * action[6] - 1.0)
        elif gripper_mode == "open":
            action[6] = 1.0
        elif gripper_mode == "close":
            action[6] = -1.0
        elif gripper_mode == "zero":
            action[6] = 0.0
        else:
            raise ValueError(f"Unknown gripper_mode: {gripper_mode}")
    return action


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> Optional[str]:
    if not frames:
        return None
    try:
        import mediapy as media

        path.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(path), list(frames), fps=fps)
        return str(path)
    except Exception as exc:  # pragma: no cover - optional dependency/rendering
        print(f"[video] skipped {path}: {exc}")
        return None


def run_episode(
    env: Any,
    policy: LatticeBridgePolicy,
    get_image_from_obs: Any,
    task: str,
    episode_id: int,
    seed: int,
    max_steps: int,
    video_dir: Path,
    save_video: bool,
    video_fps: int,
    verbose: bool,
    action_scale: float,
    gripper_mode: str,
    rotation_mode: str,
) -> Dict[str, Any]:
    policy.reset()
    obs, reset_info = reset_env(env, seed)
    image_env = getattr(env, "unwrapped", env)
    instruction = get_env_instruction(env)
    is_final_subtask = get_env_is_final_subtask(env)

    frames: List[np.ndarray] = []
    raw_actions: List[np.ndarray] = []
    transformed_actions: List[np.ndarray] = []
    clipped_actions: List[np.ndarray] = []
    chunk_metas: List[Dict[str, Any]] = []
    rewards: List[float] = []
    info: Dict[str, Any] = {}
    success = False
    truncated = False

    start = time.time()
    for step_idx in range(max_steps):
        image = get_image_from_obs(image_env, obs)
        frames.append(image)
        action, chunk_meta, generated_new_chunk = policy.step(image, instruction)
        transformed = transform_action(action, action_scale, gripper_mode, rotation_mode)
        clipped = clip_action(transformed, env.action_space)
        raw_actions.append(np.asarray(action, dtype=np.float32))
        transformed_actions.append(np.asarray(transformed, dtype=np.float32))
        clipped_actions.append(np.asarray(clipped, dtype=np.float32))
        if chunk_meta:
            chunk_metas.append(chunk_meta)

        out = env.step(clipped)
        if len(out) == 5:
            obs, reward, done_or_success, truncated, info = out
            success = bool(done_or_success)
        elif len(out) == 4:
            obs, reward, done_or_success, info = out
            success = bool(done_or_success)
            truncated = False
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(out)}")

        rewards.append(float(reward))
        if verbose or (episode_id == 0 and step_idx < 3 and generated_new_chunk):
            print(
                f"[rollout] task={task} ep={episode_id} step={step_idx} "
                f"success={success} truncated={bool(truncated)} reward={float(reward):.4f}"
            )

        new_instruction = get_env_instruction(env)
        if new_instruction != instruction:
            instruction = new_instruction
            is_final_subtask = get_env_is_final_subtask(env)
            policy.reset()
            print(f"[rollout] new instruction: {instruction}")

        if success or truncated:
            break

    elapsed = time.time() - start
    final_image = get_image_from_obs(image_env, obs)
    frames.append(final_image)

    video_path = None
    if save_video:
        suffix = "success" if success else "fail"
        video_path = write_video(
            video_dir / task / f"episode_{episode_id:03d}_seed_{seed}_{suffix}.mp4",
            frames,
            fps=video_fps,
        )

    return {
        "task": task,
        "episode_id": episode_id,
        "seed": seed,
        "instruction": instruction,
        "is_final_subtask": is_final_subtask,
        "success": bool(success),
        "truncated": bool(truncated),
        "steps": int(len(rewards)),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "final_reward": float(rewards[-1]) if rewards else 0.0,
        "elapsed_sec": elapsed,
        "reset_info": jsonable(reset_info),
        "episode_stats": jsonable((info or {}).get("episode_stats", {})),
        "final_info": jsonable(info or {}),
        "action_transform": {
            "action_scale": float(action_scale),
            "gripper_mode": gripper_mode,
            "rotation_mode": rotation_mode,
        },
        "raw_action_stats": action_stats(raw_actions),
        "transformed_action_stats": action_stats(transformed_actions),
        "clipped_action_stats": action_stats(clipped_actions),
        "chunk_generations": len(chunk_metas),
        "first_chunk_meta": jsonable(chunk_metas[0] if chunk_metas else {}),
        "video_path": video_path,
    }


def summarize(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[rec["task"]].append(rec)

    rows: List[Dict[str, Any]] = []
    for task, task_records in sorted(by_task.items()):
        successes = [float(r["success"]) for r in task_records]
        rows.append(
            {
                "task": task,
                "episodes": len(task_records),
                "successes": int(sum(successes)),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "mean_steps": float(np.mean([r["steps"] for r in task_records])),
                "truncation_rate": float(np.mean([float(r["truncated"]) for r in task_records])),
                "mean_total_reward": float(np.mean([r["total_reward"] for r in task_records])),
            }
        )

    all_successes = [float(r["success"]) for r in records]
    overall = {
        "episodes": len(records),
        "successes": int(sum(all_successes)),
        "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in records])) if records else 0.0,
        "truncation_rate": float(np.mean([float(r["truncated"]) for r in records])) if records else 0.0,
        "mean_total_reward": float(np.mean([r["total_reward"] for r in records])) if records else 0.0,
        "by_task": rows,
    }
    return overall, rows


def write_outputs(output_dir: Path, records: Sequence[Dict[str, Any]], validation: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "episodes.jsonl"
    with records_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(jsonable(rec), sort_keys=True) + "\n")

    summary, rows = summarize(records)
    summary["validation"] = validation
    with (output_dir / "summary.json").open("w") as f:
        json.dump(jsonable(summary), f, indent=2, sort_keys=True)

    with (output_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = [
            "task",
            "episodes",
            "successes",
            "success_rate",
            "mean_steps",
            "truncation_rate",
            "mean_total_reward",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[output] wrote {records_path}")
    print(f"[output] wrote {output_dir / 'summary.json'}")
    print(f"[output] wrote {output_dir / 'summary.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_checkpoint", type=Path, default=DEFAULT_POLICY_CKPT)
    parser.add_argument("--tokenizer_checkpoint", type=Path, default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", type=parse_csv_list, default=list(DEFAULT_TASKS))
    parser.add_argument("--episodes_per_task", type=int, default=3)
    parser.add_argument("--seeds", type=parse_int_csv, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--unnorm_key", default="bridge_dataset")
    parser.add_argument(
        "--action_unnorm_mode",
        default="none",
        choices=["none", "stats"],
        help="Use 'none' for sweep tokenizers that already detokenize to raw actions; "
        "'stats' applies OpenVLA q01/q99 unnormalization.",
    )
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=5)
    parser.add_argument("--dry_load_only", action="store_true")
    parser.add_argument("--skip_generation_check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_require_wrapper", action="store_true")
    parser.add_argument(
        "--image_preprocess",
        default="none",
        choices=["none", "simpler_bridge"],
        help="Optional pre-model image preprocessing. 'simpler_bridge' matches OpenVLA SimplerEnv Bridge eval.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Multiply xyz/rpy action dimensions by this factor before env.step.",
    )
    parser.add_argument(
        "--gripper_mode",
        default="identity",
        choices=["identity", "flip", "normalize01", "normalize01_flip", "open", "close", "zero"],
        help="Transform gripper command before env.step; useful for action API diagnostics.",
    )
    parser.add_argument(
        "--rotation_mode",
        default="euler",
        choices=["euler", "axis_angle"],
        help="Transform action[3:6] before env.step. OpenVLA SimplerEnv reference uses axis_angle.",
    )
    return parser


def select_device(requested: str, allow_cpu: bool) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this shell. Run on a GPU node or pass --allow_cpu "
            "for dry/debug checks only."
        )
    return device


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_arg_parser().parse_args()

    if not args.policy_checkpoint.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_checkpoint}")
    if not args.tokenizer_checkpoint.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {args.tokenizer_checkpoint}")

    device = select_device(args.device, args.allow_cpu)
    print(f"[setup] device: {device}")
    print(f"[setup] tokenizer resolved: {args.tokenizer_checkpoint.resolve()}")

    policy = LatticeBridgePolicy(
        args.policy_checkpoint,
        args.tokenizer_checkpoint,
        device=device,
        unnorm_key=args.unnorm_key,
        action_unnorm_mode=args.action_unnorm_mode,
        image_preprocess=args.image_preprocess,
        require_wrapper=not args.no_require_wrapper,
    )

    validation = {
        "policy_checkpoint": str(args.policy_checkpoint),
        "tokenizer_checkpoint": str(args.tokenizer_checkpoint),
        "tokenizer_checkpoint_resolved": str(args.tokenizer_checkpoint.resolve()),
        "image_preprocess": args.image_preprocess,
        "action_transform": {
            "action_scale": float(args.action_scale),
            "gripper_mode": args.gripper_mode,
            "rotation_mode": args.rotation_mode,
        },
    }
    if args.skip_generation_check:
        validation["generation_check_skipped"] = True
    else:
        validation.update(policy.dry_validate())
        print("[validate] dry generation passed:")
        print(json.dumps(jsonable(validation), indent=2, sort_keys=True))

    if args.dry_load_only:
        write_outputs(args.output_dir, [], validation)
        return 0

    simpler_env, get_image_from_obs = import_simpler_env()
    print(f"[setup] simpler_env imported from {getattr(simpler_env, '__file__', None)}")

    records: List[Dict[str, Any]] = []
    for task in args.tasks:
        print(f"[task] {task}")
        env = simpler_env.make(task)
        seeds = args.seeds or list(DEFAULT_SEEDS)
        for episode_id in range(args.episodes_per_task):
            seed = seeds[episode_id % len(seeds)]
            record = run_episode(
                env=env,
                policy=policy,
                get_image_from_obs=get_image_from_obs,
                task=task,
                episode_id=episode_id,
                seed=seed,
                max_steps=args.max_steps,
                video_dir=args.output_dir / "videos",
                save_video=not args.no_video,
                video_fps=args.video_fps,
                verbose=args.verbose,
                action_scale=args.action_scale,
                gripper_mode=args.gripper_mode,
                rotation_mode=args.rotation_mode,
            )
            records.append(record)
            write_outputs(args.output_dir, records, validation)
            print(
                f"[episode] task={task} ep={episode_id} seed={seed} "
                f"success={record['success']} steps={record['steps']} "
                f"total_reward={record['total_reward']:.4f}"
            )

        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary, rows = summarize(records)
    print("[summary]")
    for row in rows:
        print(
            f"  {row['task']}: {row['successes']}/{row['episodes']} "
            f"success_rate={row['success_rate']:.3f}"
        )
    print(
        f"  overall: {summary['successes']}/{summary['episodes']} "
        f"success_rate={summary['success_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
