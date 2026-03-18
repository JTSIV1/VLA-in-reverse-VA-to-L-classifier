"""
Annotate CALVIN episodes with hierarchical language decomposition via Gemini Pro.

Replicates the VLM2VLA pipeline (arXiv 2509.22195) adapted for CALVIN:
  L0 — Task instruction  (from CALVIN auto_lang_ann.npy)
  L1 — Subtask            (Gemini decomposition, one verb per phase)

Each Gemini call receives:
  - 6 evenly spaced RGB frames (first, last, and 4 intermediate)
  - Per-frame state log: rel_actions (7-d) and scene_obs (24-d) with dimension labels
  - The high-level task instruction (L0)

Usage:
  python scripts/annotate_calvin_hierarchy.py \
      --split training --max_episodes 100 --dry_run   # preview cost
  python scripts/annotate_calvin_hierarchy.py \
      --split training --max_episodes 5000             # run annotation

Requires:
  pip install google-genai Pillow
  export GEMINI_API_KEY=<your key>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import io

# ── CALVIN constants ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT, TRAIN_DIR, VAL_DIR,
    LANG_ANNOTATIONS_SUBDIR, LANG_ANNOTATIONS_FILE,
    IMAGE_KEY, ACTION_KEY, EPISODE_TEMPLATE,
)

CALVIN_HZ = 30
N_FRAMES = 6  # evenly spaced frames per episode

# ── Dimension descriptions for the state log ─────────────────────────────────

# rel_actions: 7-d
ACTION_DIMS = [
    "dx (lateral, + = robot's right)",
    "dy (depth, + = away from camera)",
    "dz (vertical, + = up)",
    "d_roll", "d_pitch", "d_yaw",
    "gripper (-1=open, +1=close)",
]

# scene_obs: 24-d
SCENE_OBS_DIMS = [
    "sliding_door_pos",       # 0
    "drawer_pos",             # 1
    "button_state",           # 2
    "switch_state",           # 3
    "lightbulb_state",        # 4
    "green_light_state",      # 5
    "red_block_x",            # 6
    "red_block_y",            # 7
    "red_block_z",            # 8
    "red_block_roll",         # 9
    "red_block_pitch",        # 10
    "red_block_yaw",          # 11
    "blue_block_x",           # 12
    "blue_block_y",           # 13
    "blue_block_z",           # 14
    "blue_block_roll",        # 15
    "blue_block_pitch",       # 16
    "blue_block_yaw",         # 17
    "pink_block_x",           # 18
    "pink_block_y",           # 19
    "pink_block_z",           # 20
    "pink_block_roll",        # 21
    "pink_block_pitch",       # 22
    "pink_block_yaw",         # 23
]


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are analyzing a 7-DoF robot arm (Franka Panda) performing a tabletop manipulation \
task in the CALVIN simulation environment. The robot is viewed from a fixed static camera.

You will receive:
1. 6 evenly-spaced RGB frames from the episode
2. A full state log (all timesteps at 30 Hz) with two vectors per timestep:
   - rel_actions (7-d): relative end-effector commands
     [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
     Position deltas normalized to [-1, 1] (1.0 ≈ 0.02 m). Gripper: -1=open, +1=close.
   - scene_obs (24-d): ground-truth scene state
     [0] sliding_door_pos, [1] drawer_pos, [2] button_state, [3] switch_state,
     [4] lightbulb_state, [5] green_light_state,
     [6:9] red_block (x,y,z), [9:12] red_block (roll,pitch,yaw),
     [12:15] blue_block (x,y,z), [15:18] blue_block (roll,pitch,yaw),
     [18:21] pink_block (x,y,z), [21:24] pink_block (roll,pitch,yaw)

Coordinate frame (PyBullet world frame):
- Positive x: roughly to the robot's right (camera's left)
- Positive y: roughly away from the camera (depth into scene)
- Positive z: upward

The CALVIN environment contains: a sliding door, a drawer, a button, a light switch, \
a lightbulb, and three colored blocks (red, blue, pink) on a desk.
"""

USER_PROMPT_TEMPLATE = """\
The episode has {n_steps} timesteps at 30 Hz ({duration:.1f} seconds). \
Below are {n_frames} evenly-spaced frames, followed by the full state log.

Your task:

1. **TASK_INSTRUCTION**: From the visual and state data, infer a single natural-language \
sentence describing the overall goal of this episode (e.g., "Open the drawer", \
"Slide the door to the left", "Pick up the red block and place it on the shelf"). \
Focus on what changed in the scene — check which scene_obs dimensions changed \
significantly between the first and last timestep.

2. **DECOMPOSITION**: Decompose this trajectory into sequential subtask phases. \
Each phase should represent a single semantically meaningful subtask with exactly \
ONE verb — for example "Approach the drawer", "Grasp the handle", "Pull the drawer open", \
"Release the handle", "Retract the arm".

Use diverse, specific verbs that describe the robot's intent — e.g., approach, \
reach, grasp, grip, lift, lower, push, pull, slide, rotate, twist, press, \
place, release, retract, align, insert, sweep, flip, tilt. Do NOT describe \
everything as "move [direction]" — choose the verb that best captures the \
purpose of the motion.

If a phase involves two sequential actions (e.g., approach THEN grasp), split \
it into two separate phases. Each phase = one verb = one coherent action.

NOTE: The robot may start holding an object from a previous task or need to \
reposition before the main task. Label such preliminary motions appropriately \
(e.g., "Release the block", "Reposition the arm") — they are still valid phases.

For each phase in the DECOMPOSITION, provide:
- STEP_DESCRIPTION: A short natural-language subtask label with exactly one verb.
- REASONING: Brief explanation of why you segmented here, referencing the state data.
- START_TIMESTEP: The 0-indexed start timestep (at 30 Hz) for this phase.
- END_TIMESTEP: The 0-indexed end timestep (inclusive) for this phase.

Output a JSON object with two keys:
- "TASK_INSTRUCTION": string (the inferred high-level task)
- "DECOMPOSITION": array of phase objects

The phases must be contiguous and non-overlapping, covering timestep 0 to {last_step}.

{state_log}
"""


# ── Vision-only prompts ──────────────────────────────────────────────────────

SYSTEM_PROMPT_VISION_ONLY = """\
You are analyzing a 7-DoF robot arm (Franka Panda) performing a tabletop manipulation \
task in the CALVIN simulation environment. The robot is viewed from a fixed static camera.

You will receive subsampled RGB frames from an episode. Your job is to:
1. Infer what high-level task the robot is performing (L0 task instruction).
2. Decompose the episode into sequential subtask phases (L1 subtasks).
"""

USER_PROMPT_VISION_ONLY = """\
The episode has {n_steps} timesteps at 30 Hz ({duration:.1f} seconds). \
Below are {n_frames} evenly-spaced frames.

From the visual evidence alone, determine:

1. **TASK_INSTRUCTION**: A single sentence describing the overall goal of the episode.

2. **DECOMPOSITION**: A sequence of subtask phases. Each phase should represent a single \
semantically meaningful subtask with exactly ONE verb.

Use diverse, specific verbs — e.g., approach, reach, grasp, grip, lift, lower, push, \
pull, slide, rotate, twist, press, place, release, retract, align, insert, sweep, flip, tilt.

For each phase, provide:
- STEP_DESCRIPTION: A short subtask label with exactly one verb.
- REASONING: Brief explanation of what you observe.
- START_TIMESTEP: 0-indexed start timestep (at 30 Hz).
- END_TIMESTEP: 0-indexed end timestep (inclusive).

Output a JSON object with two keys:
- "TASK_INSTRUCTION": string
- "DECOMPOSITION": array of phase objects

The phases must be contiguous and non-overlapping, covering timestep 0 to {last_step}.
"""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_annotations(data_dir):
    """Load CALVIN language annotations."""
    lang_path = os.path.join(data_dir, LANG_ANNOTATIONS_SUBDIR, LANG_ANNOTATIONS_FILE)
    lang_data = np.load(lang_path, allow_pickle=True).item()
    instructions = lang_data["language"]["ann"]
    indices = lang_data["info"]["indx"]
    return instructions, indices


def load_episode_data(data_dir, start_idx, end_idx):
    """Load actions, scene_obs, and evenly-spaced frames for one episode."""
    n_steps = end_idx - start_idx + 1

    # Compute evenly-spaced frame indices (always include first and last)
    if n_steps <= N_FRAMES:
        sample_indices = list(range(n_steps))
    else:
        sample_indices = np.linspace(0, n_steps - 1, N_FRAMES, dtype=int).tolist()

    actions = []
    scene_obs_list = []
    frames = []
    frame_local_indices = []

    for t in range(start_idx, end_idx + 1):
        ep_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(t))
        ep = np.load(ep_path)
        actions.append(ep[ACTION_KEY])
        scene_obs_list.append(ep["scene_obs"])

        local_t = t - start_idx
        if local_t in sample_indices:
            frames.append(ep[IMAGE_KEY])
            frame_local_indices.append(local_t)

    actions = np.array(actions)
    scene_obs = np.array(scene_obs_list)
    return actions, scene_obs, frames, frame_local_indices


def format_state_log_full(actions, scene_obs):
    """Format ALL timesteps with action + scene_obs (compact: one line per timestep)."""
    lines = []
    lines.append("State log for all {} timesteps (30 Hz):".format(len(actions)))
    lines.append("Format: t | rel_actions [dx,dy,dz,droll,dpitch,dyaw,gripper] | "
                 "scene_obs [door,drawer,btn,switch,bulb,green, "
                 "red_xyz,red_rpy, blue_xyz,blue_rpy, pink_xyz,pink_rpy]")
    lines.append("")

    for t in range(len(actions)):
        a = actions[t]
        s = scene_obs[t]
        a_str = ",".join("{:.3f}".format(float(v)) for v in a)
        # Compact scene_obs: group by semantic meaning
        s_fix = ",".join("{:.3f}".format(float(s[j])) for j in range(6))
        s_red = ",".join("{:.3f}".format(float(s[j])) for j in range(6, 12))
        s_blue = ",".join("{:.3f}".format(float(s[j])) for j in range(12, 18))
        s_pink = ",".join("{:.3f}".format(float(s[j])) for j in range(18, 24))
        lines.append("t={:3d} | [{}] | [{}  {}  {}  {}]".format(
            t, a_str, s_fix, s_red, s_blue, s_pink))

    return "\n".join(lines)


def encode_frame_to_bytes(frame_array):
    """Convert numpy RGB array to JPEG bytes."""
    img = Image.fromarray(frame_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def estimate_tokens(n_steps, n_frames):
    """Rough token estimate for cost projection."""
    # Full state log: ~30 tokens per timestep (compact one-liner with actions + scene_obs)
    state_tokens = n_steps * 30
    image_tokens = n_frames * 258  # Gemini charges 258 tokens per image
    prompt_tokens = 800  # system + user prompt template
    output_tokens = 600  # estimated output
    return state_tokens + image_tokens + prompt_tokens, output_tokens


# ── Gemini API ───────────────────────────────────────────────────────────────

def call_gemini(system_prompt, user_text, frame_bytes_list, frame_indices):
    """Call Gemini Pro with images and text. Returns parsed JSON."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Build content parts: interleave frame images with captions, then the text
    parts = []
    for fb, fidx in zip(frame_bytes_list, frame_indices):
        parts.append(types.Part.from_text(
            text="Frame at timestep {} (t={:.2f}s):".format(fidx, fidx / CALVIN_HZ)
        ))
        parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

    # Append the main user prompt text after all images
    parts.append(types.Part.from_text(text=user_text))

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,  # low temperature for structured output
            response_mime_type="application/json",
        ),
    )

    # Parse JSON from response
    text = response.text.strip()
    # Handle markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return json.loads(text)


# ── Episode annotation ───────────────────────────────────────────────────────

def annotate_episode(data_dir, instruction, start_idx, end_idx,
                     dry_run=False, vision_only=False):
    """Annotate a single episode. Returns the hierarchical decomposition dict."""
    actions, scene_obs, frames, frame_indices = load_episode_data(
        data_dir, start_idx, end_idx
    )
    n_steps = len(actions)
    n_frames = len(frames)

    if vision_only:
        system_prompt = SYSTEM_PROMPT_VISION_ONLY
        user_text = USER_PROMPT_VISION_ONLY.format(
            n_steps=n_steps,
            duration=n_steps / CALVIN_HZ,
            n_frames=n_frames,
            last_step=n_steps - 1,
        )
    else:
        system_prompt = SYSTEM_PROMPT
        state_log = format_state_log_full(actions, scene_obs)
        user_text = USER_PROMPT_TEMPLATE.format(
            n_steps=n_steps,
            duration=n_steps / CALVIN_HZ,
            n_frames=n_frames,
            last_step=n_steps - 1,
            state_log=state_log,
        )

    if dry_run:
        in_tok, out_tok = estimate_tokens(n_steps, n_frames)
        if vision_only:
            in_tok = n_frames * 258 + 600
        return {
            "instruction": instruction,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_steps": n_steps,
            "n_frames": n_frames,
            "est_input_tokens": in_tok,
            "est_output_tokens": out_tok,
        }

    frame_bytes = [encode_frame_to_bytes(f) for f in frames]
    result = call_gemini(system_prompt, user_text, frame_bytes, frame_indices)

    # Both modes now return TASK_INSTRUCTION + DECOMPOSITION
    return {
        "instruction_gt": instruction,
        "instruction_gemini": result.get("TASK_INSTRUCTION", ""),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "n_steps": n_steps,
        "n_frames": n_frames,
        "frame_indices": frame_indices,
        "decomposition": result.get("DECOMPOSITION", []),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate CALVIN with hierarchical language via Gemini"
    )
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Resume from episode index")
    parser.add_argument("--dry_run", action="store_true",
                        help="Estimate cost without calling API")
    parser.add_argument("--output_dir", type=str, default="data/hierarchy_annotations",
                        help="Output directory for annotations")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0,
                        help="Seconds between API calls (rate limiting)")
    parser.add_argument("--vision_only", action="store_true",
                        help="Send only frames (no actions/instruction); Gemini infers L0+L1")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Shard ID for parallel runs (appended to output filename)")
    args = parser.parse_args()

    data_dir = TRAIN_DIR if args.split == "training" else VAL_DIR
    instructions, indices = load_annotations(data_dir)

    n_total = len(instructions)
    end_idx = min(n_total, args.start_from + args.max_episodes) if args.max_episodes else n_total
    episode_range = range(args.start_from, end_idx)
    n_episodes = len(episode_range)

    print("Split: {}, Episodes: {} ({}..{} of {})".format(
        args.split, n_episodes, args.start_from, end_idx - 1, n_total))

    if args.dry_run:
        total_in, total_out = 0, 0
        for i in episode_range:
            start, end = indices[i]
            result = annotate_episode(data_dir, instructions[i], start, end, dry_run=True)
            total_in += result["est_input_tokens"]
            total_out += result["est_output_tokens"]

        # Gemini 2.5 Pro pricing: $1.25/M input, $10/M output
        cost_in = total_in / 1e6 * 1.25
        cost_out = total_out / 1e6 * 10.0
        print("\n=== Cost Estimate ===")
        print("Episodes: {}".format(n_episodes))
        print("Frames per episode: {}".format(N_FRAMES))
        print("Est input tokens:  {:,.0f} (${:.2f})".format(total_in, cost_in))
        print("Est output tokens: {:,.0f} (${:.2f})".format(total_out, cost_out))
        print("Est total cost:    ${:.2f}".format(cost_in + cost_out))
        return

    # Ensure API key is set
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    suffix = "_vision_only" if args.vision_only else ""
    shard_suffix = "_shard{}".format(args.shard_id) if args.shard_id is not None else ""
    output_path = os.path.join(
        args.output_dir, "calvin_{}{}{}.jsonl".format(args.split, suffix, shard_suffix)
    )
    print("Output: {}".format(output_path))

    n_done = 0
    n_errors = 0

    with open(output_path, "a") as f:
        for i in episode_range:
            start, end = indices[i]
            inst = instructions[i]

            try:
                result = annotate_episode(
                    data_dir, inst, start, end,
                    dry_run=False, vision_only=args.vision_only,
                )
                result["episode_index"] = int(i)
                f.write(json.dumps(result) + "\n")
                f.flush()
                n_done += 1

                n_steps_decomp = len(result.get("decomposition", []))
                gemini_inst = result.get("instruction_gemini", "")[:50]
                print("[{}/{}] ep={} GT=\"{}\" | Gemini=\"{}\" -> {} phases".format(
                    n_done, n_episodes, i, inst[:40], gemini_inst, n_steps_decomp))

            except Exception as e:
                n_errors += 1
                print("[{}/{}] ep={} ERROR: {}".format(
                    n_done + n_errors, n_episodes, i, e))
                error_path = os.path.join(
                    args.output_dir, "errors_{}.jsonl".format(args.split)
                )
                with open(error_path, "a") as ef:
                    ef.write(json.dumps({
                        "episode_index": int(i),
                        "instruction": inst,
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "error": str(e),
                    }) + "\n")

            time.sleep(args.rate_limit_delay)

    print("\nDone. {} annotated, {} errors.".format(n_done, n_errors))


if __name__ == "__main__":
    main()
