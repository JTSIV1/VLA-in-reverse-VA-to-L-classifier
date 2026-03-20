"""Step 2 dry run: frame-by-frame verb annotation using Gemini Flash.

Extracts 24 evenly-spaced frames from 10 DROID episodes, sends them to
Gemini with the top-50 subtask verb dictionary, and asks it to label
what the robot is doing at each frame.

Usage:
    GEMINI_API_KEY=... python scripts/test_gemini_step2.py
"""

import google.generativeai as genai
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tfrecord_parser import read_tfrecords, parse_tf_example

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)

# Top-50 subtask verbs from decomposition analysis
VERB_DICTIONARY = [
    "move", "reach", "grasp", "lift", "release", "retract", "lower",
    "place", "withdraw", "touch", "pull", "push", "approach", "press",
    "insert", "carry", "open", "turn", "tilt", "grab", "untilt",
    "position", "pick", "close", "slide", "rotate", "retreat", "align",
    "pour", "raise", "flip", "smooth", "straighten", "drop", "shift",
    "twist", "set", "contact", "drape", "scoop", "lay", "put", "fold",
    "orient", "stir", "drag", "rub", "wipe", "hang", "spread",
]

RLDS_DIR = "/data/user_data/wenjiel2/datasets/droid_rlds"
IMAGE_KEY = "steps/observation/exterior_image_1_left"
N_FRAMES = 24

PROMPT_TEMPLATE = (
    "You are watching a robot perform a manipulation task. The task instruction is:\n"
    '"{instruction}"\n\n'
    "Below are {n_frames} frames sampled evenly across the episode (frame numbers shown).\n\n"
    "First, look at ALL frames together and describe the overall progression of the "
    "task from start to finish. Think about:\n"
    "- What is the scene? What objects are relevant?\n"
    "- What major phases does the task go through?\n"
    "- At what point does each phase transition to the next?\n"
    "- How far along is the task at each frame?\n\n"
    "Then, segment the episode into sequential subtasks. Each subtask should be a "
    "descriptive sentence that captures WHAT the robot is doing, TO WHAT object, "
    "and WHY in the context of the overall task. For example:\n"
    "- \"reach toward the bottle cap to prepare for grasping\"\n"
    "- \"lift the towel edge upward to begin the first fold\"\n"
    "- \"carry the spatula across to the utensil holder\"\n\n"
    "The leading verb of each subtask MUST come from this dictionary:\n"
    "{verb_list}\n\n"
    "Guidelines:\n"
    "- Each frame gets exactly one subtask\n"
    "- Consecutive frames performing the same action share the same subtask\n"
    "- Subtasks should be specific enough to distinguish different instances "
    "(e.g., \"fold the left edge of the towel rightward\" vs \"fold the top edge "
    "of the towel downward\")\n"
    "- Track task progress: if the task has multiple stages (e.g., fold twice), "
    "make the subtask descriptions reflect which stage the robot is in\n\n"
    "Respond in this format:\n\n"
    "TASK ANALYSIS:\n"
    "<your analysis of the overall task progression>\n\n"
    "ANNOTATIONS:\n"
    '[{{"frame": 0, "verb": "...", "subtask": "..."}}, '
    '{{"frame": 1, "verb": "...", "subtask": "..."}}]\n'
)

# Test episodes (first 10 from filtered CSV)
TEST_EPISODE_IDXS = [3, 10, 12, 15, 17, 18, 28, 29, 32, 44]

OUTPUT_PATH = "data/droid_step2_dryrun_{}.json".format(MODEL_NAME.replace("gemini-", "").replace(".", ""))


def parse_json(text):
    for ch in ["\u201c", "\u201d", "\u2018", "\u2019"]:
        text = text.replace(ch, '"')
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def build_episode_map(csv_path):
    """Map episode_idx -> row from CSV."""
    df = pd.read_csv(csv_path)
    return {row["episode_idx"]: row for _, row in df.iterrows()}


def build_shard_cumsum(actions_dir):
    """Build cumulative episode count per shard: [(shard_idx, start, end), ...]."""
    cumsum = []
    total = 0
    for i in range(2048):
        path = os.path.join(actions_dir, f"shard_{i:05d}.npz")
        if not os.path.exists(path):
            break
        d = np.load(path, allow_pickle=True)
        n = int(d["n_episodes"])
        cumsum.append((i, total, total + n))
        total += n
    return cumsum


def episode_to_shard(ep_idx, cumsum):
    """Find (shard_idx, position_within_shard) for a global episode index."""
    for shard_idx, start, end in cumsum:
        if start <= ep_idx < end:
            return shard_idx, ep_idx - start
    raise ValueError(f"Episode {ep_idx} not found in any shard")


def extract_frames(shard_idx, pos_in_shard, n_frames=N_FRAMES):
    """Extract n_frames evenly-spaced frames from an episode in an RLDS shard."""
    shard_name = f"droid_101-train.tfrecord-{shard_idx:05d}-of-02048"
    shard_path = os.path.join(RLDS_DIR, shard_name)

    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    for i, raw_record in enumerate(read_tfrecords(shard_path)):
        if i != pos_in_shard:
            continue

        feat = parse_tf_example(raw_record)
        img_bytes_list = feat[IMAGE_KEY]["bytes_list"]
        n_steps = len(img_bytes_list)

        # Evenly-spaced frame indices
        indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            jpeg_bytes = img_bytes_list[idx]
            img = Image.open(io.BytesIO(jpeg_bytes))
            frames.append(img)

        return frames, indices.tolist(), n_steps

    raise ValueError(f"Position {pos_in_shard} not found in shard {shard_idx}")


def annotate_episode(instruction, frames, frame_indices):
    """Send frames + prompt to Gemini and get per-frame verb annotations."""
    verb_list = ", ".join(VERB_DICTIONARY)
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        n_frames=len(frames),
        verb_list=verb_list,
    )

    # Build multimodal content: prompt text + interleaved frame images
    content = [prompt]
    for i, (frame, fidx) in enumerate(zip(frames, frame_indices)):
        content.append(f"\nFrame {i} (step {fidx}):")
        content.append(frame)

    resp = model.generate_content(content)
    raw = resp.text

    # Extract task analysis if present
    task_analysis = ""
    analysis_match = re.search(r"TASK ANALYSIS:\s*\n(.*?)(?=\nANNOTATIONS:|\[)", raw, re.DOTALL)
    if analysis_match:
        task_analysis = analysis_match.group(1).strip()

    return parse_json(raw), raw, task_analysis


def main():
    csv_path = "data/droid_episodes_filtered.csv"
    actions_dir = "/data/user_data/wenjiel2/datasets/droid_actions"

    print("Building episode map and shard index...")
    ep_map = build_episode_map(csv_path)
    cumsum = build_shard_cumsum(actions_dir)

    results = []
    for ep_idx in TEST_EPISODE_IDXS:
        if ep_idx not in ep_map:
            print(f"Episode {ep_idx} not in filtered CSV, skipping")
            continue

        row = ep_map[ep_idx]
        instruction = row["instruction"]
        verb = row["verb"]
        shard_idx, pos = episode_to_shard(ep_idx, cumsum)

        print(f"\n=== Episode {ep_idx}: \"{instruction}\" (verb={verb}) ===")
        print(f"  Shard {shard_idx}, position {pos}")

        try:
            frames, frame_indices, n_steps = extract_frames(shard_idx, pos)
            print(f"  Extracted {len(frames)} frames from {n_steps} total steps")

            annotations, raw_text, task_analysis = annotate_episode(instruction, frames, frame_indices)
            print(f"  Got {len(annotations)} frame annotations")

            if task_analysis:
                print(f"  Task analysis: {task_analysis[:200]}...")

            for ann in annotations:
                subtask = ann.get("subtask", ann["verb"])
                print(f"    Frame {ann['frame']} (step {frame_indices[ann['frame']]}): {subtask}")

            results.append({
                "episode_idx": int(ep_idx),
                "instruction": instruction,
                "verb": verb,
                "n_steps": int(n_steps),
                "frame_indices": frame_indices,
                "annotations": annotations,
                "task_analysis": task_analysis,
            })

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results.append({
                "episode_idx": int(ep_idx),
                "instruction": instruction,
                "verb": verb,
                "error": str(e),
            })

        time.sleep(2)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if "annotations" in r)
    print(f"\nDone: {n_ok}/{len(results)} episodes annotated successfully")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()