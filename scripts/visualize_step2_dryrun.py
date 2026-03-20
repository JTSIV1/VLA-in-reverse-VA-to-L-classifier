"""Visualize Step 2 dry run: show frames with verb annotations per episode."""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tfrecord_parser import read_tfrecords, parse_tf_example

RLDS_DIR = "/data/user_data/wenjiel2/datasets/droid_rlds"
IMAGE_KEY = "steps/observation/exterior_image_1_left"
DRYRUN_PATH = os.environ.get("DRYRUN_PATH", "data/droid_step2_dryrun.json")
ACTIONS_DIR = "/data/user_data/wenjiel2/datasets/droid_actions"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "figures/step2_dryrun")


def build_shard_cumsum():
    cumsum = []
    total = 0
    for i in range(2048):
        path = os.path.join(ACTIONS_DIR, f"shard_{i:05d}.npz")
        if not os.path.exists(path):
            break
        d = np.load(path, allow_pickle=True)
        n = int(d["n_episodes"])
        cumsum.append((i, total, total + n))
        total += n
    return cumsum


def episode_to_shard(ep_idx, cumsum):
    for shard_idx, start, end in cumsum:
        if start <= ep_idx < end:
            return shard_idx, ep_idx - start
    raise ValueError(f"Episode {ep_idx} not found")


def extract_frames(shard_idx, pos_in_shard, frame_indices):
    shard_name = f"droid_101-train.tfrecord-{shard_idx:05d}-of-02048"
    shard_path = os.path.join(RLDS_DIR, shard_name)

    for i, raw_record in enumerate(read_tfrecords(shard_path)):
        if i != pos_in_shard:
            continue
        feat = parse_tf_example(raw_record)
        img_bytes_list = feat[IMAGE_KEY]["bytes_list"]
        frames = []
        for idx in frame_indices:
            img = Image.open(io.BytesIO(img_bytes_list[idx]))
            frames.append(img)
        return frames
    raise ValueError(f"Position {pos_in_shard} not found in shard {shard_idx}")


def wrap_text(text, max_chars=30):
    """Wrap text to fit under a frame thumbnail."""
    import textwrap
    return "\n".join(textwrap.wrap(text, max_chars))


def plot_episode(result, frames, output_path):
    n_frames = len(frames)
    n_cols = 4
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * 3.8))
    fig.suptitle(
        'Ep {}: "{}" (high-level verb: {})'.format(
            result["episode_idx"], result["instruction"], result["verb"]
        ),
        fontsize=14, fontweight="bold", y=0.99,
    )

    # Color map: assign colors to unique verbs in order of appearance
    annotations = result["annotations"]
    frame_indices = result["frame_indices"]
    seen_verbs = []
    for ann in annotations:
        if ann["verb"] not in seen_verbs:
            seen_verbs.append(ann["verb"])
    cmap = plt.cm.get_cmap("tab20", max(len(seen_verbs), 1))
    verb_colors = {v: cmap(i) for i, v in enumerate(seen_verbs)}

    axes_flat = axes.flatten() if n_rows > 1 else (axes if n_cols > 1 else [axes])

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_frames:
            ax.imshow(frames[i])
            verb = annotations[i]["verb"]
            step = frame_indices[i]
            color = verb_colors[verb]
            subtask = annotations[i].get("subtask", verb)
            ax.set_title(wrap_text(subtask, 35), fontsize=8, fontweight="bold",
                         color=color, pad=4, linespacing=1.1)
            ax.set_xlabel("[{}]  step {}".format(verb, step), fontsize=8, fontweight="bold", color=color)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.5)
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {output_path}")


def main():
    with open(DRYRUN_PATH) as f:
        results = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cumsum = build_shard_cumsum()

    for result in results:
        if "error" in result:
            print(f"Skipping episode {result['episode_idx']} (error)")
            continue

        ep_idx = result["episode_idx"]
        frame_indices = result["frame_indices"]
        shard_idx, pos = episode_to_shard(ep_idx, cumsum)

        print(f"Episode {ep_idx}: \"{result['instruction']}\"")
        frames = extract_frames(shard_idx, pos, frame_indices)

        output_path = os.path.join(OUTPUT_DIR, f"ep_{ep_idx:05d}.png")
        plot_episode(result, frames, output_path)


if __name__ == "__main__":
    main()
