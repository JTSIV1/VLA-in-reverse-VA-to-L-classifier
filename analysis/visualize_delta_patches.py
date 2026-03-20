"""Visualize state changes between first/last frames for DROID episodes.

Shows first frame, last frame, and amplified pixel difference to assess
whether goal state change is visually captured. Focuses on confusable
verb pairs (open/close, turn on/turn off).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Config ---
FRAMES_DIR = "/data/user_data/wenjiel2/datasets/droid_frames"
CSV_PATH = "data/droid_episodes_filtered.csv"
DISPLAY_SIZE = 320
# Verb pairs: confusable opposites + other interesting verbs
TARGET_VERBS = ["open", "close", "turn on", "turn off", "push", "pull",
                "pick up", "place", "slide", "flip", "pour", "wipe"]
SAMPLES_PER_VERB = 3


def build_frames_index(frames_dir):
    shard_files = sorted(glob.glob(os.path.join(frames_dir, "frames_*.npz")))
    index = {}
    global_idx = 0
    for sf in tqdm(shard_files, desc="Indexing frames"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        for i in range(n_eps):
            index[global_idx] = (sf, i)
            global_idx += 1
    print(f"Indexed {global_idx} episodes")
    return index


def decode_image(jpeg_bytes, size=None):
    if isinstance(jpeg_bytes, np.void):
        jpeg_bytes = bytes(jpeg_bytes)
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    # Load CSV and filter
    df = pd.read_csv(CSV_PATH)
    vc = df["verb"].value_counts()
    keep = set(vc[vc >= 30].index)
    df = df[df["verb"].isin(keep)].reset_index(drop=True)

    # Build frames index
    frames_index = build_frames_index(FRAMES_DIR)
    df = df[df["episode_idx"].isin(frames_index)].reset_index(drop=True)

    # Use val split for consistency
    _, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["verb"])
    val_df = val_df.reset_index(drop=True)

    # Sample episodes per verb
    available_verbs = [v for v in TARGET_VERBS if v in val_df["verb"].values]
    print(f"Available target verbs: {available_verbs}")

    shard_cache = {}
    results = []
    for verb in available_verbs:
        verb_df = val_df[val_df["verb"] == verb]
        n = min(SAMPLES_PER_VERB, len(verb_df))
        sampled = verb_df.sample(n=n, random_state=42)
        for _, row in sampled.iterrows():
            ep_idx = row["episode_idx"]
            shard_path, local_idx = frames_index[ep_idx]
            if shard_path not in shard_cache:
                shard_cache[shard_path] = np.load(shard_path, allow_pickle=True)
            shard_data = shard_cache[shard_path]

            first_pil = decode_image(shard_data[f"first_frame_{local_idx}"], DISPLAY_SIZE)
            last_pil = decode_image(shard_data[f"last_frame_{local_idx}"], DISPLAY_SIZE)

            # Pixel-level diff
            first_arr = np.array(first_pil).astype(np.float32)
            last_arr = np.array(last_pil).astype(np.float32)
            diff_rgb = np.abs(last_arr - first_arr)

            results.append({
                "verb": verb,
                "instruction": row.get("instruction", ""),
                "first": first_pil,
                "last": last_pil,
                "diff_rgb": diff_rgb,
            })

    print(f"Total samples: {len(results)}")

    # --- Plot ---
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        axes[i, 0].imshow(res["first"])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(res["last"])
        axes[i, 1].axis("off")

        # Amplified pixel diff (normalize to 95th percentile for contrast)
        diff_display = res["diff_rgb"].copy()
        p95 = np.percentile(diff_display, 95)
        if p95 > 0:
            diff_display = np.clip(diff_display / p95, 0, 1)
        else:
            diff_display = diff_display / 255.0
        axes[i, 2].imshow(diff_display)
        axes[i, 2].axis("off")

        # Row label
        instr = res["instruction"]
        if len(instr) > 55:
            instr = instr[:52] + "..."
        axes[i, 0].set_ylabel(
            "{}\n{}".format(res["verb"], instr),
            fontsize=11, fontweight="bold", rotation=0,
            labelpad=160, va="center", ha="right")

        if i == 0:
            axes[i, 0].set_title("First Frame", fontsize=16, fontweight="bold")
            axes[i, 1].set_title("Last Frame", fontsize=16, fontweight="bold")
            axes[i, 2].set_title("Pixel Difference (amplified)", fontsize=16, fontweight="bold")

    fig.suptitle("DROID: First vs Last Frame State Changes",
                 fontsize=22, fontweight="bold", y=1.005)
    plt.tight_layout(h_pad=0.3)
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/droid_state_changes.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
