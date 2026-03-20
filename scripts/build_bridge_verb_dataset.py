"""Build a verb classification dataset from Emma-X segments + BridgeV2 actions.

For each segment in Emma-X GCOT, extracts the corresponding action sub-trajectory
from BridgeV2 shards and saves a compact dataset for training.

Output:
  - data/bridge_verb_segments.csv: segment metadata with verb labels
  - /data/user_data/wenjiel2/datasets/bridge_actions/segment_actions.npz:
    action sub-trajectories keyed by segment index

Usage:
    python scripts/build_bridge_verb_dataset.py
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

EMMA_CSV = "data/emma_x_segments.csv"
SHARD_DIR = "/data/user_data/wenjiel2/datasets/bridge_actions"
OUTPUT_CSV = "data/bridge_verb_segments.csv"
OUTPUT_NPZ = os.path.join(SHARD_DIR, "segment_actions.npz")
MIN_CLASS_COUNT = 30
ACTION_DIM = 7


def main():
    # Load Emma-X segments
    df = pd.read_csv(EMMA_CSV)
    print(f"Emma-X segments: {len(df)}, unique verbs: {df['verb'].nunique()}")

    # Filter to verbs with >= MIN_CLASS_COUNT
    vc = df["verb"].value_counts()
    keep_verbs = set(vc[vc >= MIN_CLASS_COUNT].index)
    df = df[df["verb"].isin(keep_verbs)].reset_index(drop=True)
    print(f"After filtering (>={MIN_CLASS_COUNT}): {len(df)} segments, "
          f"{df['verb'].nunique()} verbs")

    # Build episode key -> (shard_idx, local_idx, n_steps) index
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
    print(f"Indexing {len(shard_files)} action shards...")

    key_to_loc = {}
    for sf in tqdm(shard_files, desc="Indexing shards"):
        data = np.load(sf, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        shard_idx = int(os.path.basename(sf).split("_")[1].split(".")[0])
        for i in range(n_eps):
            key = str(data[f"episode_key_{i}"])
            n_steps = int(data[f"n_steps_{i}"])
            key_to_loc[key] = (shard_idx, i, n_steps)

    print(f"Indexed {len(key_to_loc)} episodes")

    # Extract segment action sub-trajectories
    # Cache loaded shards to avoid re-reading
    shard_cache = {}
    segment_actions = {}
    valid_rows = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting segments"):
        ep_key = row["episode_key"]
        if ep_key not in key_to_loc:
            skipped += 1
            continue

        shard_idx, local_idx, total_steps = key_to_loc[ep_key]

        # Load shard if not cached
        if shard_idx not in shard_cache:
            # Evict old cache entries to save memory (keep last 5)
            if len(shard_cache) > 5:
                oldest = list(shard_cache.keys())[0]
                del shard_cache[oldest]
            sf = os.path.join(SHARD_DIR, f"shard_{shard_idx:05d}.npz")
            shard_cache[shard_idx] = np.load(sf, allow_pickle=True)

        data = shard_cache[shard_idx]
        actions = data[f"actions_{local_idx}"]

        # Extract sub-trajectory for this segment (±1 frame context)
        start = int(row["start_frame"]) - 1
        end = int(row["end_frame"]) + 1

        # Clamp to valid range
        start = max(0, min(start, len(actions) - 1))
        end = max(start, min(end, len(actions) - 1))

        seg_actions = actions[start:end + 1]

        if len(seg_actions) == 0:
            skipped += 1
            continue

        seg_idx = len(valid_rows)
        segment_actions[f"actions_{seg_idx}"] = seg_actions.astype(np.float32)
        valid_rows.append({
            "seg_idx": seg_idx,
            "episode_key": ep_key,
            "instruction": row["instruction"],
            "verb": row["verb"],
            "segment_num": row["segment_num"],
            "start_frame": start,
            "end_frame": end,
            "seg_length": len(seg_actions),
            "total_ep_length": total_steps,
        })

    print(f"\nExtracted {len(valid_rows)} segments, skipped {skipped}")

    # Save
    out_df = pd.DataFrame(valid_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV to {OUTPUT_CSV}")

    # Stats
    print(f"\nSegment length stats:")
    print(f"  Mean: {out_df['seg_length'].mean():.1f}")
    print(f"  Median: {out_df['seg_length'].median():.0f}")
    print(f"  Max: {out_df['seg_length'].max()}")
    print(f"\nVerb distribution:")
    print(out_df["verb"].value_counts().to_string())

    # Save actions
    segment_actions["n_segments"] = len(valid_rows)
    print(f"\nSaving segment actions to {OUTPUT_NPZ}...")
    np.savez_compressed(OUTPUT_NPZ, **segment_actions)
    print(f"Done. File size: {os.path.getsize(OUTPUT_NPZ) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
