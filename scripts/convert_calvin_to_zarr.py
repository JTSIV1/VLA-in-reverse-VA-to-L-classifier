"""
Convert CALVIN task_D_D training split to zarr format for OAT/FAST/QueST tokenizer training.

Usage:
    python scripts/convert_calvin_to_zarr.py --output data/calvin_N500.zarr
    python scripts/convert_calvin_to_zarr.py --output data/calvin_Nall.zarr --max_demos None
"""

import argparse
import os
import sys
import numpy as np
import zarr
import numcodecs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_DIR, ACTION_KEY, EPISODE_TEMPLATE
from utils import load_calvin_to_dataframe


def convert(data_dir, output_path, max_demos=None):
    print(f"Loading CALVIN training annotations from {data_dir} ...")
    df = load_calvin_to_dataframe(data_dir)

    if max_demos is not None:
        df = df.head(max_demos)

    n_episodes = len(df)
    print(f"Converting {n_episodes} episodes ...")

    # Collect all action sequences
    all_actions = []
    episode_ends = []
    total_steps = 0

    for i, (_, row) in enumerate(df.iterrows()):
        s, e = int(row["start_idx"]), int(row["end_idx"])
        traj = []
        for idx in range(s, e + 1):
            path = os.path.join(data_dir, EPISODE_TEMPLATE.format(idx))
            traj.append(np.load(path)[ACTION_KEY].astype(np.float32))
        actions = np.stack(traj)  # (T, 7)
        all_actions.append(actions)
        total_steps += len(actions)
        episode_ends.append(total_steps - 1)

        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{n_episodes} episodes ...")

    print(f"Total steps: {total_steps} across {n_episodes} episodes")

    # Stack into flat arrays
    flat_actions = np.concatenate(all_actions, axis=0)  # (total_steps, 7)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    # Write to zarr
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    compressor = numcodecs.Blosc(cname="lz4", clevel=5)

    root.array(
        "data/action",
        data=flat_actions,
        chunks=(1000, 7),
        compressor=compressor,
        dtype=np.float32,
    )
    root.array(
        "meta/episode_ends",
        data=episode_ends,
        chunks=(min(1000, n_episodes),),
        compressor=compressor,
        dtype=np.int64,
    )

    print(f"Saved zarr to {output_path}")
    print(f"  data/action:       {flat_actions.shape}")
    print(f"  meta/episode_ends: {episode_ends.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/calvin_N500.zarr")
    parser.add_argument("--max_demos", type=int, default=500,
                        help="Max episodes to include (default 500, set 0 for all)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="CALVIN training dir (default: TRAIN_DIR from config.py)")
    args = parser.parse_args()

    max_demos = args.max_demos if args.max_demos > 0 else None
    convert(args.data_dir or TRAIN_DIR, args.output, max_demos=max_demos)
