"""Extract first/last frame images from DROID RLDS shards.

Reads locally-stored TFRecord shards, extracts only the first and last
frame from exterior_image_1_left for each episode, and saves as .npz files.

Uses pure-protobuf parsing (no tensorflow dependency).
Same shard indexing as extract_droid_actions.py for consistent episode ordering.

Usage:
    python scripts/extract_droid_frames.py --shard_start 0 --shard_end 2048

Each shard produces one .npz file containing:
    - first_frame_{i}: JPEG bytes of the first frame
    - last_frame_{i}: JPEG bytes of the last frame
    - n_episodes: total number of episodes in this shard
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tfrecord_parser import read_tfrecords, parse_tf_example

RLDS_DIR = "/data/user_data/wenjiel2/datasets/droid_rlds"
IMAGE_KEY = "steps/observation/exterior_image_1_left"


def extract_shard(shard_idx, total_shards, output_dir):
    """Read one local shard, extract first/last frames, save as .npz."""
    shard_name = f"droid_101-train.tfrecord-{shard_idx:05d}-of-{total_shards:05d}"
    local_path = os.path.join(RLDS_DIR, shard_name)
    out_path = os.path.join(output_dir, f"frames_{shard_idx:05d}.npz")

    if os.path.exists(out_path):
        print(f"[{shard_idx}] Already extracted, skipping.")
        return

    if not os.path.exists(local_path):
        print(f"[{shard_idx}] Shard not found at {local_path}, skipping.")
        return

    save_dict = {}
    ep_count = 0

    for raw_record in read_tfrecords(local_path):
        feat = parse_tf_example(raw_record)

        img_bytes_list = feat[IMAGE_KEY]["bytes_list"]
        n_steps = len(img_bytes_list)

        first_frame = img_bytes_list[0]
        last_frame = img_bytes_list[n_steps - 1]

        save_dict[f"first_frame_{ep_count}"] = np.void(first_frame)
        save_dict[f"last_frame_{ep_count}"] = np.void(last_frame)

        ep_count += 1

    save_dict["n_episodes"] = ep_count
    np.savez_compressed(out_path, **save_dict)
    print(f"[{shard_idx}] Saved {ep_count} episodes ({ep_count * 2} frames) to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_start", type=int, default=0)
    parser.add_argument("--shard_end", type=int, default=2048)
    parser.add_argument("--total_shards", type=int, default=2048)
    parser.add_argument("--output_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/droid_frames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for shard_idx in range(args.shard_start, args.shard_end):
        try:
            extract_shard(shard_idx, args.total_shards, args.output_dir)
        except Exception as e:
            print(f"[{shard_idx}] Error: {e}")
            continue


if __name__ == "__main__":
    main()
