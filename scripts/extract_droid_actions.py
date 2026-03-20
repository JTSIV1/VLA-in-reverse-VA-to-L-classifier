"""Extract action trajectories + metadata from DROID RLDS shards.

Streams through TFRecord shards on GCS via gsutil, extracts only actions
and language instructions (skipping images), and saves compact .npz files.

Usage:
    python scripts/extract_droid_actions.py --shard_start 0 --shard_end 2048

Each shard produces one .npz file containing:
    - actions_{i}: (T_i, 7) float32 action trajectory for episode i
    - lang1_{i}, lang2_{i}, lang3_{i}: language instructions (str)
    - episode_path_{i}: original file path (str)
    - n_episodes: total number of episodes in this shard
"""

import os
import argparse
import subprocess
import tempfile
import numpy as np

def extract_shard(shard_idx, total_shards, output_dir, cache_dir):
    """Download one shard, extract actions + metadata, save as .npz, delete shard."""
    shard_name = f"droid_101-train.tfrecord-{shard_idx:05d}-of-{total_shards:05d}"
    gcs_path = f"gs://gresearch/robotics/droid/1.0.1/{shard_name}"
    local_path = os.path.join(cache_dir, shard_name)
    out_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")

    if os.path.exists(out_path):
        print(f"[{shard_idx}] Already extracted, skipping.")
        return

    # Download shard
    print(f"[{shard_idx}] Downloading {shard_name}...")
    result = subprocess.run(
        ["gsutil", "-q", "cp", gcs_path, local_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"[{shard_idx}] Download failed: {result.stderr}")
        return

    # Extract actions + metadata
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(local_path)

    save_dict = {}
    ep_count = 0

    for raw in ds:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        feat = example.features.feature

        # Episode path
        ep_path = feat["episode_metadata/file_path"].bytes_list.value[0].decode()

        # Language instructions
        lang1 = feat["steps/language_instruction"].bytes_list.value[0].decode()
        lang2 = feat["steps/language_instruction_2"].bytes_list.value[0].decode()
        lang3 = feat["steps/language_instruction_3"].bytes_list.value[0].decode()

        # Actions (flattened: N_steps * 7)
        actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
        n_steps = len(feat["steps/is_first"].int64_list.value)
        action_dim = len(actions_flat) // n_steps
        actions = actions_flat.reshape(n_steps, action_dim)

        save_dict[f"actions_{ep_count}"] = actions
        save_dict[f"lang1_{ep_count}"] = lang1
        save_dict[f"lang2_{ep_count}"] = lang2
        save_dict[f"lang3_{ep_count}"] = lang3
        save_dict[f"episode_path_{ep_count}"] = ep_path

        ep_count += 1

    save_dict["n_episodes"] = ep_count
    np.savez_compressed(out_path, **save_dict)
    print(f"[{shard_idx}] Saved {ep_count} episodes to {out_path}")

    # Clean up downloaded shard
    os.remove(local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_start", type=int, default=0)
    parser.add_argument("--shard_end", type=int, default=2048)
    parser.add_argument("--total_shards", type=int, default=2048)
    parser.add_argument("--output_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/droid_actions")
    parser.add_argument("--cache_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/droid_rlds_cache")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    for shard_idx in range(args.shard_start, args.shard_end):
        try:
            extract_shard(shard_idx, args.total_shards, args.output_dir, args.cache_dir)
        except Exception as e:
            print(f"[{shard_idx}] Error: {e}")
            continue


if __name__ == "__main__":
    main()
