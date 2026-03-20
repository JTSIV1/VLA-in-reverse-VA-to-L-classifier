"""Extract action trajectories from BridgeV2 TFDS shards.

Reads tfrecord shards, extracts action vectors and metadata,
saves compact .npz files keyed by Emma-X episode_key format.

Usage (single shard):
    python scripts/extract_bridge_actions.py --shard_idx 0

Usage (SLURM array):
    See scripts/submit_bridge_extraction.sh
"""

import os
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

BRIDGE_TFDS_DIR = "/data/user_data/wenjiel2/datasets/bridge_v2"
OUTPUT_DIR = "/data/user_data/wenjiel2/datasets/bridge_actions"
TOTAL_SHARDS = 1024
ACTION_DIM = 7


def extract_shard(shard_idx):
    shard_name = f"bridge_dataset-train.tfrecord-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}"
    shard_path = os.path.join(BRIDGE_TFDS_DIR, shard_name)

    if not os.path.exists(shard_path):
        print(f"Shard {shard_path} not found, skipping")
        return

    out_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}.npz")
    if os.path.exists(out_path):
        print(f"Output {out_path} already exists, skipping")
        return

    ds = tf.data.TFRecordDataset(shard_path)
    episodes = {}
    n_episodes = 0

    for raw in ds:
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        feat = example.features.feature

        # Extract metadata
        file_path = feat["episode_metadata/file_path"].bytes_list.value[0].decode()
        episode_id = feat["episode_metadata/episode_id"].int64_list.value[0]
        episode_key = f"{file_path}|{episode_id}"

        # Extract actions (flattened 7-dim)
        actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
        n_steps = len(actions_flat) // ACTION_DIM
        actions = actions_flat.reshape(n_steps, ACTION_DIM)

        # Extract state (flattened 7-dim)
        state_flat = np.array(feat["steps/observation/state"].float_list.value, dtype=np.float32)
        state = state_flat.reshape(n_steps, ACTION_DIM)

        # Extract instruction
        instruction = feat["steps/language_instruction"].bytes_list.value[0].decode()

        episodes[f"actions_{n_episodes}"] = actions
        episodes[f"state_{n_episodes}"] = state
        episodes[f"episode_key_{n_episodes}"] = episode_key
        episodes[f"instruction_{n_episodes}"] = instruction
        episodes[f"n_steps_{n_episodes}"] = n_steps
        n_episodes += 1

    episodes["n_episodes"] = n_episodes
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(out_path, **episodes)
    print(f"Shard {shard_idx}: {n_episodes} episodes -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_idx", type=int, required=True)
    args = parser.parse_args()
    extract_shard(args.shard_idx)


if __name__ == "__main__":
    main()