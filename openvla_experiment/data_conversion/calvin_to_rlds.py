"""Convert CALVIN split D to RLDS (TFRecord) format for OpenVLA-mini.

Reads per-timestep .npz files + language annotations from CALVIN task_D_D
and writes RLDS-compatible TFRecord shards that OpenVLA-mini's data pipeline
can consume.

Usage:
    python -m openvla_experiment.data_conversion.calvin_to_rlds \
        --output_dir /data/user_data/wenjiel2/datasets/calvin_rlds

Output structure:
    <output_dir>/calvin_dataset/1.0.0/
        calvin_dataset-train.tfrecord-00000-of-NNNNN
        ...
        calvin_dataset-val.tfrecord-00000-of-NNNNN
        ...
        dataset_info.json
        features.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Project root imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: tensorflow is required. Install with: pip install tensorflow")
    sys.exit(1)


def load_annotations(split_dir):
    """Load CALVIN language annotations for a split.

    Returns list of dicts: {instruction, start_idx, end_idx}
    """
    ann_path = os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy")
    ann = np.load(ann_path, allow_pickle=True).item()

    instructions = ann['language']['ann']
    indices = ann['info']['indx']  # list of (start_idx, end_idx) tuples

    episodes = []
    for instruction, (start, end) in zip(instructions, indices):
        episodes.append({
            'instruction': instruction,
            'start_idx': int(start),
            'end_idx': int(end),
        })
    return episodes


def episode_to_tf_example(episode, data_dir):
    """Convert a CALVIN episode to a serialized tf.train.Example.

    Each episode becomes one RLDS example containing a sequence of steps.
    """
    steps = []
    for idx in range(episode['start_idx'], episode['end_idx'] + 1):
        path = os.path.join(data_dir, "episode_{:07d}.npz".format(idx))
        if not os.path.exists(path):
            continue
        data = np.load(path)

        step = {
            'rgb_static': data['rgb_static'],           # (200, 200, 3) uint8
            'rgb_gripper': data['rgb_gripper'],          # (84, 84, 3) uint8
            'rel_actions': data['rel_actions'].astype(np.float32),  # (7,) float32
            'robot_obs': data['robot_obs'].astype(np.float32),      # (15,) float32
            'scene_obs': data['scene_obs'].astype(np.float32),      # (24,) float32
        }
        steps.append(step)

    if len(steps) == 0:
        return None

    # Build RLDS-compatible feature structure
    # Each step is stored as a SequenceExample or repeated features
    steps_features = []
    for i, step in enumerate(steps):
        is_first = (i == 0)
        is_last = (i == len(steps) - 1)

        step_feature = {
            'observation/image': _bytes_feature(
                tf.io.encode_png(step['rgb_static']).numpy()),
            'observation/wrist_image': _bytes_feature(
                tf.io.encode_png(step['rgb_gripper']).numpy()),
            'observation/state': _float_feature(step['robot_obs']),
            'action': _float_feature(step['rel_actions']),
            'language_instruction': _bytes_feature(
                episode['instruction'].encode('utf-8')),
            'discount': _float_feature(np.array([1.0], dtype=np.float32)),
            'reward': _float_feature(
                np.array([1.0 if is_last else 0.0], dtype=np.float32)),
            'is_first': _int_feature(int(is_first)),
            'is_last': _int_feature(int(is_last)),
            'is_terminal': _int_feature(int(is_last)),
        }
        steps_features.append(step_feature)

    return steps_features


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecords(episodes, data_dir, output_path, split_name,
                    steps_per_shard=5000):
    """Write episodes to sharded TFRecord files."""
    os.makedirs(output_path, exist_ok=True)

    shard_idx = 0
    step_count = 0
    total_episodes = 0
    total_steps = 0
    writer = None

    for ep in tqdm(episodes, desc="Converting {}".format(split_name)):
        steps_features = episode_to_tf_example(ep, data_dir)
        if steps_features is None:
            continue

        for step_feat in steps_features:
            if writer is None or step_count >= steps_per_shard:
                if writer:
                    writer.close()
                shard_path = os.path.join(
                    output_path,
                    "calvin_dataset-{}.tfrecord-{:05d}".format(
                        split_name, shard_idx))
                writer = tf.io.TFRecordWriter(shard_path)
                shard_idx += 1
                step_count = 0

            example = tf.train.Example(
                features=tf.train.Features(feature=step_feat))
            writer.write(example.SerializeToString())
            step_count += 1
            total_steps += 1

        total_episodes += 1

    if writer:
        writer.close()

    print("  {} split: {} episodes, {} steps, {} shards".format(
        split_name, total_episodes, total_steps, shard_idx))
    return total_episodes, total_steps, shard_idx


def write_metadata(output_path, train_info, val_info):
    """Write dataset_info.json and features.json for TFDS compatibility."""
    dataset_info = {
        "name": "calvin_dataset",
        "version": "1.0.0",
        "description": "CALVIN split D (task_D_D) converted to RLDS format",
        "splits": {
            "train": {"num_examples": train_info[1], "num_shards": train_info[2]},
            "val": {"num_examples": val_info[1], "num_shards": val_info[2]},
        },
    }
    with open(os.path.join(output_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    features = {
        "observation/image": {"shape": [200, 200, 3], "dtype": "uint8",
                               "encoding": "png"},
        "observation/wrist_image": {"shape": [84, 84, 3], "dtype": "uint8",
                                     "encoding": "png"},
        "observation/state": {"shape": [15], "dtype": "float32"},
        "action": {"shape": [7], "dtype": "float32"},
        "language_instruction": {"dtype": "string"},
        "discount": {"shape": [1], "dtype": "float32"},
        "reward": {"shape": [1], "dtype": "float32"},
        "is_first": {"dtype": "bool"},
        "is_last": {"dtype": "bool"},
        "is_terminal": {"dtype": "bool"},
    }
    with open(os.path.join(output_path, "features.json"), "w") as f:
        json.dump(features, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CALVIN D_D to RLDS format")
    parser.add_argument("--calvin_dir", type=str,
                        default="/data/user_data/yashagar/task_D_D",
                        help="Root of CALVIN task_D_D directory")
    parser.add_argument("--output_dir", type=str,
                        default="/data/user_data/wenjiel2/datasets/calvin_rlds",
                        help="Output directory for RLDS TFRecords")
    parser.add_argument("--steps_per_shard", type=int, default=5000)
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, "calvin_dataset", "1.0.0")
    os.makedirs(output_path, exist_ok=True)

    # Training split
    print("Loading training annotations...")
    train_dir = os.path.join(args.calvin_dir, "training")
    train_episodes = load_annotations(train_dir)
    print("  {} annotated episodes".format(len(train_episodes)))

    print("Writing training TFRecords...")
    train_info = write_tfrecords(
        train_episodes, train_dir, output_path, "train",
        args.steps_per_shard)

    # Validation split
    print("\nLoading validation annotations...")
    val_dir = os.path.join(args.calvin_dir, "validation")
    val_episodes = load_annotations(val_dir)
    print("  {} annotated episodes".format(len(val_episodes)))

    print("Writing validation TFRecords...")
    val_info = write_tfrecords(
        val_episodes, val_dir, output_path, "val",
        args.steps_per_shard)

    # Write metadata
    write_metadata(output_path, train_info, val_info)
    print("\nDone! RLDS dataset saved to {}".format(output_path))


if __name__ == "__main__":
    main()
