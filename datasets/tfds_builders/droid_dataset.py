"""
droid_dataset.py

TFDS DatasetBuilder for filtered DROID episodes in RLDS episode format.

This builder reuses the downloaded raw DROID TFRecord shards and filters them with
the repo-local metadata cache produced for tokenizer training. That keeps the policy
data aligned with the single-verb DROID subset used in the tokenizer sweep.

Usage:
    python -m datasets.tfds_builders.droid_dataset \
        --output_dir /data/user_data/wenjiel2/datasets/droid_rlds_cache

Then pass --data_root_dir /data/user_data/wenjiel2/datasets/droid_rlds_cache
and --dataset_name droid_dataset to policy training.
"""

import glob
import os
import sys
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from datasets.tfrecord_parser import parse_tf_example, read_tfrecords


DROID_RLDS_DIR = "/data/user_data/wenjiel2/datasets/droid_rlds"
DROID_METADATA_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "droid_tokenizer_metadata.csv",
)


def _decode_bytes_scalar(feature: Dict[str, Any], key: str, default: str = "") -> str:
    values = feature.get(key, {}).get("bytes_list", [])
    if not values:
        return default
    return values[0].decode("utf-8", errors="ignore")


def _bytes_list(feature: Dict[str, Any], key: str) -> list[bytes]:
    return list(feature.get(key, {}).get("bytes_list", []))


def _float_array(feature: Dict[str, Any], key: str, rows: int, cols: int) -> np.ndarray:
    values = feature.get(key, {}).get("float_list", [])
    array = np.asarray(values, dtype=np.float32)
    if array.size != rows * cols:
        raise ValueError(
            f"Feature {key} has {array.size} values, expected {rows * cols}"
        )
    return array.reshape(rows, cols)


def _float_vector(feature: Dict[str, Any], key: str, length: int, default: float = 0.0) -> np.ndarray:
    values = feature.get(key, {}).get("float_list", [])
    if not values:
        return np.full(length, default, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.size != length:
        raise ValueError(
            f"Feature {key} has {array.size} values, expected {length}"
        )
    return array


def _bool_vector(feature: Dict[str, Any], key: str, length: int) -> np.ndarray:
    values = feature.get(key, {}).get("int64_list", [])
    if not values:
        return np.zeros(length, dtype=np.bool_)
    array = np.asarray(values, dtype=np.int64)
    if array.size != length:
        raise ValueError(
            f"Feature {key} has {array.size} values, expected {length}"
        )
    return array.astype(np.bool_)


def _load_split_lookup(
    metadata_cache: str,
    val_fraction: float,
    seed: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    df = pd.read_csv(metadata_cache)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_fraction))
    val_df = df.iloc[perm[:n_val]].reset_index(drop=True)
    train_df = df.iloc[perm[n_val:]].reset_index(drop=True)

    def to_lookup(split_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        for row in split_df.itertuples(index=False):
            lookup[getattr(row, "episode_path")] = {
                "instruction": getattr(row, "instruction"),
                "primary_verb": getattr(row, "primary_verb", ""),
                "episode_idx": int(getattr(row, "episode_idx")),
                "shard_path": getattr(row, "shard_path"),
            }
        return lookup

    return {
        "train": to_lookup(train_df),
        "val": to_lookup(val_df),
    }


class DroidDataset(tfds.core.GeneratorBasedBuilder):
    """Filtered DROID dataset in RLDS episode format for policy training."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": (
            "Initial release from raw DROID RLDS shards, filtered to the single-verb "
            "metadata cache used by the tokenizer sweep."
        )
    }

    def __init__(self, *args, metadata_cache: str = DROID_METADATA_CACHE,
                 droid_rlds_dir: str = DROID_RLDS_DIR, val_fraction: float = 0.1,
                 seed: int = 42, max_shards: int | None = None,
                 max_episodes_per_split: int | None = None, **kwargs):
        self._metadata_cache = metadata_cache
        self._droid_rlds_dir = droid_rlds_dir
        self._val_fraction = val_fraction
        self._seed = seed
        self._max_shards = max_shards
        self._max_episodes_per_split = max_episodes_per_split
        self._split_lookup = _load_split_lookup(metadata_cache, val_fraction, seed)
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "DROID real-robot manipulation dataset filtered to the single-verb subset "
                "used for tokenizer training. Observations keep the left exterior camera, "
                "the left wrist camera, and a compact 14-d proprioceptive state."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(encoding_format="jpeg"),
                        "wrist_image": tfds.features.Image(encoding_format="jpeg"),
                        "state": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "primary_verb": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
            homepage="https://droid-dataset.github.io/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(self._split_lookup["train"]),
            "val": self._generate_examples(self._split_lookup["val"]),
        }

    def _generate_examples(self, split_lookup: Dict[str, Dict[str, Any]]) -> Iterator[Tuple[str, Any]]:
        shard_paths = sorted(glob.glob(os.path.join(self._droid_rlds_dir, "*.tfrecord-*")))
        if self._max_shards is not None:
            shard_paths = shard_paths[:self._max_shards]
        if not shard_paths:
            raise FileNotFoundError(
                f"No DROID TFRecord shards found under {self._droid_rlds_dir}"
            )

        yielded = 0
        for shard_path in shard_paths:
            for raw_record in read_tfrecords(shard_path):
                feature = parse_tf_example(raw_record)
                episode_path = _decode_bytes_scalar(feature, "episode_metadata/file_path")
                if not episode_path:
                    continue

                meta = split_lookup.get(episode_path)
                if meta is None:
                    continue

                images = _bytes_list(feature, "steps/observation/exterior_image_1_left")
                if not images:
                    images = _bytes_list(feature, "steps/observation/exterior_image_2_left")
                wrist_images = _bytes_list(feature, "steps/observation/wrist_image_left")
                if not wrist_images:
                    wrist_images = images

                num_steps = len(images)
                if num_steps == 0:
                    continue

                try:
                    actions = _float_array(feature, "steps/action", num_steps, 7)
                    tcp = _float_array(feature, "steps/observation/cartesian_position", num_steps, 6)
                    gripper = _float_vector(
                        feature, "steps/observation/gripper_position", num_steps
                    ).reshape(num_steps, 1)
                    joints = _float_array(feature, "steps/observation/joint_position", num_steps, 7)
                except ValueError:
                    continue

                state = np.concatenate([tcp, gripper, joints], axis=1).astype(np.float32)
                rewards = _float_vector(feature, "steps/reward", num_steps)
                discounts = _float_vector(feature, "steps/discount", num_steps, default=1.0)
                is_first = _bool_vector(feature, "steps/is_first", num_steps)
                is_last = _bool_vector(feature, "steps/is_last", num_steps)
                is_terminal = _bool_vector(feature, "steps/is_terminal", num_steps)
                instruction = meta["instruction"]

                steps = []
                for index in range(num_steps):
                    steps.append({
                        "observation": {
                            "image": images[index],
                            "wrist_image": wrist_images[index],
                            "state": state[index],
                        },
                        "action": actions[index],
                        "language_instruction": instruction,
                        "is_first": bool(is_first[index]),
                        "is_last": bool(is_last[index]),
                        "is_terminal": bool(is_terminal[index]),
                        "reward": float(rewards[index]),
                        "discount": float(discounts[index]),
                    })

                yield episode_path, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": episode_path,
                        "primary_verb": meta.get("primary_verb", ""),
                    },
                }
                yielded += 1
                if self._max_episodes_per_split is not None and yielded >= self._max_episodes_per_split:
                    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build filtered DROID TFDS dataset")
    parser.add_argument(
        "--output_dir",
        default="/data/user_data/wenjiel2/datasets/droid_rlds_cache",
        help="Output directory (TFDS data_dir)",
    )
    parser.add_argument(
        "--metadata_cache",
        default=DROID_METADATA_CACHE,
        help="CSV cache with filtered DROID metadata",
    )
    parser.add_argument(
        "--droid_rlds_dir",
        default=DROID_RLDS_DIR,
        help="Directory containing raw DROID TFRecord shards",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Optional limit on raw DROID shards for smoke builds")
    parser.add_argument("--max_episodes_per_split", type=int, default=None,
                        help="Optional limit on episodes generated per split for smoke builds")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Building DROID TFDS dataset -> {args.output_dir}")
    builder = DroidDataset(
        data_dir=args.output_dir,
        metadata_cache=args.metadata_cache,
        droid_rlds_dir=args.droid_rlds_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_shards=args.max_shards,
        max_episodes_per_split=args.max_episodes_per_split,
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            manual_dir=args.droid_rlds_dir,
            beam_options=None,
        )
    )
    print("Done! Dataset info:")
    print(builder.info)