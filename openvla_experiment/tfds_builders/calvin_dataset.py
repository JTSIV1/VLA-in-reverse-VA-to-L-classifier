"""
calvin_dataset.py

TFDS DatasetBuilder for CALVIN task_D_D in RLDS episode format.

Usage (build once before training):
    python -m openvla_experiment.tfds_builders.calvin_dataset \
        --output_dir /data/user_data/wenjiel2/datasets/calvin_rlds

Then pass --data_root_dir /data/user_data/wenjiel2/datasets/calvin_rlds
and --dataset_name calvin_dataset to the finetune script.

CALVIN robot_obs layout (15-dim):
    [0:3]  tcp_pos (xyz)
    [3:6]  tcp_orn (euler xyz)
    [6:9]  tcp_vel (xyz linear velocity)
    [9]    gripper_opening_width (0=closed, ~0.08=open)
    [10]   gripper_vel
    [11:15] arm_joint_states (4 joints)

CALVIN rel_actions layout (7-dim):
    [0:3]  delta_xyz
    [3:6]  delta_euler
    [6]    gripper: -1=open, +1=close (relative)
"""

import os
from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

CALVIN_DIR = "/data/user_data/yashagar/task_D_D"


class CalvinDataset(tfds.core.GeneratorBasedBuilder):
    """CALVIN task_D_D dataset in RLDS episode format for OpenVLA-mini."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release. CALVIN task_D_D, language-annotated episodes."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "CALVIN task_D_D language-conditioned robot manipulation dataset. "
                "5124 train / 1011 val episodes with EEF relative actions."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        # Primary camera: static scene view (200x200)
                        "image": tfds.features.Image(
                            shape=(200, 200, 3), dtype=np.uint8, encoding_format="png"
                        ),
                        # Wrist camera: robot end-effector view (84x84)
                        "wrist_image": tfds.features.Image(
                            shape=(84, 84, 3), dtype=np.uint8, encoding_format="png"
                        ),
                        # Robot proprioception (15-dim, see module docstring)
                        "state": tfds.features.Tensor(shape=(15,), dtype=np.float32),
                    }),
                    # Relative EEF action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    # gripper: -1=open, +1=close (will be converted in transform)
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                }),
            }),
            supervised_keys=None,
            homepage="https://github.com/mees/calvin",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(
                split_dir=os.path.join(CALVIN_DIR, "training")
            ),
            "val": self._generate_examples(
                split_dir=os.path.join(CALVIN_DIR, "validation")
            ),
        }

    def _generate_examples(self, split_dir: str) -> Iterator[Tuple[str, Any]]:
        ann_path = os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy")
        ann = np.load(ann_path, allow_pickle=True).item()

        instructions = ann["language"]["ann"]  # list of strings
        indices = ann["info"]["indx"]           # list of (start_idx, end_idx) tuples

        for ep_idx, (instruction, (start, end)) in enumerate(zip(instructions, indices)):
            steps = []
            for frame_idx in range(int(start), int(end) + 1):
                path = os.path.join(split_dir, "episode_{:07d}.npz".format(frame_idx))
                if not os.path.exists(path):
                    continue
                data = np.load(path)
                steps.append({
                    "observation": {
                        "image": data["rgb_static"],           # (200, 200, 3) uint8
                        "wrist_image": data["rgb_gripper"],    # (84, 84, 3) uint8
                        "state": data["robot_obs"].astype(np.float32),  # (15,) float32
                    },
                    "action": data["rel_actions"].astype(np.float32),   # (7,) float32
                    "language_instruction": str(instruction),
                    "is_first": bool(frame_idx == start),
                    "is_last": bool(frame_idx == end),
                    "is_terminal": bool(frame_idx == end),
                    "reward": float(frame_idx == end),
                    "discount": 1.0,
                })

            if steps:
                yield ep_idx, {"steps": steps}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build CALVIN TFDS dataset")
    parser.add_argument(
        "--output_dir",
        default="/data/user_data/wenjiel2/datasets/calvin_rlds",
        help="Output directory (TFDS data_dir)"
    )
    parser.add_argument("--num_shards", type=int, default=64)
    args = parser.parse_args()

    # Add project root to path
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    print(f"Building CALVIN TFDS dataset -> {args.output_dir}")
    builder = CalvinDataset(data_dir=args.output_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            manual_dir=CALVIN_DIR,
            beam_options=None,
        )
    )
    print("Done! Dataset info:")
    print(builder.info)
