"""Convert LIBERO-Goal HDF5 demos → flat (CSV + per-demo .npy) layout that mirrors
Bridge's `bridge_episodes_filtered.csv` + `bridge_actions/` so our existing
tokenizer training pipeline (tokenization/train_tokenizer.py) can read it
unchanged.

Outputs:
  - data/libero_goal_episodes.csv with columns:
        episode_idx, task_name, instruction, verb, n_steps, episode_key
  - /data/user_data/wenjiel2/datasets/libero_goal_actions/
        episode_<idx>.npy   (T, 7) float32 — Δx Δy Δz Δrx Δry Δrz gripper

10 tasks × 50 demos = 500 episodes. Verbs are coarse (only ~4 unique:
open, put, push, turn) — verb-MI on this tokenizer will be lower-resolution
than Bridge but enough to train OAT and run policy fine-tuning.

Run:
    python scripts/convert_libero_to_csv.py
"""
import argparse
import csv
import os
import re
from pathlib import Path

import h5py
import numpy as np


VERB_MAP = {
    # First word of task_name → coarse verb label.
    "open": "open",
    "push": "push",
    "put": "put",
    "turn": "turn",
}


def task_to_verb(task_name: str) -> str:
    return VERB_MAP.get(task_name.split("_")[0], "")


def task_name_to_instruction(task_name: str) -> str:
    """`put_the_bowl_on_the_plate_demo` → `put the bowl on the plate`."""
    base = re.sub(r"_demo$", "", task_name)
    return base.replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir",
                        default="/data/user_data/wenjiel2/datasets/libero_data/libero_goal")
    parser.add_argument("--out_csv",
                        default="data/libero_goal_episodes.csv")
    parser.add_argument("--actions_dir",
                        default="/data/user_data/wenjiel2/datasets/libero_goal_actions")
    args = parser.parse_args()

    os.makedirs(args.actions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows, episode_idx = [], 0
    hdf5_files = sorted(Path(args.hdf5_dir).glob("*.hdf5"))
    print(f"Reading {len(hdf5_files)} HDF5 files from {args.hdf5_dir}")

    for hdf5_path in hdf5_files:
        task_name = hdf5_path.stem  # e.g. "put_the_bowl_on_the_plate_demo"
        instruction = task_name_to_instruction(task_name)
        verb = task_to_verb(task_name)

        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for d in demos:
                actions = np.asarray(f[f"data/{d}/actions"], dtype=np.float32)
                np.save(os.path.join(args.actions_dir, f"episode_{episode_idx}.npy"),
                        actions)
                rows.append({
                    "episode_idx":  episode_idx,
                    "task_name":    task_name,
                    "instruction":  instruction,
                    "verb":         verb,
                    "n_steps":      actions.shape[0],
                    "episode_key":  f"{hdf5_path.name}|{d}",
                })
                episode_idx += 1
        print(f"  {hdf5_path.name}: {len(demos)} demos  (instruction={instruction!r}  verb={verb!r})")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Quick stats.
    import collections
    verb_counts = collections.Counter(r["verb"] for r in rows)
    print(f"\nWrote {len(rows)} episodes → {args.out_csv}")
    print(f"  Verb distribution: {dict(verb_counts)}")
    print(f"  Action npy → {args.actions_dir}")


if __name__ == "__main__":
    main()
