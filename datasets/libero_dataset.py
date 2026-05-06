"""LIBERO-Goal dataset loaders.

Mirrors the BridgeV2 / DROID interfaces (see `bridge_dataset.py`,
`droid_dataset.py`) so the existing tokenizer training pipeline in
`tokenization/train_tokenizer.py` can use LIBERO via a `--dataset libero_goal`
switch with no further changes downstream.

Data layout produced by `scripts/convert_libero_to_csv.py`:

    data/libero_goal_episodes.csv
        episode_idx, task_name, instruction, verb, n_steps, episode_key

    /data/user_data/wenjiel2/datasets/libero_goal_actions/
        episode_<idx>.npy   (T, 7) float32
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_CSV = "data/libero_goal_episodes.csv"
DEFAULT_ACTIONS_DIR = "/data/user_data/wenjiel2/datasets/libero_goal_actions"


# ──────────────────────────────────────────────────────────────────────────
# Action arrays
# ──────────────────────────────────────────────────────────────────────────
def load_libero_actions(
    actions_dir: str = DEFAULT_ACTIONS_DIR,
    csv_path: str = DEFAULT_CSV,
) -> Tuple[List[np.ndarray], List[str]]:
    """Load (actions, episode_keys) for every episode listed in `csv_path`.

    Returns
    -------
    actions      : list of (T, 7) float32 arrays
    episode_keys : list of strings — the `episode_key` column of the CSV,
                   in the same order as `actions`.
    """
    df = pd.read_csv(csv_path)
    actions: List[np.ndarray] = []
    keys: List[str] = []
    for _, row in df.iterrows():
        path = os.path.join(actions_dir, f"episode_{int(row['episode_idx'])}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Action file missing: {path}. Did you run "
                f"scripts/convert_libero_to_csv.py?"
            )
        actions.append(np.load(path).astype(np.float32))
        keys.append(row["episode_key"])
    print(f"[libero] loaded {len(actions)} episodes from {actions_dir}")
    return actions, keys


# ──────────────────────────────────────────────────────────────────────────
# Verb labels
# ──────────────────────────────────────────────────────────────────────────
def load_libero_verb_labels(
    csv_path: str,
    keys: List[str],
    min_class_count: int = 0,
) -> Tuple[List[int], Dict[str, int]]:
    """Return (verb_ids in same order as `keys`, verb_to_id)."""
    df = pd.read_csv(csv_path)
    verb_by_key = dict(zip(df["episode_key"], df["verb"]))

    # Optional min-class-count filter (drops verbs with < N episodes).
    if min_class_count > 0:
        from collections import Counter
        counts = Counter(verb_by_key.values())
        keep = {v for v, c in counts.items() if c >= min_class_count}
        verb_by_key = {k: (v if v in keep else "") for k, v in verb_by_key.items()}

    verb_to_id: Dict[str, int] = {}
    verb_ids: List[int] = []
    for k in keys:
        v = verb_by_key.get(k, "")
        if not v:
            verb_ids.append(-1)         # unmatched / dropped
            continue
        if v not in verb_to_id:
            verb_to_id[v] = len(verb_to_id)
        verb_ids.append(verb_to_id[v])
    print(f"[libero] {len(verb_to_id)} verb classes "
          f"(min_class_count={min_class_count})")
    return verb_ids, verb_to_id


# ──────────────────────────────────────────────────────────────────────────
# Instructions
# ──────────────────────────────────────────────────────────────────────────
def load_libero_instructions(
    csv_path: str,
    keys: List[str],
) -> List[str]:
    """Return the instruction string for each `episode_key`, in order."""
    df = pd.read_csv(csv_path)
    instr_by_key = dict(zip(df["episode_key"], df["instruction"]))
    out: List[str] = []
    matched, missing = 0, 0
    for k in keys:
        text = instr_by_key.get(k)
        if isinstance(text, str) and text.strip():
            out.append(text); matched += 1
        else:
            out.append(""); missing += 1
    print(f"[libero] instructions: matched {matched}, missing {missing}")
    return out
