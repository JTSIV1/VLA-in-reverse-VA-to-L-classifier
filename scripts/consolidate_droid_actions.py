"""Consolidate extracted DROID action shards into a single dataset.

Reads all shard_*.npz files, extracts verbs via spaCy, applies filtering,
and produces:
  1. data/droid_episodes.csv — episode index with verb labels
  2. /data/user_data/wenjiel2/datasets/droid_actions/all_actions.npz —
     compact action trajectories keyed by episode index

Usage:
    python scripts/consolidate_droid_actions.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_verb

SHARD_DIR = "/data/user_data/wenjiel2/datasets/droid_actions"
CSV_OUT = "data/droid_episodes.csv"


def main():
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
    print(f"Found {len(shard_files)} shard files")

    rows = []
    all_actions = {}
    global_idx = 0

    for shard_path in tqdm(shard_files, desc="Loading shards"):
        data = np.load(shard_path, allow_pickle=True)
        n_episodes = int(data["n_episodes"])

        for i in range(n_episodes):
            actions = data[f"actions_{i}"]
            lang1 = str(data[f"lang1_{i}"])
            lang2 = str(data[f"lang2_{i}"])
            lang3 = str(data[f"lang3_{i}"])
            ep_path = str(data[f"episode_path_{i}"])

            # Extract verb from primary instruction
            verbs = extract_verb(lang1)

            rows.append({
                "episode_idx": global_idx,
                "instruction": lang1,
                "instruction_2": lang2,
                "instruction_3": lang3,
                "episode_path": ep_path,
                "n_steps": actions.shape[0],
                "n_verbs": len(verbs),
                "verb": verbs[0] if len(verbs) == 1 else "",
                "all_verbs": ";".join(verbs),
            })

            all_actions[f"actions_{global_idx}"] = actions
            global_idx += 1

    print(f"\nTotal episodes loaded: {global_idx}")

    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"Verb extraction: {(df['n_verbs'] == 1).sum()} single-verb, "
          f"{(df['n_verbs'] == 0).sum()} zero-verb, "
          f"{(df['n_verbs'] > 1).sum()} multi-verb")

    # Filter to single-verb episodes
    df_single = df[df["n_verbs"] == 1].copy()

    # Filter 'then' and 'and' instructions (same as CALVIN pipeline)
    pre_then = len(df_single)
    df_single = df_single[~df_single["instruction"].str.contains(r"\bthen\b", case=False)].copy()
    print(f"Filtered {pre_then - len(df_single)} 'then' instructions")

    pre_and = len(df_single)
    and_mask = (df_single["instruction"].str.contains(r"\band\b", case=False) &
                ~df_single["instruction"].str.lower().str.startswith("go"))
    df_single = df_single[~and_mask].copy()
    print(f"Filtered {pre_and - len(df_single)} 'and' instructions")

    # Disambiguate 'turn on' / 'turn off'
    turn_on = (df_single["verb"] == "turn") & df_single["instruction"].str.contains(r"\bturn on\b", case=False)
    df_single.loc[turn_on, "verb"] = "turn on"
    turn_off = (df_single["verb"] == "turn") & df_single["instruction"].str.contains(r"\bturn off\b", case=False)
    df_single.loc[turn_off, "verb"] = "turn off"

    # Collapse verb variants — merge map
    VERB_MERGE = {
        # Clear merges: particle doesn't change meaning
        "flip over": "flip", "flip up": "flip", "flip down": "flip",
        "fold up": "fold", "fold over": "fold",
        "stack up": "stack",
        "pile up": "pile",
        "straighten out": "straighten", "straighten up": "straighten",
        "spread out": "spread", "spread across": "spread",
        "stretch out": "stretch",
        "clean up": "clean",
        "hang up": "hang",
        "coil up": "coil",
        "scrunch up": "scrunch",
        "wrap up": "wrap",
        "zip up": "zip",
        "roll up": "roll", "roll out": "roll",
        "pick": "pick up",
        "lay down": "lay", "lay out": "lay",
        "scoop up": "scoop",
        "spill out": "spill",
        "spoon out": "spoon",
        "spell out": "spell",
        "drop down": "drop",
        "knock down": "knock",
        # Merge with caution: particle slightly specializes
        "press down": "press",
        "push down down": "push down",
        "slide up": "slide", "slide down": "slide",
        "slide out": "slide", "slide off": "slide",
        "remove out": "remove",
        "open up": "open", "open off": "open",
        "put down": "put", "put on": "put", "put up": "put", "put in": "put",
        "place up": "place", "place in": "place",
        "move up": "move", "move over": "move", "move on": "move",
        "lift up": "lift",
        # Keep-separate overrides: merge directional variants
        "pull out": "pull", "pull up": "pull", "pull down": "pull", "pull on": "pull",
        "push down": "push", "push up": "push", "push in": "push",
        "push on": "push", "push off": "push", "push out": "push",
        "pour out": "pour", "pour on": "pour",
        "flick up": "flick", "flick down": "flick", "flick off": "flick",
        "plug in": "plug",
        # Cross-verb merges
        "turn over": "flip",
    }
    df_single["verb"] = df_single["verb"].replace(VERB_MERGE)

    print(f"\nFinal: {len(df_single)} episodes, {df_single['verb'].nunique()} unique verbs")
    print(f"\nVerb distribution (top 30):")
    print(df_single["verb"].value_counts().head(30).to_string())

    # Save CSV (all episodes, filtered column marks which to use)
    df["filtered_in"] = df.index.isin(df_single.index)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved full CSV to {CSV_OUT}")

    # Save filtered CSV separately
    filtered_csv = CSV_OUT.replace(".csv", "_filtered.csv")
    df_single[["episode_idx", "instruction", "verb", "n_steps", "episode_path"]].to_csv(
        filtered_csv, index=False
    )
    print(f"Saved filtered CSV to {filtered_csv}")

    # Save consolidated actions
    all_actions["n_episodes"] = global_idx
    actions_out = os.path.join(SHARD_DIR, "all_actions.npz")
    print(f"Saving consolidated actions to {actions_out}...")
    np.savez_compressed(actions_out, **all_actions)
    print(f"Done. File size: {os.path.getsize(actions_out) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
