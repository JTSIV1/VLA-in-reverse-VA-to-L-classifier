"""Consolidate BridgeV2 action shards into a high-level verb dataset.

Extracts verbs from episode-level instructions (not subtask segments).
Produces:
  - data/bridge_episodes_filtered.csv

Usage:
    python scripts/consolidate_bridge_actions.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_verb

SHARD_DIR = "/data/user_data/wenjiel2/datasets/bridge_actions"
CSV_OUT = "data/bridge_episodes.csv"


def main():
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))
    print(f"Found {len(shard_files)} shard files")

    rows = []
    global_idx = 0

    for shard_path in tqdm(shard_files, desc="Loading shards"):
        data = np.load(shard_path, allow_pickle=True)
        n_episodes = int(data["n_episodes"])

        for i in range(n_episodes):
            instruction = str(data[f"instruction_{i}"])
            n_steps = int(data[f"n_steps_{i}"])
            ep_key = str(data[f"episode_key_{i}"])

            verbs = extract_verb(instruction)

            rows.append({
                "episode_idx": global_idx,
                "instruction": instruction,
                "episode_key": ep_key,
                "n_steps": n_steps,
                "n_verbs": len(verbs),
                "verb": verbs[0] if len(verbs) == 1 else "",
                "all_verbs": ";".join(verbs),
            })
            global_idx += 1

    print(f"\nTotal episodes: {global_idx}")

    df = pd.DataFrame(rows)
    print(f"Verb extraction: {(df['n_verbs'] == 1).sum()} single-verb, "
          f"{(df['n_verbs'] == 0).sum()} zero-verb, "
          f"{(df['n_verbs'] > 1).sum()} multi-verb")

    # Filter to single-verb
    df_single = df[df["n_verbs"] == 1].copy()

    pre_then = len(df_single)
    df_single = df_single[~df_single["instruction"].str.contains(r"\bthen\b", case=False)].copy()
    print(f"Filtered {pre_then - len(df_single)} 'then' instructions")

    pre_and = len(df_single)
    and_mask = (df_single["instruction"].str.contains(r"\band\b", case=False) &
                ~df_single["instruction"].str.lower().str.startswith("go"))
    df_single = df_single[~and_mask].copy()
    print(f"Filtered {pre_and - len(df_single)} 'and' instructions")

    # --- Lemmatize conjugated/misspelled forms to base verb ---
    VERB_LEMMA = {
        # Conjugated forms
        "moved": "move", "moves": "move", "moving": "move", "moove": "move",
        "movve": "move", "mpve": "move", "mov": "move", "mowed": "move",
        "placed": "place", "places": "place", "placing": "place",
        "puts": "put", "putting": "put", "puting": "put", "putto": "put",
        "pput": "put", "puuting": "put", "putted": "put",
        "folded": "fold", "folds": "fold", "folding": "fold",
        "flod": "fold", "foold": "fold", "folld": "fold",
        "unfolded": "unfold", "unfolds": "unfold", "unfolding": "unfold",
        "unflod": "unfold", "unfoldt": "unfold", "nfold": "unfold",
        "opened": "open", "opens": "open", "opening": "open",
        "oppen": "open", "opeen": "open",
        "closed": "close", "closes": "close", "closing": "close",
        "clóse": "close",
        "removed": "remove", "removes": "remove", "removing": "remove",
        "pushed": "push", "pushes": "push", "pushing": "push",
        "pulled": "pull", "pulling": "pull",
        "touched": "touch", "touches": "touch", "touching": "touch",
        "covered": "cover", "covers": "cover",
        "took": "take", "takes": "take", "taken": "take", "taking": "take",
        "picked": "pick up", "picks": "pick up", "picking": "pick up",
        "picked up": "pick up", "picking up": "pick up",
        "dropped": "drop", "drops": "drop",
        "lifted": "lift", "lifting": "lift",
        "turned": "turn", "turning": "turn",
        "dragging": "drag", "dragged": "drag",
        "spreading": "spread",
        "transferred": "transfer", "transferring": "transfer", "transfered": "transfer",
        "reaching": "reach",
        "showing": "show",
        "doing": "do", "does": "do", "did": "do", "done": "do",
        "loaded": "load",
        "rotated": "rotate",
        "scratched": "scratch",
        "flipped": "flip",
        "arranged": "arrange",
        "wiped": "wipe",
        "separating": "separate", "separates": "separate",
        "carrying": "carry", "carried": "carry",
        "cooking": "cook",
        "standing": "stand", "standing up": "stand",
        "sets": "set",
        "raised": "raise",
        "straightens": "straighten",
        "builds": "build",
        "grabs": "grab",
        "gets": "get",
        "went": "go",
        "kept": "keep",
        "entering": "enter",
        "working": "work",
        "balancing": "balance",
        "pressing down": "press",
        "forms": "form",
        "fits": "fit",
        "fires": "fire",
        "throws": "throw",
        "holding": "hold",
        "cleaning": "clean",
        # Typos / non-English → drop (map to "")
        "front": "", "pan": "",
        "strawberry": "", "pear": "", "spoon": "", "fork": "", "broccoli": "",
        "azul": "", "abriu": "", "colocou": "", "sacar": "", "roxo": "",
        "tapa": "", "souleve": "", "sortie": "", "sortir": "", "vers": "",
        "quemadores": "", "fogão": "", "atras": "", "prend": "", "l'objet": "",
        "papa": "", "object": "",
        "gre": "", "iuyt": "", "ove": "", "noothing": "", "ideplace": "",
        "rezfz": "", "cub": "", "cxg": "", "dfg": "", "sliver": "",
        "hgmbnjgkktk": "", "botttom": "", "tuu": "", "iray": "", "inti": "",
        "midle": "", "top": "", "retge": "", "moveu": "",
        "parallelepiped": "", "uncloth": "",
        "chose": "", "mention": "",
        "is": "", "has": "", "had": "", "found": "", "spotted": "",
        "given": "", "appear": "",
    }
    df_single["verb"] = df_single["verb"].replace(VERB_LEMMA)

    # Drop rows where verb was mapped to "" (non-verbs / non-English)
    n_before_drop = len(df_single)
    df_single = df_single[df_single["verb"] != ""].copy()
    print(f"Dropped {n_before_drop - len(df_single)} non-verb/non-English rows")

    # Verb merge map (directional variants → base)
    VERB_MERGE = {
        "flip over": "flip", "flip up": "flip", "flip down": "flip",
        "flip out": "flip", "turn over": "flip",
        "fold up": "fold", "fold over": "fold", "fold down": "fold",
        "fold in": "fold",
        "stack up": "stack",
        "pile up": "pile",
        "pick": "pick up", "pick in": "pick up", "pick up on": "pick up",
        "lay down": "lay", "lay out": "lay", "laying down": "lay",
        "put down": "put", "put on": "put", "put up": "put", "put in": "put",
        "put out": "put", "put over": "put", "put th": "put",
        "place up": "place", "place in": "place", "place on": "place",
        "place down": "place",
        "move up": "move", "move over": "move", "move on": "move",
        "move down": "move", "move in": "move", "move out": "move",
        "move off": "move", "move back": "move",
        "moved out": "move", "moved down": "move", "moved up": "move",
        "moving out": "move",
        "lift up": "lift",
        "pull out": "pull", "pull up": "pull", "pull down": "pull",
        "pulled up": "pull", "pulled out": "pull",
        "push down": "push", "push up": "push", "push in": "push",
        "push on": "push", "push off": "push",
        "slide up": "slide", "slide down": "slide",
        "slide out": "slide", "slide off": "slide", "slide over": "slide",
        "open up": "open", "opens up": "open",
        "close up": "close",
        "press down": "press",
        "pour out": "pour",
        "spread out": "spread",
        "roll up": "roll", "roll out": "roll",
        "hang up": "hang",
        "clean up": "clean",
        "wrap up": "wrap",
        "cover up": "cover",
        "take out": "take", "take off": "take", "taking out": "take",
        "took up": "take",
        "bring across": "bring",
        "remove out": "remove", "remov on": "remove",
        "straighten up": "straighten",
        "turn up": "turn",
        "stand up": "stand",
        "get out": "get",
        "puts on": "put",
        "putting down": "put",
    }
    df_single["verb"] = df_single["verb"].replace(VERB_MERGE)

    print(f"\nFinal: {len(df_single)} episodes, {df_single['verb'].nunique()} unique verbs")
    print(f"\nVerb distribution (top 30):")
    print(df_single["verb"].value_counts().head(30).to_string())

    # Save
    filtered_csv = CSV_OUT.replace(".csv", "_filtered.csv")
    df_single[["episode_idx", "instruction", "verb", "n_steps", "episode_key"]].to_csv(
        filtered_csv, index=False
    )
    print(f"\nSaved filtered CSV to {filtered_csv}")


if __name__ == "__main__":
    main()
