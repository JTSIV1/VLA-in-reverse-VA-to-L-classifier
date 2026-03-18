"""Build episode_task_types.csv: classify each CALVIN episode and extract
syntactic slots from the instruction.

Task-type classification uses scene_obs + robot_obs only (no instruction):
  - fixture_manip: fixture dims (scene_obs[0:6]) changed
  - block_acquire: block moved AND gripper closed at end (holding block)
  - block_displace: block moved AND gripper open at end (released block)

Collateral-contact rule: when both fixture and block dims change, check
the most-displaced block's start and end z.  If both are below
DRAWER_Z_THRESH the block was sitting in the drawer and rode along →
fixture_manip.

Syntactic slots are extracted from instruction text with spaCy.

Usage:
    conda activate mmml
    python scripts/build_episode_task_types.py

Output:
    data/episode_task_types.csv
"""
import numpy as np
import os
import pandas as pd
import spacy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAIN_DIR, VAL_DIR, EPISODE_TEMPLATE, SPACY_MODEL
from utils import load_calvin_to_dataframe

nlp = spacy.load(SPACY_MODEL)

# ── scene_obs layout (24-d) ──────────────────────────────────────────────────
# 0: sliding door joint, 1: drawer joint, 2: button,
# 3: switch, 4: lightbulb, 5: green light
# 6-11: red block (xyz + euler), 12-17: blue block, 18-23: pink block

# ── Thresholds ────────────────────────────────────────────────────────────────
FIXTURE_THRESH = 0.01
BLOCK_XYZ_THRESH = 0.01       # 1 cm displacement
BLOCK_EULER_THRESH = 0.1      # ~5.7 degrees rotation
GRIPPER_CLOSED_THRESH = 0.05  # robot_obs[6] below this → holding block
DRAWER_Z_THRESH = 0.38        # block z below this → inside the drawer
FIXTURE_DISCRETE_THRESH = 0.5 # fixture delta above this → binary state flip (e.g. light on/off)

BLOCK_XYZ_SLICES = [slice(6, 9), slice(12, 15), slice(18, 21)]
BLOCK_EULER_SLICES = [slice(9, 12), slice(15, 18), slice(21, 24)]
BLOCK_Z_INDICES = [8, 14, 20]


# ── Task-type classification (scene_obs + robot_obs, no instruction) ─────────

def classify_task_type(ep_start, ep_end):
    """Classify episode purely from observations.

    Returns one of: fixture_manip, block_acquire, block_displace, neither.
    """
    sd = ep_end["scene_obs"] - ep_start["scene_obs"]
    end_gripper = ep_end["robot_obs"][6]

    # 1. Fixture changed?
    fixture_mag = max(abs(sd[i]) for i in range(6))
    any_fixture = fixture_mag > FIXTURE_THRESH

    # 2. Block changed?  (xyz OR euler)
    any_block_xyz = any(
        np.abs(sd[s]).max() > BLOCK_XYZ_THRESH for s in BLOCK_XYZ_SLICES
    )
    any_block_euler = any(
        np.abs(sd[s]).max() > BLOCK_EULER_THRESH for s in BLOCK_EULER_SLICES
    )
    any_block = any_block_xyz or any_block_euler

    gripper_closed = end_gripper < GRIPPER_CLOSED_THRESH

    # 3. Both changed → check collateral
    if any_fixture and any_block:
        # 3a. Discrete fixture state flip (e.g. light on→off, delta≈1.0)
        #     Block movement is incidental arm contact → fixture_manip
        if fixture_mag > FIXTURE_DISCRETE_THRESH:
            return "fixture_manip"

        # 3b. Block stayed inside the drawer → rode along with drawer
        block_mags = [np.abs(sd[s]).max() for s in BLOCK_XYZ_SLICES]
        max_block_idx = int(np.argmax(block_mags))
        block_start_z = ep_start["scene_obs"][BLOCK_Z_INDICES[max_block_idx]]
        block_end_z = ep_end["scene_obs"][BLOCK_Z_INDICES[max_block_idx]]

        if block_start_z < DRAWER_Z_THRESH and block_end_z < DRAWER_Z_THRESH:
            return "fixture_manip"
        elif gripper_closed:
            return "block_acquire"
        else:
            return "block_displace"

    # 4. Straightforward cases
    if any_fixture:
        return "fixture_manip"
    if any_block and gripper_closed:
        return "block_acquire"
    if any_block:
        return "block_displace"
    return "neither"


# ── Syntactic slot extraction (instruction text) ─────────────────────────────

PHRASAL_VERB_CLASSES = {"pick up", "take off", "turn on", "turn off"}
DISCOURSE_VERBS = {"go", "let"}
DIRECTION_WORDS = {"left", "right"}
LOCATION_NOUNS = {"drawer", "cabinet", "slider", "shelf", "table", "tower", "stack"}
SOURCE_PREPS = {"from"}
TARGET_PREPS = {"in", "into", "on"}
DIRECTION_PREPS = {"to", "towards"}


def _find_root_verb(doc):
    """Find the main action verb, skipping discourse verbs like 'go'."""
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            if token.text in DISCOURSE_VERBS:
                for child in token.children:
                    if child.dep_ in ("xcomp", "advcl", "conj") and child.pos_ == "VERB":
                        return child
            return token
    return None


def _get_compound_noun(token):
    """Get multi-word noun (e.g. 'sliding cabinet', 'light bulb')."""
    parts = []
    for child in token.children:
        if child.dep_ in ("amod", "compound") and child.pos_ in ("VERB", "ADJ", "NOUN"):
            if child.text in ("sliding", "light", "cabinet", "led"):
                parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


def extract_slots(inst, verb):
    """Extract object_modifier, object, direction, target_location from instruction."""
    doc = nlp(inst.lower())
    root = _find_root_verb(doc)

    result = {
        "object_modifier": None,
        "object": None,
        "direction": None,
        "target_location": None,
    }
    source_location = None

    if root is None:
        return result

    verbs = [root]
    for child in root.children:
        if child.dep_ in ("conj", "xcomp") and child.pos_ == "VERB":
            verbs.append(child)
    doc_root = [t for t in doc if t.dep_ == "ROOT"][0]
    if doc_root.text in DISCOURSE_VERBS and doc_root not in verbs:
        verbs.insert(0, doc_root)

    # --- Object and modifier ---
    for v in verbs:
        for child in v.children:
            if child.dep_ == "dobj" and result["object"] is None:
                if child.pos_ == "PRON":
                    continue
                result["object"] = child.text
                mods = [c.text for c in child.children
                        if c.dep_ in ("amod", "compound") and c.text not in DIRECTION_WORDS]
                if mods:
                    result["object_modifier"] = ";".join(mods)

    if result["object"] is None:
        for token in doc:
            if token.dep_ == "pobj" and token.head.text == "towards":
                if token.text not in ("left", "right"):
                    result["object"] = token.text
                    mods = [c.text for c in token.children
                            if c.dep_ == "amod" and c.text not in DIRECTION_WORDS]
                    if mods:
                        result["object_modifier"] = ";".join(mods)
                    break

    # --- Direction, source_location, target_location from PPs ---
    acquire_verbs = {"pick", "grasp", "lift", "take", "remove", "go"}
    place_verbs = {"place", "put", "store", "push", "slide", "sweep"}

    for token in doc:
        if token.dep_ == "prep" or (token.dep_ == "prt" and token.head.pos_ == "VERB"):
            if token.dep_ == "prt":
                full_verb = token.head.text + " " + token.text
                if full_verb in PHRASAL_VERB_CLASSES:
                    continue
                result["direction"] = token.text
                continue

            pobj = None
            for child in token.children:
                if child.dep_ == "pobj":
                    pobj = child
                    break
            if pobj is None:
                continue

            prep_text = token.text
            pobj_text = pobj.text
            pobj_compound = _get_compound_noun(pobj)

            if prep_text in DIRECTION_PREPS and pobj_text in ("left", "right"):
                result["direction"] = pobj_text
            elif prep_text in DIRECTION_PREPS and pobj_text == "side":
                for c in pobj.children:
                    if c.dep_ == "amod" and c.text in ("left", "right"):
                        result["target_location"] = c.text + " side"
            elif pobj_text == "top" and prep_text == "on":
                result["direction"] = "on top"
                result["target_location"] = "top of stack"
            elif prep_text == "into":
                result["direction"] = "into"
                if pobj_text in LOCATION_NOUNS:
                    result["target_location"] = pobj_compound
            elif pobj_text in LOCATION_NOUNS or pobj_compound in LOCATION_NOUNS:
                loc = pobj_compound
                head_verb = token.head
                while head_verb.pos_ not in ("VERB", "AUX") and head_verb.dep_ != "ROOT":
                    head_verb = head_verb.head

                if prep_text in SOURCE_PREPS:
                    source_location = loc
                elif prep_text in TARGET_PREPS and head_verb.text in place_verbs:
                    result["target_location"] = loc
                elif prep_text in TARGET_PREPS and head_verb.text in acquire_verbs:
                    source_location = loc
                elif prep_text in TARGET_PREPS:
                    source_location = loc

    # --- Adverb-based directions ---
    for token in doc:
        if token.dep_ in ("advmod", "npadvmod") and result["direction"] is None:
            if token.text in ("upwards", "downwards"):
                result["direction"] = token.text.replace("wards", "")
            elif token.text in ("left", "right") and token.dep_ == "advmod":
                result["direction"] = token.text

    # --- Fallback: "turn the block right" — "right" parsed as intj or dobj ---
    if result["direction"] is None:
        for token in doc:
            if token.text in ("left", "right") and token.dep_ in ("dobj", "intj"):
                result["direction"] = token.text

    # --- Fallback: orphan prep "up"/"down" with no pobj ---
    if result["direction"] is None:
        for token in doc:
            if token.dep_ == "prep" and token.text in ("up", "down"):
                has_pobj = any(c.dep_ == "pobj" for c in token.children)
                if not has_pobj:
                    result["direction"] = token.text

    # --- Fallback: "left" misparsed as advcl/acomp ---
    if result["direction"] is None:
        for token in doc:
            if token.text in ("left", "right") and token.dep_ in ("advcl", "acomp", "oprd"):
                result["direction"] = token.text

    # --- spaCy fallbacks for CALVIN-specific misparsings ---

    # "turn on/off the X" — X parsed as pobj of on/off
    if result["object"] is None and verb in ("turn on", "turn off"):
        particle = "on" if verb == "turn on" else "off"
        for token in doc:
            if token.text == particle and token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ == "pobj":
                        result["object"] = child.text
                        mods = [c.text for c in child.children
                                if c.dep_ == "amod" and c.text not in DIRECTION_WORDS]
                        if mods:
                            result["object_modifier"] = ";".join(mods)
                        break

    # "grasp the block lying in X" — "block" as nsubj of ccomp
    if result["object"] is None:
        for v in verbs:
            for child in v.children:
                if child.dep_ in ("ccomp", "acl") and child.pos_ == "VERB":
                    for grandchild in child.children:
                        if grandchild.dep_ == "nsubj" and grandchild.pos_ == "NOUN":
                            result["object"] = grandchild.text
                            mods = [c.text for c in grandchild.children
                                    if c.dep_ == "amod" and c.text not in DIRECTION_WORDS]
                            if mods:
                                result["object_modifier"] = ";".join(mods)
                            break

    # "go close the drawer" — "drawer" as npadvmod
    if result["object"] is None:
        for token in doc:
            if token.dep_ == "npadvmod" and token.pos_ == "NOUN":
                result["object"] = token.text
                mods = [c.text for c in token.children
                        if c.dep_ == "amod" and c.text not in DIRECTION_WORDS]
                if mods:
                    result["object_modifier"] = ";".join(mods)
                break

    # "rotate right the blue block" — "block" as pobj of "right"
    if result["object"] is None:
        for token in doc:
            if token.dep_ == "pobj" and token.pos_ == "NOUN":
                if token.head.text in ("right", "left"):
                    result["object"] = token.text
                    result["direction"] = token.head.text
                    mods = [c.text for c in token.children
                            if c.dep_ == "amod" and c.text not in DIRECTION_WORDS]
                    if mods:
                        result["object_modifier"] = ";".join(mods)
                    break

    # --- Merge source_location into object_modifier ---
    if source_location is not None:
        existing = result["object_modifier"]
        if existing:
            result["object_modifier"] = existing + ";in " + source_location
        else:
            result["object_modifier"] = "in " + source_location

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []
    for split, data_dir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        df = load_calvin_to_dataframe(data_dir)

        for _, row in df.iterrows():
            start_idx, end_idx = row["start_idx"], row["end_idx"]
            inst, verb = row["instruction"], row["primary_verb"]

            try:
                ep_s = np.load(os.path.join(data_dir, EPISODE_TEMPLATE.format(start_idx)))
                ep_e = np.load(os.path.join(data_dir, EPISODE_TEMPLATE.format(end_idx)))
            except FileNotFoundError:
                continue

            task_type = classify_task_type(ep_s, ep_e)
            slots = extract_slots(inst, verb)

            rows.append({
                "split": split,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "instruction": inst,
                "task_type": task_type,
                "verb": verb,
                **slots,
            })

        print(f"{split}: {len([r for r in rows if r['split'] == split])} episodes")

    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "episode_task_types.csv")
    out.to_csv(out_path, index=False)
    print(f"\nTotal: {len(out)} episodes")
    print(f"\nTask type distribution:\n{out['task_type'].value_counts()}")
    print(f"\nVerb x task_type:\n{pd.crosstab(out['verb'], out['task_type'])}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
