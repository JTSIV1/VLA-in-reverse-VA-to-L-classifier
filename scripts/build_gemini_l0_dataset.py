"""
Build Gemini L0 dataset: full episodes labeled with Gemini-inferred task verbs.

Extracts the primary verb from Gemini's TASK_INSTRUCTION using spaCy,
consolidates near-synonyms, and filters by min_class_count.

Usage:
  python scripts/build_gemini_l0_dataset.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

nlp = spacy.load("en_core_web_sm")

# Consolidation for Gemini L0 verbs
CONSOLIDATION = {
    # "go" is a discourse verb in CALVIN
    "go": None,  # skip
    "let": None,
    # closed → close
    "closed": "close",
    # stand → place (as in "stand the block up")
    "stand": "place",
    # unstack → remove
    "unstack": "remove",
    # drop → release
    "drop": "release",
    # use → skip (too vague)
    "use": None,
    # take → take off (usually "take X off")
    "take": "take off",
    # push off → push
    "push off": "push",
    # flip down → flip
    "flip down": "flip",
}


def extract_first_verb(text):
    """Extract the primary verb (with particle) from a sentence using spaCy."""
    doc = nlp(text.lower().strip().rstrip("."))
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj", "xcomp", "advcl"):
            parts = [t.text for t in token.children if t.dep_ == "prt"]
            return " ".join([token.text] + parts)
    # fallback: first word
    words = text.strip().split()
    return words[0].lower() if words else ""


def consolidate_verb(raw_verb):
    """Map raw verb to canonical form. Returns None to skip."""
    if raw_verb in CONSOLIDATION:
        return CONSOLIDATION[raw_verb]
    return raw_verb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="data/gemini_l0_segments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["training", "validation"]:
        ann_path = "data/hierarchy_annotations/calvin_{}.jsonl".format(split)
        with open(ann_path) as f:
            episodes = [json.loads(line) for line in f]
        print("{}: {} episodes".format(split, len(episodes)))

        segments = []
        skipped = 0
        for ep in episodes:
            raw_verb = extract_first_verb(ep.get("instruction_gemini", ""))
            verb = consolidate_verb(raw_verb)
            if verb is None:
                skipped += 1
                continue

            segments.append({
                "episode_index": ep["episode_index"],
                "global_start": ep["start_idx"],
                "global_end": ep["end_idx"],
                "phase_start": 0,
                "phase_end": ep["n_steps"] - 1,
                "seg_len": ep["n_steps"],
                "verb": verb,
                "description": ep.get("instruction_gemini", ""),
                "instruction_gt": ep.get("instruction_gt", ""),
                "instruction_gemini": ep.get("instruction_gemini", ""),
            })

        print("  Kept: {}, Skipped (None verbs): {}".format(len(segments), skipped))

        if split == "training":
            train_segs = segments
        else:
            val_segs = segments

    # Verb distribution before filtering
    train_vc = Counter(s["verb"] for s in train_segs)
    print("\nGemini L0 verb counts (training, before filter):")
    for v, c in train_vc.most_common():
        print("  {:<20} {:>5}".format(v, c))
    print("  Unique:", len(train_vc))

    # Filter by min_class_count
    keep_verbs = {v for v, c in train_vc.items() if c >= args.min_class_count}
    print("\nAfter min_class_count={}: {} classes kept".format(
        args.min_class_count, len(keep_verbs)))
    removed = {v: c for v, c in train_vc.items() if c < args.min_class_count}
    if removed:
        print("  Removed:", dict(sorted(removed.items(), key=lambda x: -x[1])))

    train_segs = [s for s in train_segs if s["verb"] in keep_verbs]
    val_segs = [s for s in val_segs if s["verb"] in keep_verbs]

    # Build label map
    sorted_verbs = sorted(keep_verbs)
    verb2id = {v: i for i, v in enumerate(sorted_verbs)}
    for s in train_segs:
        s["label"] = verb2id[s["verb"]]
    for s in val_segs:
        s["label"] = verb2id[s["verb"]]

    # Final stats
    train_vc_final = Counter(s["verb"] for s in train_segs)
    val_vc_final = Counter(s["verb"] for s in val_segs)
    print("\nFinal dataset:")
    print("  Train: {} episodes, {} classes".format(len(train_segs), len(train_vc_final)))
    print("  Val:   {} episodes, {} classes".format(len(val_segs), len(val_vc_final)))
    print("\n  {:<20} {:>6} {:>6}".format("Verb", "Train", "Val"))
    print("  " + "-" * 35)
    for v in sorted_verbs:
        print("  {:<20} {:>6} {:>6}".format(v, train_vc_final.get(v, 0), val_vc_final.get(v, 0)))

    # Save
    for name, segs in [("train", train_segs), ("val", val_segs)]:
        path = os.path.join(args.output_dir, "{}.jsonl".format(name))
        with open(path, "w") as f:
            for s in segs:
                f.write(json.dumps(s) + "\n")

    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"verb2id": verb2id, "id2verb": {i: v for v, i in verb2id.items()}}, f, indent=2)

    stats = {
        "n_train": len(train_segs),
        "n_val": len(val_segs),
        "n_classes": len(sorted_verbs),
        "classes": sorted_verbs,
        "min_class_count": args.min_class_count,
        "train_class_counts": dict(train_vc_final.most_common()),
        "val_class_counts": dict(val_vc_final.most_common()),
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved to", args.output_dir)


if __name__ == "__main__":
    main()
