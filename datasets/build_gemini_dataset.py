"""
Build verb-labeled datasets from Gemini hierarchical annotations.

Supports two granularity levels:
  L0 — full-episode verb labels (from Gemini's TASK_INSTRUCTION)
  L1 — sub-phase verb labels (from Gemini's DECOMPOSITION phases)

Both levels apply verb consolidation (near-synonym mapping) and
min_class_count filtering.

Usage:
  python datasets/build_gemini_dataset.py --level l0
  python datasets/build_gemini_dataset.py --level l1
  python datasets/build_gemini_dataset.py --level both
  python datasets/build_gemini_dataset.py --level l1 --min_class_count 50

Outputs:
  data/gemini_l0_segments/{train,val}.jsonl, label_map.json, stats.json
  data/l1_segments/{train,val}.jsonl, label_map.json, stats.json
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

# ── Verb consolidation maps ─────────────────────────────────────────────────

# L0: full-episode verbs extracted via spaCy from Gemini task instruction
L0_CONSOLIDATION = {
    "go": None,       # discourse verb, skip
    "let": None,
    "use": None,       # too vague
    "closed": "close",
    "stand": "place",  # "stand the block up"
    "unstack": "remove",
    "drop": "release",
    "take": "take off",
    "push off": "push",
    "flip down": "flip",
}

# L1: sub-phase verbs (first word of STEP_DESCRIPTION)
L1_CONSOLIDATION = {
    # approach family
    "reach": "approach", "advance": "approach", "navigate": "approach",
    "traverse": "approach",
    # retract family
    "withdraw": "retract",
    # grasp family
    "grip": "grasp", "hold": "grasp", "secure": "grasp", "engage": "grasp",
    "contact": "grasp", "hook": "grasp",
    # release family
    "disengage": "release", "drop": "release", "discard": "release",
    # lift family
    "raise": "lift",
    # lower family
    "descend": "lower",
    # move family
    "transport": "move", "translate": "move", "carry": "move",
    "relocate": "move", "transfer": "move", "shift": "move", "drag": "move",
    # position family
    "align": "position", "reposition": "position", "adjust": "position",
    "orient": "position", "reorient": "position", "realign": "position",
    "readjust": "position", "prepare": "position", "settle": "position",
    # rotate family
    "twist": "rotate", "turn": "rotate", "swing": "rotate",
    "tilt": "rotate", "tip": "rotate",
    # place family
    "stack": "place", "stow": "place",
    # push family
    "nudge": "push", "knock": "push",
    # flip family
    "flick": "flip",
}

nlp = None  # lazy-loaded for L0 only


def _get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp


# ── L0: full-episode verb extraction ────────────────────────────────────────

def extract_first_verb_spacy(text):
    """Extract the primary verb (with particle) from a sentence using spaCy."""
    doc = _get_nlp()(text.lower().strip().rstrip("."))
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj", "xcomp", "advcl"):
            parts = [t.text for t in token.children if t.dep_ == "prt"]
            return " ".join([token.text] + parts)
    words = text.strip().split()
    return words[0].lower() if words else ""


def build_l0(ann_dir, min_class_count):
    """Build full-episode (L0) verb dataset from Gemini annotations."""
    all_segs = {}
    for split in ["training", "validation"]:
        ann_path = os.path.join(ann_dir, "calvin_{}.jsonl".format(split))
        with open(ann_path) as f:
            episodes = [json.loads(line) for line in f]
        print("{}: {} episodes".format(split, len(episodes)))

        segments = []
        skipped = 0
        for ep in episodes:
            raw_verb = extract_first_verb_spacy(ep.get("instruction_gemini", ""))
            verb = L0_CONSOLIDATION.get(raw_verb, raw_verb)
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
        key = "train" if split == "training" else "val"
        all_segs[key] = segments

    return _filter_and_save(all_segs, min_class_count, "data/gemini_l0_segments",
                            level="L0")


def _get_first_word(desc):
    """Extract first word (verb) from step description."""
    words = desc.strip().split()
    return words[0].lower() if words else ""


# ── L1: sub-phase verb extraction ───────────────────────────────────────────

def build_l1(ann_dir, min_class_count, min_seg_len=3):
    """Build sub-phase (L1) verb dataset from Gemini annotations."""
    all_segs = {}
    for split in ["training", "validation"]:
        ann_path = os.path.join(ann_dir, "calvin_{}.jsonl".format(split))
        with open(ann_path) as f:
            episodes = [json.loads(line) for line in f]

        print("{}: {} episodes".format(split, len(episodes)))
        segments = []
        skipped_bad = 0
        skipped_short = 0

        for ep in episodes:
            ep_idx = ep["episode_index"]
            start_idx = ep["start_idx"]

            for phase in ep["decomposition"]:
                ps = phase.get("START_TIMESTEP", 0)
                pe = phase.get("END_TIMESTEP", 0)
                if pe < ps:
                    skipped_bad += 1
                    continue
                seg_len = pe - ps + 1
                if seg_len < min_seg_len:
                    skipped_short += 1
                    continue

                raw_verb = _get_first_word(phase.get("STEP_DESCRIPTION", ""))
                verb = L1_CONSOLIDATION.get(raw_verb, raw_verb)

                segments.append({
                    "episode_index": ep_idx,
                    "global_start": start_idx + ps,
                    "global_end": start_idx + pe,
                    "phase_start": ps,
                    "phase_end": pe,
                    "seg_len": seg_len,
                    "verb": verb,
                    "description": phase.get("STEP_DESCRIPTION", ""),
                    "instruction_gt": ep.get("instruction_gt", ""),
                    "instruction_gemini": ep.get("instruction_gemini", ""),
                })

        print("  Segments: {}, skipped bad boundary: {}, skipped short (<{}): {}".format(
            len(segments), skipped_bad, min_seg_len, skipped_short))
        key = "train" if split == "training" else "val"
        all_segs[key] = segments

    return _filter_and_save(all_segs, min_class_count, "data/l1_segments",
                            level="L1", min_seg_len=min_seg_len)


# ── Shared filtering and saving ─────────────────────────────────────────────

def _filter_and_save(all_segs, min_class_count, output_dir, level="L0",
                     min_seg_len=None):
    """Filter by min_class_count, build label map, save JSONL + stats."""
    os.makedirs(output_dir, exist_ok=True)

    train_segs = all_segs["train"]
    val_segs = all_segs["val"]

    train_vc = Counter(s["verb"] for s in train_segs)
    print("\n{} verb counts (training, before filter):".format(level))
    for v, c in train_vc.most_common():
        print("  {:<20} {:>5}".format(v, c))
    print("  Unique:", len(train_vc))

    keep_verbs = {v for v, c in train_vc.items() if c >= min_class_count}
    print("\nAfter min_class_count={}: {} classes kept".format(
        min_class_count, len(keep_verbs)))
    removed = {v: c for v, c in train_vc.items() if c < min_class_count}
    if removed:
        print("  Removed:", dict(sorted(removed.items(), key=lambda x: -x[1])))

    train_segs = [s for s in train_segs if s["verb"] in keep_verbs]
    val_segs = [s for s in val_segs if s["verb"] in keep_verbs]

    sorted_verbs = sorted(keep_verbs)
    verb2id = {v: i for i, v in enumerate(sorted_verbs)}
    for s in train_segs + val_segs:
        s["label"] = verb2id[s["verb"]]

    train_vc_final = Counter(s["verb"] for s in train_segs)
    val_vc_final = Counter(s["verb"] for s in val_segs)
    print("\nFinal {} dataset:".format(level))
    print("  Train: {} segments, {} classes".format(len(train_segs), len(train_vc_final)))
    print("  Val:   {} segments, {} classes".format(len(val_segs), len(val_vc_final)))
    print("\n  {:<20} {:>6} {:>6}".format("Verb", "Train", "Val"))
    print("  " + "-" * 35)
    for v in sorted_verbs:
        print("  {:<20} {:>6} {:>6}".format(v, train_vc_final.get(v, 0), val_vc_final.get(v, 0)))

    if level == "L1":
        train_lens = np.array([s["seg_len"] for s in train_segs])
        print("\n  Segment lengths: mean={:.1f}, median={}, min={}, max={}".format(
            train_lens.mean(), int(np.median(train_lens)),
            train_lens.min(), train_lens.max()))

    # Save
    for name, segs in [("train", train_segs), ("val", val_segs)]:
        path = os.path.join(output_dir, "{}.jsonl".format(name))
        with open(path, "w") as f:
            for s in segs:
                f.write(json.dumps(s) + "\n")

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"verb2id": verb2id,
                    "id2verb": {i: v for v, i in verb2id.items()}}, f, indent=2)

    stats = {
        "n_train": len(train_segs),
        "n_val": len(val_segs),
        "n_classes": len(sorted_verbs),
        "classes": sorted_verbs,
        "min_class_count": min_class_count,
        "train_class_counts": dict(train_vc_final.most_common()),
        "val_class_counts": dict(val_vc_final.most_common()),
    }
    if min_seg_len is not None:
        stats["min_seg_len"] = min_seg_len

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved to {}".format(output_dir))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build verb-labeled datasets from Gemini annotations.")
    parser.add_argument("--level", choices=["l0", "l1", "both"], default="both",
                        help="Which granularity to build (default: both)")
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--min_seg_len", type=int, default=3,
                        help="Min segment length in timesteps (L1 only)")
    parser.add_argument("--ann_dir", type=str,
                        default="data/hierarchy_annotations",
                        help="Directory with Gemini annotation JSONL files")
    args = parser.parse_args()

    if args.level in ("l0", "both"):
        print("=" * 60)
        print("Building L0 (full-episode) dataset")
        print("=" * 60)
        build_l0(args.ann_dir, args.min_class_count)

    if args.level in ("l1", "both"):
        print("\n" + "=" * 60)
        print("Building L1 (sub-phase) dataset")
        print("=" * 60)
        build_l1(args.ann_dir, args.min_class_count, args.min_seg_len)


if __name__ == "__main__":
    main()
