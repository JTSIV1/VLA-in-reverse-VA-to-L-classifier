"""
Build L1 segment dataset from Gemini hierarchical annotations.

Each L1 phase becomes a training sample with:
  - action segment: rel_actions[phase_start:phase_end+1]
  - scene_obs segment: scene_obs[phase_start:phase_end+1]
  - verb label: consolidated L1 verb

Consolidation maps near-synonyms to canonical verbs.
Applies min_class_count filtering (default 30).

Usage:
  python scripts/build_l1_dataset.py
  python scripts/build_l1_dataset.py --min_class_count 50

Outputs:
  data/l1_segments/train.jsonl
  data/l1_segments/val.jsonl
  data/l1_segments/label_map.json
  data/l1_segments/stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, VAL_DIR, ACTION_KEY, EPISODE_TEMPLATE

# ── Verb consolidation map ──────────────────────────────────────────────────
# Maps raw Gemini first-word verbs → canonical verbs.
# Verbs not in this map keep their original form.
CONSOLIDATION = {
    # approach family
    "reach": "approach",
    "advance": "approach",
    "navigate": "approach",
    "traverse": "approach",
    # retract family
    "withdraw": "retract",
    # grasp family
    "grip": "grasp",
    "hold": "grasp",
    "secure": "grasp",
    "engage": "grasp",
    "contact": "grasp",
    "hook": "grasp",
    # release family
    "disengage": "release",
    "drop": "release",
    "discard": "release",
    # lift family
    "raise": "lift",
    # lower family
    "descend": "lower",
    # move family
    "transport": "move",
    "translate": "move",
    "carry": "move",
    "relocate": "move",
    "transfer": "move",
    "shift": "move",
    "drag": "move",
    # position family
    "align": "position",
    "reposition": "position",
    "adjust": "position",
    "orient": "position",
    "reorient": "position",
    "realign": "position",
    "readjust": "position",
    "prepare": "position",
    "settle": "position",
    # rotate family
    "twist": "rotate",
    "turn": "rotate",
    "swing": "rotate",
    "tilt": "rotate",
    "tip": "rotate",
    # flip stays flip
    # push stays push
    # pull stays pull
    # slide stays slide
    # press stays press
    # place stays place
    # open stays open
    # close stays close
    # sweep stays sweep
    # insert stays insert
    # stack → place
    "stack": "place",
    "stow": "place",
    # misc rare → skip (will be filtered by min_class_count)
    "nudge": "push",
    "knock": "push",
    "flick": "flip",
}


def get_verb(desc):
    """Extract first word (verb) from step description."""
    words = desc.strip().split()
    return words[0].lower() if words else ""


def consolidate_verb(raw_verb):
    """Map raw verb to canonical form."""
    return CONSOLIDATION.get(raw_verb, raw_verb)


def build_split(annotation_path, data_dir, min_seg_len=3):
    """Parse annotations and build list of (segment_info, verb) tuples."""
    with open(annotation_path) as f:
        episodes = [json.loads(line) for line in f]

    segments = []
    skipped_bad_boundary = 0
    skipped_short = 0

    for ep in episodes:
        ep_idx = ep["episode_index"]
        start_idx = ep["start_idx"]  # global CALVIN frame index
        end_idx = ep["end_idx"]
        n_steps = ep["n_steps"]

        for phase in ep["decomposition"]:
            ps = phase.get("START_TIMESTEP", 0)
            pe = phase.get("END_TIMESTEP", 0)

            # Skip bad boundaries
            if pe < ps:
                skipped_bad_boundary += 1
                continue

            seg_len = pe - ps + 1
            if seg_len < min_seg_len:
                skipped_short += 1
                continue

            raw_verb = get_verb(phase.get("STEP_DESCRIPTION", ""))
            verb = consolidate_verb(raw_verb)

            segments.append({
                "episode_index": ep_idx,
                "global_start": start_idx + ps,  # global CALVIN frame
                "global_end": start_idx + pe,
                "phase_start": ps,  # local within episode
                "phase_end": pe,
                "seg_len": seg_len,
                "verb": verb,
                "description": phase.get("STEP_DESCRIPTION", ""),
                "instruction_gt": ep.get("instruction_gt", ""),
                "instruction_gemini": ep.get("instruction_gemini", ""),
            })

    print("  Segments: {}, skipped bad boundary: {}, skipped short (<{}): {}".format(
        len(segments), skipped_bad_boundary, min_seg_len, skipped_short))
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_class_count", type=int, default=30)
    parser.add_argument("--min_seg_len", type=int, default=3,
                        help="Min segment length in timesteps")
    parser.add_argument("--output_dir", type=str, default="data/l1_segments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building L1 segment dataset...")
    print("  min_class_count={}, min_seg_len={}".format(
        args.min_class_count, args.min_seg_len))

    # Build segments
    print("\nTraining split:")
    train_segs = build_split(
        "data/hierarchy_annotations/calvin_training.jsonl",
        TRAIN_DIR, min_seg_len=args.min_seg_len)

    print("\nValidation split:")
    val_segs = build_split(
        "data/hierarchy_annotations/calvin_validation.jsonl",
        VAL_DIR, min_seg_len=args.min_seg_len)

    # Verb distribution before filtering
    train_vc = Counter(s["verb"] for s in train_segs)
    print("\nConsolidated verb counts (training, before class filter):")
    for v, c in train_vc.most_common():
        print("  {:<20} {:>5}".format(v, c))
    print("  Unique:", len(train_vc))

    # Filter by min_class_count (based on training set)
    keep_verbs = {v for v, c in train_vc.items() if c >= args.min_class_count}
    print("\nAfter min_class_count={}: {} classes kept".format(
        args.min_class_count, len(keep_verbs)))
    removed = {v: c for v, c in train_vc.items() if c < args.min_class_count}
    if removed:
        print("  Removed:", dict(sorted(removed.items(), key=lambda x: -x[1])))

    train_segs = [s for s in train_segs if s["verb"] in keep_verbs]
    val_segs = [s for s in val_segs if s["verb"] in keep_verbs]

    # Build label map (sorted alphabetically)
    sorted_verbs = sorted(keep_verbs)
    verb2id = {v: i for i, v in enumerate(sorted_verbs)}

    # Add label ids
    for s in train_segs:
        s["label"] = verb2id[s["verb"]]
    for s in val_segs:
        s["label"] = verb2id[s["verb"]]

    # Final stats
    train_vc_final = Counter(s["verb"] for s in train_segs)
    val_vc_final = Counter(s["verb"] for s in val_segs)
    print("\nFinal dataset:")
    print("  Train: {} segments, {} classes".format(len(train_segs), len(train_vc_final)))
    print("  Val:   {} segments, {} classes".format(len(val_segs), len(val_vc_final)))
    print("\n  {:<20} {:>6} {:>6}".format("Verb", "Train", "Val"))
    print("  " + "-" * 35)
    for v in sorted_verbs:
        print("  {:<20} {:>6} {:>6}".format(v, train_vc_final.get(v, 0), val_vc_final.get(v, 0)))

    # Segment length stats
    train_lens = np.array([s["seg_len"] for s in train_segs])
    print("\n  Segment lengths: mean={:.1f}, median={}, min={}, max={}".format(
        train_lens.mean(), int(np.median(train_lens)), train_lens.min(), train_lens.max()))

    # Save
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    label_path = os.path.join(args.output_dir, "label_map.json")
    stats_path = os.path.join(args.output_dir, "stats.json")

    with open(train_path, "w") as f:
        for s in train_segs:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val_segs:
            f.write(json.dumps(s) + "\n")

    with open(label_path, "w") as f:
        json.dump({"verb2id": verb2id, "id2verb": {i: v for v, i in verb2id.items()}}, f, indent=2)

    stats = {
        "n_train": len(train_segs),
        "n_val": len(val_segs),
        "n_classes": len(sorted_verbs),
        "classes": sorted_verbs,
        "min_class_count": args.min_class_count,
        "min_seg_len": args.min_seg_len,
        "train_class_counts": dict(train_vc_final.most_common()),
        "val_class_counts": dict(val_vc_final.most_common()),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
