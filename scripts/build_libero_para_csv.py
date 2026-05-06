"""Build a CSV of all LIBERO-Para paraphrased instructions, joined with their
original LIBERO-Goal instructions.

Reuses the BDDL-parsing helpers from
LIBERO-Para/eval_scripts/examples/eval_openvla_oft.py:
    parse_bddl_instruction(path) → instruction text from the (:language ...) line
    parse_bddl_filename(path)    → paraphrase_type, categories, eval_id, variant_id
    extract_eval_id(filename)    → eval_id (0-9, one per LIBERO-Goal task)

Output: data/libero_para_instructions.csv with columns
    bddl_file, paraphrase_type, categories, subcategories, eval_id, variant_id,
    paraphrased_instruction, task_name, original_instruction

Run:
    python scripts/build_libero_para_csv.py
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

LIBERO_REPO = Path("/data/user_data/wenjiel2/Code/LIBERO-Para")
sys.path.insert(0, str(LIBERO_REPO / "eval_scripts" / "examples"))


def parse_bddl_instruction(bddl_path: str) -> str:
    """Extract the (:language ...) line from a BDDL file."""
    with open(bddl_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("(:language"):
                return line.replace("(:language", "").rstrip(")").strip()
    return ""


KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}


def _split_category_subcategory(body: str):
    for known_cat in KNOWN_CATEGORIES:
        if body.startswith(known_cat + "_"):
            return known_cat, body[len(known_cat) + 1:]
    return body, ""


def parse_bddl_filename(filename: str) -> dict:
    """Decode the paraphrase metadata encoded in a libero_para BDDL filename.

    Files look like  obj_lexical_synonym_eval3_ver7.bddl
                     act_pragmatical_addition_eval0_ver12.bddl
                     comp_obj+act_lexical_synonym+lexical_synonym_eval5_ver2.bddl
    """
    base = os.path.basename(filename).replace(".bddl", "")
    if "_eval" not in base:
        return {"paraphrase_type": "unknown", "categories": [],
                "subcategories": [], "eval_id": -1, "variant_id": -1}

    prefix_part, eval_ver_part = base.rsplit("_eval", 1)
    eval_str, ver_str = eval_ver_part.split("_ver")
    eval_id = int(eval_str)
    variant_id = int(ver_str)

    if prefix_part.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix_part[5:]
        first_plus = body.index("+")
        cat1 = body[:first_plus]
        remainder = body[first_plus + 1:]
        cat2, subcat1, subcat2 = None, None, None
        for known_cat in KNOWN_CATEGORIES:
            if remainder.startswith(known_cat + "_"):
                cat2 = known_cat
                after_cat2 = remainder[len(known_cat) + 1:]
                subcat1, subcat2 = after_cat2.rsplit("+", 1)
                break
        if cat2 is None:
            categories, subcategories = [body], [body]
        else:
            categories, subcategories = [cat1, cat2], [subcat1, subcat2]
    elif prefix_part.startswith("act_"):
        paraphrase_type = "act"
        cat, subcat = _split_category_subcategory(prefix_part[4:])
        categories, subcategories = [cat], [subcat]
    elif prefix_part.startswith("obj_"):
        paraphrase_type = "obj"
        cat, subcat = _split_category_subcategory(prefix_part[4:])
        categories, subcategories = [cat], [subcat]
    else:
        paraphrase_type = "unknown"
        categories, subcategories = [], []

    return {"paraphrase_type": paraphrase_type, "categories": categories,
            "subcategories": subcategories, "eval_id": eval_id,
            "variant_id": variant_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--para_dir",
                        default=str(LIBERO_REPO / "libero/libero/bddl_files/libero_para"))
    parser.add_argument("--goal_dir",
                        default=str(LIBERO_REPO / "libero/libero/bddl_files/libero_goal"))
    parser.add_argument("--out_csv",
                        default="data/libero_para_instructions.csv")
    args = parser.parse_args()

    # ---- 1. Build eval_id → (task_name, original_instruction) lookup ----
    goal_files = sorted(Path(args.goal_dir).glob("*.bddl"))
    print(f"Reading {len(goal_files)} original LIBERO-Goal BDDLs from {args.goal_dir}")
    eval_id_to_orig = {}
    for i, p in enumerate(goal_files):
        instr = parse_bddl_instruction(str(p))
        task_name = p.stem
        eval_id_to_orig[i] = (task_name, instr)
        print(f"  eval_id={i:2d}  {task_name}  →  {instr!r}")

    # ---- 2. Walk paraphrased BDDLs ----
    para_files = sorted(Path(args.para_dir).glob("*.bddl"))
    print(f"\nReading {len(para_files)} paraphrased BDDLs from {args.para_dir}")

    rows = []
    for p in para_files:
        meta = parse_bddl_filename(p.name)
        instr = parse_bddl_instruction(str(p))
        eid = meta["eval_id"]
        task_name, orig_instr = eval_id_to_orig.get(eid, ("", ""))
        rows.append({
            "bddl_file":               p.name,
            "paraphrase_type":         meta["paraphrase_type"],
            "categories":              "|".join(meta["categories"]),
            "subcategories":           "|".join(meta["subcategories"]),
            "eval_id":                 eid,
            "variant_id":              meta["variant_id"],
            "paraphrased_instruction": instr,
            "task_name":               task_name,
            "original_instruction":    orig_instr,
        })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 3. Stats ----
    import collections
    pt = collections.Counter(r["paraphrase_type"] for r in rows)
    per_eval = collections.Counter(r["eval_id"] for r in rows)
    print(f"\nWrote {len(rows)} rows → {args.out_csv}")
    print(f"  paraphrase_type counts: {dict(pt)}")
    print(f"  per-eval_id counts: "
          f"min={min(per_eval.values())}, max={max(per_eval.values())}, "
          f"mean={sum(per_eval.values()) / len(per_eval):.1f}")


if __name__ == "__main__":
    main()
