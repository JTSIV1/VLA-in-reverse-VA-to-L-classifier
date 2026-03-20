"""Run Gemini subtask decomposition on ALL unique DROID instructions.

Supports SLURM array parallelism: each task handles a slice of instructions.
Set SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT env vars, or run standalone.

Usage:
    python scripts/gemini_decompose_all.py                    # all instructions
    SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=10 \
        python scripts/gemini_decompose_all.py                # shard 0 of 10
"""

import google.generativeai as genai
import json
import os
import re
import time
import pandas as pd
from tqdm import tqdm

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

PROMPT_TEMPLATE = (
    "You are helping build a verb vocabulary for language grounding research "
    "on robot manipulation. Given a high-level task instruction, decompose it "
    "into the ordered sequence of atomic subtasks a person would naturally "
    "describe when watching the robot perform the task. Each subtask should "
    "represent a single, indivisible physical action -- not a combination of "
    "actions.\n\n"
    "Guidelines:\n"
    "- Each subtask should use a single, common English verb that describes "
    "what the robot is physically doing\n"
    "- Use verbs that a non-expert would naturally use to describe what they see\n"
    "- Include all phases of the task, including preparatory and transitional "
    "motions, not just the main action\n"
    '- Format each subtask as: "<verb> <object/target>"\n'
    "- Typically 3-7 subtasks per instruction\n\n"
    "Respond with ONLY a JSON array:\n"
    '[{{"step": 1, "subtask": "..."}}, {{"step": 2, "subtask": "..."}}]\n\n'
    'Decompose this instruction: "{instruction}"'
)

CSV_PATH = "data/droid_episodes_filtered.csv"
OUTPUT_DIR = "data/droid_decompositions"
MAX_RETRIES = 3


def parse_json(text):
    for ch in ["\u201c", "\u201d", "\u2018", "\u2019"]:
        text = text.replace(ch, '"')
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def main():
    # Get all unique instructions with their verbs
    df = pd.read_csv(CSV_PATH)
    unique = df.drop_duplicates(subset="instruction")[["instruction", "verb"]].reset_index(drop=True)
    print(f"Total unique instructions: {len(unique)}")

    # SLURM array sharding
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    shard_size = (len(unique) + task_count - 1) // task_count
    start = task_id * shard_size
    end = min(start + shard_size, len(unique))
    shard = unique.iloc[start:end].reset_index(drop=True)

    print(f"Task {task_id}/{task_count}: instructions {start}-{end-1} ({len(shard)} total)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"shard_{task_id:03d}.json")

    # Resume from existing results
    results = []
    done_instructions = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_instructions = {r["instruction"] for r in results}
        print(f"Resuming: {len(done_instructions)} already done")

    todo = shard[~shard["instruction"].isin(done_instructions)]
    print(f"Remaining: {len(todo)} instructions")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc=f"Shard {task_id}"):
        inst = row["instruction"]
        verb = row["verb"]

        for attempt in range(MAX_RETRIES):
            try:
                resp = model.generate_content(
                    PROMPT_TEMPLATE.format(instruction=inst)
                )
                subtasks = parse_json(resp.text)
                results.append({
                    "instruction": inst,
                    "verb": verb,
                    "subtasks": [s["subtask"] for s in subtasks],
                })
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    results.append({
                        "instruction": inst,
                        "verb": verb,
                        "subtasks": [],
                        "error": str(e),
                    })

        # Save every 50
        if len(results) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        time.sleep(1)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    n_success = sum(1 for r in results if r["subtasks"])
    n_fail = sum(1 for r in results if not r["subtasks"])
    print(f"\nDone: {n_success} success, {n_fail} failed")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
