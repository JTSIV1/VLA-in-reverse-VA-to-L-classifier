"""Run Gemini subtask decomposition on 1K diverse DROID instructions.

Samples instructions stratified by verb, calls Gemini 2.5 Pro to decompose
each into atomic subtasks, saves results to data/droid_decompositions_1k.json.

Usage:
    python scripts/gemini_decompose_1k.py
"""

import google.generativeai as genai
import json
import os
import re
import time
import random
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

OUTPUT_PATH = "data/droid_decompositions_1k.json"
N_SAMPLE = 1000
MAX_RETRIES = 3


def parse_json(text):
    """Extract JSON array from Gemini response."""
    for ch in ["\u201c", "\u201d", "\u2018", "\u2019"]:
        text = text.replace(ch, '"')
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def sample_diverse_instructions(csv_path, n=N_SAMPLE, seed=42):
    """Sample n unique instructions, stratified by verb."""
    df = pd.read_csv(csv_path)
    unique = df.drop_duplicates(subset="instruction")[["instruction", "verb"]]

    # Stratified sample: proportional to verb frequency, min 1 per verb
    random.seed(seed)
    verb_groups = unique.groupby("verb")
    n_verbs = len(verb_groups)
    samples = []

    # First pass: 1 per verb
    for verb, group in verb_groups:
        samples.append(group.sample(1, random_state=seed))

    remaining = n - len(samples)
    if remaining > 0:
        # Second pass: proportional allocation
        pool = unique[~unique.index.isin(pd.concat(samples).index)]
        verb_counts = pool["verb"].value_counts()
        for verb, count in verb_counts.items():
            alloc = max(1, int(remaining * count / len(pool)))
            verb_pool = pool[pool["verb"] == verb]
            take = min(alloc, len(verb_pool))
            if take > 0:
                samples.append(verb_pool.sample(take, random_state=seed))

    result = pd.concat(samples).drop_duplicates(subset="instruction")
    if len(result) > n:
        result = result.sample(n, random_state=seed)
    return result.reset_index(drop=True)


def main():
    csv_path = "data/droid_episodes_filtered.csv"
    print(f"Sampling {N_SAMPLE} diverse instructions...")
    sampled = sample_diverse_instructions(csv_path, N_SAMPLE)
    print(f"Sampled {len(sampled)} instructions across {sampled['verb'].nunique()} verbs")

    # Resume from existing results if any
    results = []
    done_instructions = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        done_instructions = {r["instruction"] for r in results}
        print(f"Resuming: {len(done_instructions)} already done")

    todo = sampled[~sampled["instruction"].isin(done_instructions)]
    print(f"Remaining: {len(todo)} instructions")

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc="Decomposing"):
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

        # Save periodically
        if len(results) % 20 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

        # Rate limit: ~30 RPM for Gemini Pro
        time.sleep(2)

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    n_success = sum(1 for r in results if r["subtasks"])
    n_fail = sum(1 for r in results if not r["subtasks"])
    print(f"\nDone: {n_success} success, {n_fail} failed")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
