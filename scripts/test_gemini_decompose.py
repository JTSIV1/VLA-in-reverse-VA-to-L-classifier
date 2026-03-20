"""Test Gemini subtask decomposition on 5 sample instructions."""

import google.generativeai as genai
import json
import os
import re
import time

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

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


def parse_json(text):
    """Extract JSON array from Gemini response, handling markdown fences and smart quotes."""
    # Replace all Unicode quote variants with ASCII
    for ch in ["\u201c", "\u201d", "\u2018", "\u2019", "\u00ab", "\u00bb"]:
        text = text.replace(ch, '"')
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    # Find and parse the JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


test_instructions = [
    "Put the marker in the mug",
    "Close the laptop lid",
    "Fold the towel in half",
    "Pour water from the bottle into the cup",
    "Hang the cloth on the clear screen",
]

for inst in test_instructions:
    print(f"=== {inst} ===", flush=True)
    try:
        resp = model.generate_content(PROMPT_TEMPLATE.format(instruction=inst))
        print(f"  RAW: {repr(resp.text[:300])}", flush=True)
        subtasks = parse_json(resp.text)
        for s in subtasks:
            print(f"  {s['step']}. {s['subtask']}", flush=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
    print(flush=True)
    time.sleep(1)
