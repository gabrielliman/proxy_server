import os
import json

DATA_DIR = "/scratch/global/datasets/juan"
OUTPUT = "programs_dataset.json"

# safe slice of the text for models with ~18k token limit
MAX_CHARS = 7000   # ~5k tokens


def make_prompts(text):
    # only use a safe substring of the text
    snippet = text[:MAX_CHARS]

    prompts = [
        {"from": "human", "value": f"Summarize the following text:\n\n{snippet}"},
        {"from": "human", "value": f"List the key themes present in the text:\n\n{snippet}"},
        {"from": "human", "value": f"Identify the most important characters in the text:\n\n{snippet}"},
        {"from": "human", "value": f"What is the central conflict in the text?\n\n{snippet}"},
        {"from": "human", "value": f"Rewrite the opening paragraph in simpler language:\n\n{snippet}"}
    ]

    # ensure exactly 30 dependent prompts
    while len(prompts) < 30:
        step = len(prompts) + 1
        prompts.append({
            "from": "human",
            "value": f"Step {step}: Based on everything above, continue the analysis using only the same snippet:\n\n{snippet}"
        })

    return prompts


programs = []

for filename in sorted(os.listdir(DATA_DIR)):
    if filename.endswith(".txt"):
        path = os.path.join(DATA_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        programs.append({
            "id": filename.replace(".txt", ""),
            "conversations": make_prompts(text)
        })

with open(OUTPUT, "w", encoding="utf-8") as out:
    json.dump(programs, out, indent=2)

print(f"Saved dataset to {OUTPUT}")
