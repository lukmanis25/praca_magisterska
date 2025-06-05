import json
import os
import sys
from time import sleep
import asyncio

sys.path.append(".")
from query_naive import query, init_rag

INPUT_FILE = "../evaluation/test_data/study_rules/test_set.json"
OUTPUT_FILE = "../evaluation/test_data/study_rules/test_set_naive_rag.json"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def run():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: input file not found at {INPUT_FILE}")
        return

    input_data = load_json(INPUT_FILE)

    rag = await init_rag()
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing output file: {OUTPUT_FILE} — resuming.")
        output_data = load_json(OUTPUT_FILE)
    else:
        print("Creating new output file.")
        output_data = {
            "question": input_data["question"],
            "ground_truth": input_data["ground_truth"],
            "answer": ["" for _ in input_data["question"]]
        }

    for i, q in enumerate(output_data["question"]):
        if output_data["answer"][i].strip():
            print(f"[{i+1}] SKIPPED — already answered.")
            continue

        print(f"[{i+1}] Sending query...")
        try:
            answer = await query(rag, q)
        except Exception as e:
            print(f"[{i+1}] ❌ Error during async query: {e}")
            break

        if answer is None:
            print(f"[{i+1}] ❌ Query failed — no answer returned.")
            continue

        output_data["answer"][i] = answer.strip()
        save_json(output_data, OUTPUT_FILE)
        print(f"[{i+1}] ✅ Answer saved.")

        # sleep(1)

    print("✔️ All done.")

if __name__ == "__main__":
    asyncio.run(run())
