import json
import os


QUESTION_DIR="./test_data/study_rules"

input=f"{QUESTION_DIR}/questions.txt"
output=f"{QUESTION_DIR}/test_set.json"

def parse_to_lists(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    ground_truths = []

    current_q, current_a = "", ""
    state = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("=== Q"):
            if current_q and current_a:
                questions.append(current_q.strip())
                ground_truths.append(current_a.strip())
                current_q, current_a = "", ""

            state = "question"
            continue

        elif stripped.startswith("=== A"):
            state = "answer"
            continue

        if state == "question":
            current_q += line
        elif state == "answer":
            current_a += line

    if current_q and current_a:
        questions.append(current_q.strip())
        ground_truths.append(current_a.strip())

    return {
        "question": questions,
        "ground_truth": ground_truths,
        "answer": ["" for _ in questions]
    }

if os.path.exists(output):
    print(f"File already exists:  {output} — skipping creation.")
else:
    data = parse_to_lists(input)
    print(data)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
