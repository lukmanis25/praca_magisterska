import subprocess
import asyncio

async def query(question: str) -> str:
    result = subprocess.run([
        "graphrag", "query",
        "--root", ".", 
        "--method", "drift",
        "--query", question
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error: {result.stderr}")

    lines = result.stdout.splitlines()
    answer_lines = []
    # result=''
    # with open("test_output.txt", "r", encoding="utf-8") as f:
    #     result = f.read()
    # lines = result.splitlines()
    # answer_lines = []

    skipping_info_block = False
    brace_balance = 0

    for line in lines:
        if line.startswith("INFO:"):
            skipping_info_block = True
            brace_balance = line.count("{") - line.count("}")
            continue

        if skipping_info_block:
            brace_balance += line.count("{") - line.count("}")
            if brace_balance <= 0:
                skipping_info_block = False
            continue

        if line.startswith("SUCCESS:"):
            continue

        answer_lines.append(line)

    answer_text = "\n".join(answer_lines).strip()
    return answer_text

# if __name__ == "__main__":
#     question = "What types of training are students required to complete during the first semester of studies at Gdańsk University of Technology?"
#     answer = query(question)

#     with open("test_output3.txt", "w", encoding="utf-8") as f:
#         f.write(answer)

#     print("Czysta odpowiedź została zapisana do output.txt.")
