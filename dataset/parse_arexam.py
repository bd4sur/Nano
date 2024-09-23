import json

all_qa = []

with open("sft-amateur-radio.txt", mode="r", encoding="utf-8") as f:
    current_question = ""
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("[Q]"):
            current_question = line[3:]
        elif line.startswith("[A]"):
            answer = line[3:]
            item = {"question": current_question, "answer": answer}
            all_qa.append(item)
            current_question = ""

with open("sft-amateur-radio.jsonl", mode="w", encoding="utf-8") as f:
    for item in all_qa:
        f.writelines(json.dumps(item) + "\n")
