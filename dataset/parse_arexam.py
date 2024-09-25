import json

def ar_sft():
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
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n")

def general_jsonl():
    all_text = []
    with open("pretrain-general.jsonl", mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            obj = json.loads(line)
            all_text.append(obj["text"])
    with open("pretrain-general.txt", mode="w", encoding="utf-8") as f:
        for item in all_text:
            f.writelines("<|bos|>" + item + "<|eos|>\n")

# ar_sft()
# general_jsonl()
