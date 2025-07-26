"""
Step 2: Preprocess the OpenAssistant dataset into prompt-response pairs for fine-tuning.
"""
from datasets import load_from_disk
import json
import os

def preprocess():
    data_dir = "finetune/data/openassistant_sample"
    out_path = "finetune/data/finetune_pairs.jsonl"
    dataset = load_from_disk(data_dir)
    pairs = []
    for item in dataset:
        if item.get("role") == "prompter" and "text" in item:
            prompt = item["text"]
            # Find the next assistant reply
            idx = item["message_id"]
            for reply in dataset:
                if reply.get("parent_id") == idx and reply.get("role") == "assistant":
                    response = reply["text"]
                    pairs.append({"prompt": prompt, "response": response})
                    break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} prompt-response pairs to {out_path}")

if __name__ == "__main__":
    preprocess()
