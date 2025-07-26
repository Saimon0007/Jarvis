"""
Step 2 (Reasoning): Preprocess GSM8K and StrategyQA datasets into chain-of-thought prompt-response pairs.
"""
from datasets import load_from_disk
import json
import os

def preprocess_gsm8k():
    data_dir = "finetune/data/gsm8k_sample"
    out_path = "finetune/data/gsm8k_cot_pairs.jsonl"
    dataset = load_from_disk(data_dir)
    pairs = []
    for item in dataset:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        # Chain-of-thought prompt
        prompt = f"Let's break this down step-by-step: {question}"
        response = answer
        pairs.append({"prompt": prompt, "response": response})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} GSM8K pairs to {out_path}")

def preprocess_strategyqa():
    data_dir = "finetune/data/strategyqa_sample"
    out_path = "finetune/data/strategyqa_cot_pairs.jsonl"
    dataset = load_from_disk(data_dir)
    pairs = []
    for item in dataset:
        question = item.get("question", "").strip()
        # Use the "facts" field as reasoning steps if available
        facts = item.get("facts", [])
        answer = str(item.get("answer", "")).strip()
        cot = " ".join(facts) if facts else ""
        prompt = f"Let's break this down step-by-step: {question}"
        response = f"{cot}\nFinal answer: {answer}"
        pairs.append({"prompt": prompt, "response": response})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} StrategyQA pairs to {out_path}")

if __name__ == "__main__":
    preprocess_gsm8k()
    preprocess_strategyqa()
