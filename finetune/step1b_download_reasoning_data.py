"""
Step 1 (Reasoning): Download GSM8K and StrategyQA datasets from Hugging Face for reasoning fine-tuning.
"""
from datasets import load_dataset
import os

def download_reasoning_datasets():
    # Download GSM8K (math reasoning)
    gsm8k = load_dataset("gsm8k", "main", split="train[:1000]")
    gsm8k.save_to_disk("finetune/data/gsm8k_sample")
    print("GSM8K dataset downloaded and saved to finetune/data/gsm8k_sample")

    # Download StrategyQA (logical reasoning)
    strategyqa = load_dataset("tasksource/strategy-qa", split="train[:1000]")
    strategyqa.save_to_disk("finetune/data/strategyqa_sample")
    print("StrategyQA dataset downloaded and saved to finetune/data/strategyqa_sample")

if __name__ == "__main__":
    download_reasoning_datasets()
