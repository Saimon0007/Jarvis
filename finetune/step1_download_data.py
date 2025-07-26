"""
Step 1: Download a conversational dataset from Hugging Face (OpenAssistant).
"""
from datasets import load_dataset

def download_dataset():
    # Download OpenAssistant conversations dataset (small sample)
    dataset = load_dataset("OpenAssistant/oasst1", split="train[:1000]")
    dataset.save_to_disk("finetune/data/openassistant_sample")
    print("Dataset downloaded and saved to finetune/data/openassistant_sample")

if __name__ == "__main__":
    download_dataset()
