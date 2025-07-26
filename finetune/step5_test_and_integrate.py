"""
Step 5: Test the fine-tuned model and instructions to integrate with Jarvis.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def test_model():
    model_dir = "finetune/results"  # or "finetune/results_peft" for LoRA
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "<|user|> How does photosynthesis work?\n<|assistant|>"
    output = pipe(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    print("Sample output:\n", output)

if __name__ == "__main__":
    test_model()

"""
Integration instructions:
1. In your Jarvis config.py, set the model path to 'finetune/results' (or 'finetune/results_peft' for LoRA).
2. Update the LLM provider in skills/llm.py to use your local fine-tuned model.
3. Restart Jarvis to use the new model.
"""
