"""
Step 5 (Reasoning): Test the reasoning fine-tuned model and provide integration instructions.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def test_reasoning_model():
    model_dir = "finetune/results_reasoning"  # or "finetune/results_reasoning_peft" for LoRA
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "<|user|> Let's break this down step-by-step: What is 17 + 25?\n<|assistant|>"
    output = pipe(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    print("Sample output:\n", output)

if __name__ == "__main__":
    test_reasoning_model()

"""
Integration instructions:
1. In your Jarvis config.py, set the model path to 'finetune/results_reasoning' (or 'finetune/results_reasoning_peft' for LoRA).
2. Update the LLM provider in skills/llm.py to use your local reasoning fine-tuned model.
3. For math or logic queries, prepend prompts with chain-of-thought instructions (e.g., 'Let's break this down step-by-step: ...').
4. Restart Jarvis to use the new model.
"""
