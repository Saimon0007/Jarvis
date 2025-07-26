"""
Step 3 (Reasoning): Fine-tune a model on reasoning data (GSM8K and StrategyQA) using Hugging Face Trainer.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

def fine_tune_reasoning():
    model_name = "gpt2"  # Change to your preferred model
    output_dir = "finetune/results_reasoning"
    # Combine both datasets into one txt file
    gsm8k_path = "finetune/data/gsm8k_cot_pairs.jsonl"
    strategyqa_path = "finetune/data/strategyqa_cot_pairs.jsonl"
    txt_path = "finetune/data/reasoning_finetune_pairs.txt"
    with open(txt_path, "w", encoding="utf-8") as fout:
        for path in [gsm8k_path, strategyqa_path]:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = eval(line)
                    fout.write(f"<|user|> {obj['prompt']}\n<|assistant|> {obj['response']}\n\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=txt_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Reasoning model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_reasoning()
