"""
Step 4: Parameter-Efficient Fine-Tuning (PEFT/LoRA) example using peft and transformers.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

def peft_finetune():
    model_name = "gpt2"  # Change to your preferred model
    data_path = "finetune/data/finetune_pairs.jsonl"
    output_dir = "finetune/results_peft"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    # Convert jsonl to txt for TextDataset
    txt_path = "finetune/data/finetune_pairs.txt"
    with open(data_path, "r", encoding="utf-8") as fin, open(txt_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = eval(line)
            fout.write(f"<|user|> {obj['prompt']}\n<|assistant|> {obj['response']}\n\n")
    from transformers import TextDataset, DataCollatorForLanguageModeling
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
    print(f"PEFT/LoRA fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    peft_finetune()
