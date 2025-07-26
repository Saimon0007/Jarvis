"""
Step 4 (Reasoning): PEFT/LoRA fine-tuning for reasoning datasets (GSM8K and StrategyQA).
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def peft_finetune_reasoning():
    model_name = "gpt2"  # Change to your preferred model
    output_dir = "finetune/results_reasoning_peft"
    txt_path = "finetune/data/reasoning_finetune_pairs.txt"
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
    print(f"PEFT/LoRA reasoning model saved to {output_dir}")

if __name__ == "__main__":
    peft_finetune_reasoning()
