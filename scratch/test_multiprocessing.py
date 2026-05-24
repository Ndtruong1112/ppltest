# -*- coding: utf-8 -*-
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples["en"], max_length=128, truncation=True)
    labels = tokenizer(text_target=examples["vi"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "Helsinki-NLP/opus-mt-en-vi"
    
    # 1. Create a dummy dataset
    data = {
        "en": ["Hello world", "How are you?"] * 50,
        "vi": ["Xin chào thế giới", "Bạn khỏe không?"] * 50,
    }
    raw_dataset = Dataset.from_dict(data)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    
    # 2. Load model & PEFT
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, use_safetensors=True).to(device)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    
    # 3. Create collator with model=None
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100)
    
    # 4. Training args with num_workers=2
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_test",
        max_steps=2,
        per_device_train_batch_size=4,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training test...")
    trainer.train()
    print("Training test finished successfully!")

if __name__ == "__main__":
    main()
