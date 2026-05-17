# -*- coding: utf-8 -*-
import json
import os
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from project_config import CACHE_ROOT, DATA_DIR, RESULTS_DIR, configure_runtime_dirs


def read_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_parallel_data(split, max_samples=None):
    en_path = os.path.join(DATA_DIR, split, f"{split}.en")
    vi_path = os.path.join(DATA_DIR, split, f"{split}.vi")
    en_lines = read_txt(en_path)
    vi_lines = read_txt(vi_path)

    if len(en_lines) != len(vi_lines):
        raise ValueError(
            f"So dong khong khop o split '{split}': EN={len(en_lines)}, VI={len(vi_lines)}"
        )

    if max_samples is not None:
        en_lines = en_lines[:max_samples]
        vi_lines = vi_lines[:max_samples]

    if not en_lines:
        raise ValueError(f"Split '{split}' dang rong. Kiem tra lai du lieu trong {DATA_DIR}")

    return {"en": en_lines, "vi": vi_lines}


def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples["en"], max_length=128, truncation=True)
    labels = tokenizer(text_target=examples["vi"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def save_lora_metadata(output_dir, base_model_id, train_samples, eval_samples, device):
    metadata = {
        "base_model_id": base_model_id,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "device": device,
        "peft_type": "LoRA",
    }
    metadata_path = os.path.join(output_dir, "lora_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    configure_runtime_dirs()

    base_model_id = os.getenv("PHOMT_BASE_MODEL_ID", "Helsinki-NLP/opus-mt-en-vi")
    train_samples = int(os.getenv("PHOMT_LORA_TRAIN_SAMPLES", "10000"))
    eval_samples = int(os.getenv("PHOMT_LORA_EVAL_SAMPLES", "1000"))
    output_dir = os.getenv("PHOMT_LORA_RESULTS_DIR", str(RESULTS_DIR / "lora_results"))
    adapter_output_dir = os.getenv("PHOMT_LORA_ADAPTER_DIR", str(RESULTS_DIR / "lora_adapter"))

    lora_r = int(os.getenv("PHOMT_LORA_R", "8"))
    lora_alpha = int(os.getenv("PHOMT_LORA_ALPHA", "16"))
    lora_dropout = float(os.getenv("PHOMT_LORA_DROPOUT", "0.1"))

    if not torch.cuda.is_available():
        print("⚠️ CUDA không khả dụng. LoRA vẫn có thể chạy trên CPU nhưng sẽ rất chậm.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- THIẾT BỊ ĐANG DÙNG: {str(device).upper()} ---")
    print(f"📦 Base model: {base_model_id}")
    print(f"📁 Adapter output: {adapter_output_dir}")

    print("--- BƯỚC 1: ĐANG NẠP DỮ LIỆU ---")
    raw_datasets = DatasetDict({
        "train": Dataset.from_dict(load_parallel_data("train", max_samples=train_samples)),
        "test": Dataset.from_dict(load_parallel_data("test", max_samples=eval_samples)),
    })

    print("--- BƯỚC 2: TOKENIZE ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=1 if os.name == "nt" else min(4, os.cpu_count() or 1),
        fn_kwargs={"tokenizer": tokenizer},
    )

    print("--- BƯỚC 3: NẠP MÔ HÌNH GỐC VÀ GẮN LORA ---")
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, **model_kwargs).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=4 if torch.cuda.is_available() else 2,
        per_device_eval_batch_size=4 if torch.cuda.is_available() else 2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        logging_steps=50,
        save_steps=500,
        gradient_accumulation_steps=4 if torch.cuda.is_available() else 1,
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n✅ LoRA Training Mode")
    print(f"   LoRA r: {lora_r}")
    print(f"   LoRA alpha: {lora_alpha}")
    print(f"   LoRA dropout: {lora_dropout}")
    print(f"   Batch Size: {training_args.per_device_train_batch_size}")
    print(f"   FP16: {training_args.fp16}")

    print("\n--- BƯỚC 4: BẮT ĐẦU TRAINING LORA ---")
    trainer.train()

    print("--- BƯỚC 5: LƯU ADAPTER LORA ---")
    os.makedirs(adapter_output_dir, exist_ok=True)
    model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    save_lora_metadata(
        adapter_output_dir,
        base_model_id=base_model_id,
        train_samples=train_samples,
        eval_samples=eval_samples,
        device=str(device),
    )
    print(f"✅ Đã lưu LoRA adapter tại: {adapter_output_dir}")
