# -*- coding: utf-8 -*-
import json
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def translation_generator(split, max_samples=None):
    en_path = os.path.join(DATA_DIR, split, f"{split}.en")
    vi_path = os.path.join(DATA_DIR, split, f"{split}.vi")
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        raise FileNotFoundError(f"Khong tim thay file song ngu o split '{split}':\n  EN: {en_path}\n  VI: {vi_path}")
    
    with open(en_path, "r", encoding="utf-8") as f_en, open(vi_path, "r", encoding="utf-8") as f_vi:
        count = 0
        for line_en, line_vi in zip(f_en, f_vi):
            yield {"en": line_en.strip(), "vi": line_vi.strip()}
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def preprocess_function(examples, tokenizer):
    max_length = int(os.getenv("PHOMT_MAX_LENGTH", "128"))
    model_inputs = tokenizer(examples["en"], max_length=max_length, truncation=True)
    labels = tokenizer(text_target=examples["vi"], max_length=max_length, truncation=True)
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
    train_samples_env = os.getenv("PHOMT_LORA_TRAIN_SAMPLES", "500000")
    train_samples = None if train_samples_env.lower() in ("full", "none", "") else int(train_samples_env)
    eval_samples_env = os.getenv("PHOMT_LORA_EVAL_SAMPLES", "10000")
    eval_samples = None if eval_samples_env.lower() in ("full", "none", "") else int(eval_samples_env)
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

    print(f"--- BƯỚC 1: ĐANG NẠP DỮ LIỆU (Train: {train_samples_env}, Test: {eval_samples_env}) ---")
    raw_datasets = DatasetDict({
        "train": Dataset.from_generator(translation_generator, gen_kwargs={"split": "train", "max_samples": train_samples}),
        "test": Dataset.from_generator(translation_generator, gen_kwargs={"split": "test", "max_samples": eval_samples}),
    })

    print("--- BƯỚC 2: TOKENIZE ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    num_proc_default = "4" if (os.cpu_count() or 1) >= 4 else str(os.cpu_count() or 1)
    num_proc = int(os.getenv("PHOMT_TOKENIZE_NUM_PROC", num_proc_default))
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer},
    )

    print("--- BƯỚC 3: NẠP MÔ HÌNH GỐC VÀ GẮN LORA ---")
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, use_safetensors=True, **model_kwargs).to(device)

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

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100)

    import inspect
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    
    batch_size = int(os.getenv("PHOMT_BATCH_SIZE", "64"))
    grad_checkpointing = os.getenv("PHOMT_GRADIENT_CHECKPOINTING", "False").lower() == "true"
    grad_accum_steps = int(os.getenv("PHOMT_GRADIENT_ACCUMULATION_STEPS", "4" if torch.cuda.is_available() else "1"))

    training_kwargs = {
        "output_dir": output_dir,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "predict_with_generate": False,
        "fp16": torch.cuda.is_available(),
        "push_to_hub": False,
        "report_to": "none",
        "logging_steps": 50,
        "save_steps": 500,
        "gradient_checkpointing": grad_checkpointing if torch.cuda.is_available() else False,
        "gradient_accumulation_steps": grad_accum_steps if torch.cuda.is_available() else 1,
        "dataloader_pin_memory": True if torch.cuda.is_available() else False,
        "dataloader_num_workers": int(os.getenv("PHOMT_NUM_WORKERS", "0")),
    }
    
    eval_strategy = os.getenv("PHOMT_EVAL_STRATEGY", "epoch")
    if "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
        
    if torch.cuda.is_available():
        if "train_sampling_strategy" in sig.parameters:
            training_kwargs["train_sampling_strategy"] = "group_by_length"
        elif "group_by_length" in sig.parameters:
            training_kwargs["group_by_length"] = True

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
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
