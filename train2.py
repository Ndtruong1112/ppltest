# -*- coding: utf-8 -*-
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from project_config import DATA_DIR, MODEL_DIR, RESULTS_DIR, configure_runtime_dirs

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
    inputs = examples["en"]
    targets = examples["vi"]
    max_length = int(os.getenv("PHOMT_MAX_LENGTH", "128"))
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == '__main__':
    configure_runtime_dirs()

    # ÉP CHẠY GPU: Kiểm tra và khai báo
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA is NOT available! GPU not detected.")
        print("Make sure you have:")
        print("  1. NVIDIA GPU installed")
        print("  2. NVIDIA driver installed and working (nvidia-smi)")
        print("  3. PyTorch with CUDA support")
        print("  4. Run setup_env_d.ps1 to create a GPU-ready venv on drive D")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- THIẾT BỊ ĐANG DÙNG: {str(device).upper()} ---")
    
    if device.type == "cuda":
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ PyTorch Version: {torch.__version__}")

    model_checkpoint = "Helsinki-NLP/opus-mt-en-vi"
    output_dir = str(RESULTS_DIR)
    model_output_dir = str(MODEL_DIR)
    
    train_samples_env = os.getenv("PHOMT_TRAIN_SAMPLES", "500000")
    train_samples = None if train_samples_env.lower() in ("full", "none", "") else int(train_samples_env)
    eval_samples_env = os.getenv("PHOMT_EVAL_SAMPLES", "10000")
    eval_samples = None if eval_samples_env.lower() in ("full", "none", "") else int(eval_samples_env)

    print(f"--- BƯỚC 1: ĐANG NẠP DỮ LIỆU (Train: {train_samples_env}, Test: {eval_samples_env}) ---")
    raw_datasets = DatasetDict({
        "train": Dataset.from_generator(translation_generator, gen_kwargs={"split": "train", "max_samples": train_samples}),
        "test": Dataset.from_generator(translation_generator, gen_kwargs={"split": "test", "max_samples": eval_samples}),
    })

    print("--- BƯỚC 2: TOKENIZE ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    num_proc_default = "4" if (os.cpu_count() or 1) >= 4 else str(os.cpu_count() or 1)
    num_proc = int(os.getenv("PHOMT_TOKENIZE_NUM_PROC", num_proc_default))
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer}
    )

    print("--- BƯỚC 3: NẠP MÔ HÌNH VÀO GPU ---")
    model_kwargs = {
        "use_safetensors": True
    }
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, **model_kwargs).to(device) # Ép model lên GPU
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100)

    import inspect
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    batch_size = int(os.getenv("PHOMT_BATCH_SIZE", "64"))
    grad_checkpointing = os.getenv("PHOMT_GRADIENT_CHECKPOINTING", "False").lower() == "true"
    grad_accum_steps = int(os.getenv("PHOMT_GRADIENT_ACCUMULATION_STEPS", "8" if torch.cuda.is_available() else "1"))

    training_kwargs = {
        "output_dir": output_dir,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "save_total_limit": 3,
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

    # Kiểm tra xác nhận trước khi training
    if device.type == "cuda":
        print("\n✅ GPU Training Mode")
        print(f"   Batch Size: {training_args.per_device_train_batch_size}")
        print(f"   FP16: {training_args.fp16}")
        print(f"   Pin Memory: {training_args.dataloader_pin_memory}")
    else:
        print("\n⚠️ CPU Training Mode (SLOW!)")
    
    print("\n--- BƯỚC 4: BẮT ĐẦU TRAINING ---")
    trainer.train()
    
    print("--- BƯỚC 5: LƯU MÔ HÌNH ---")
    trainer.save_model(model_output_dir)
    print(f"✅ Xong! Mô hình đã lưu tại: {model_output_dir}")
