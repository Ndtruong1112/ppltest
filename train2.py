# -*- coding: utf-8 -*-
import os

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from project_config import DATA_DIR, MODEL_DIR, RESULTS_DIR, configure_runtime_dirs

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
    inputs = examples["en"]
    targets = examples["vi"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
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
    
    print("--- BƯỚC 1: ĐANG NẠP DỮ LIỆU ---")
    raw_datasets = DatasetDict({
        "train": Dataset.from_dict(load_parallel_data("train", max_samples=10000)),
        "test": Dataset.from_dict(load_parallel_data("test", max_samples=1000)),
    })

    print("--- BƯỚC 2: TOKENIZE ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    num_proc = 1 if os.name == "nt" else min(4, os.cpu_count() or 1)
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer}
    )

    print("--- BƯỚC 3: NẠP MÔ HÌNH VÀO GPU ---")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device) # Ép model lên GPU
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        logging_steps=50,
        save_steps=500,
        gradient_accumulation_steps=8 if torch.cuda.is_available() else 1,
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
