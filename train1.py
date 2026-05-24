# -*- coding: utf-8 -*-
# ⚠️ DEPRECATED: This script was an initial prototype.
# Please use train2.py for full fine-tuning or train_lora.py for LoRA fine-tuning.
import os
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

from project_config import DATA_DIR, configure_runtime_dirs

# Đưa các hàm xử lý ra ngoài
def read_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def preprocess_function(examples, tokenizer):
    inputs = examples["en"]
    targets = examples["vi"]
    
    # Tokenize tiếng Anh (Input)
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Tokenize tiếng Việt (Labels) - Cách viết mới thay cho as_target_tokenizer
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- MỌI THỨ PHẢI NẰM TRONG KHỐI NÀY ---
if __name__ == '__main__':
    configure_runtime_dirs()
    base_dir = str(DATA_DIR)
    
    print("Đang đọc file...")
    # Thử nghiệm với 10k dòng để chạy cho nhanh
    train_ds = Dataset.from_dict({
        "en": read_txt(os.path.join(base_dir, "train/train.en"))[:10000],
        "vi": read_txt(os.path.join(base_dir, "train/train.vi"))[:10000]
    })
    
    test_ds = Dataset.from_dict({
        "en": read_txt(os.path.join(base_dir, "test/test.en")),
        "vi": read_txt(os.path.join(base_dir, "test/test.vi"))
    })
    
    raw_datasets = DatasetDict({"train": train_ds, "test": test_ds})

    model_checkpoint = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("Đang Tokenize song song...")
    # Truyền tokenizer vào hàm thông qua fn_kwargs
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        num_proc=1 if os.name == "nt" else min(4, os.cpu_count() or 1),
        fn_kwargs={"tokenizer": tokenizer} 
    )
    
    print("Xong rồi Trường ơi! Done.")
    print(tokenized_datasets["train"][0])
