import os
os.environ["HF_SKIP_PROTECTED_LOAD"] = "1" # Tạm thời nhảy rào để nạp mô hình cũ
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "Helsinki-NLP/opus-mt-en-vi"
save_path = "D:/Users/Admin/Downloads/PhoMT/models/opus-mt-safe"

print("--- ĐANG CHUYỂN ĐỔI MÔ HÌNH SANG ĐỊNH DẠNG AN TOÀN ---")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Lưu lại dưới dạng safetensors
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
print(f"--- XONG! Bạn hãy dùng đường dẫn này trong code: {save_path}")