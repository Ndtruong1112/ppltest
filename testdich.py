# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from project_config import MODEL_DIR, configure_runtime_dirs

# 1. Đường dẫn đến thư mục mô hình trên ổ D
configure_runtime_dirs()
model_path = str(MODEL_DIR)

# 2. Nạp mô hình và tokenizer
print("Đang nạp bộ não IT vừa train...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# 3. Hàm dịch thuật
def translate_it(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_ids, max_length=128)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 4. Thử nghiệm với một câu tiếng Anh chuyên ngành
test_sentences = [
    "The pointer points to the memory address of the variable.",
    "Using a Min-Heap can optimize the priority queue in this algorithm.",
    "The database schema needs to be normalized to 3NF."
]

print("\n--- KẾT QUẢ DỊCH THUẬT ---")
for sentence in test_sentences:
    result = translate_it(sentence)
    print(f"EN: {sentence}")
    print(f"VI: {result}")
    print("-" * 30)
