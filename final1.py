# -*- coding: utf-8 -*-
import os

import torch
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from project_config import DATA_DIR, MODEL_DIR, configure_runtime_dirs

# ==================== CẤU HÌNH ====================
configure_runtime_dirs()
model_path = str(MODEL_DIR)
base_dir = str(DATA_DIR)
max_samples = None  # None = test toàn bộ, hoặc số lượng câu cụ thể (ví dụ: 100)

# Kiểm tra thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- THIẾT BỊ ĐANG DÙNG: {device.upper()} ---\n")

# ==================== LOAD MÔ HÌNH ====================
print("📦 Đang load mô hình đã train...")
try:
    metric = load("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    print("✅ Mô hình đã load thành công!\n")
except Exception as e:
    print(f"❌ Lỗi load mô hình: {e}")
    exit()

# ==================== ĐỌC DỮ LIỆU TEST ====================
print("📂 Đang đọc dữ liệu test...")

def read_txt(path):
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

test_en = read_txt(os.path.join(base_dir, "test/test.en"))
test_vi = read_txt(os.path.join(base_dir, "test/test.vi"))

if not test_en or not test_vi:
    print("❌ Dữ liệu test trống! Hãy kiểm tra lại đường dẫn.")
    exit()

if max_samples:
    test_en = test_en[:max_samples]
    test_vi = test_vi[:max_samples]

print(f"✅ Đã load {len(test_en)} câu test\n")

# ==================== DỊCH TOÀN BỘ BỘ TEST ====================
print("🔄 Đang dịch...")
predictions = []
batch_size = 8

with torch.no_grad():
    for i in tqdm(range(0, len(test_en), batch_size), desc="Translating"):
        batch = test_en[i:i + batch_size]
        
        # Tokenize câu tiếng Anh
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Dự đoán dịch
        output_ids = model.generate(**inputs, max_length=128)
        
        # Decode thành text
        preds_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(preds_batch)

# ==================== TÍNH BLEU SCORE ====================
print("\n📊 Đang tính BLEU score...")

# Format references để phù hợp với sacrebleu
references = [[ref] for ref in test_vi]

results = metric.compute(predictions=predictions, references=references)

# ==================== IN KẾT QUẢ ====================
print("\n" + "="*60)
print("🎯 KẾT QUẢ ĐÁNH GIÁ BLEU SCORE")
print("="*60)
print(f"BLEU Score: {results['score']:.2f}")
print(f"Precisions (1-gram, 2-gram, 3-gram, 4-gram): {results['precisions']}")
print(f"BP (Brevity Penalty): {results['bp']:.4f}")
print(f"Ratio (độ dài dịch / độ dài chuẩn): {results['ratio']:.4f}")
print(f"Tổng độ dài dịch: {results['translation_length']}")
print(f"Tổng độ dài chuẩn: {results['reference_length']}")
print("="*60)

# ==================== HIỂN THỊ VÍ DỤ ====================
print("\n📝 VÍ DỤ DỊCH (5 câu đầu tiên):")
print("-" * 60)
for i in range(min(5, len(test_en))):
    print(f"\n{i+1}. EN: {test_en[i]}")
    print(f"   VI (dự đoán): {predictions[i]}")
    print(f"   VI (chuẩn): {test_vi[i]}")
print("-" * 60)
