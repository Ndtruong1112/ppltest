# -*- coding: utf-8 -*-
import os
import sys
import time

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from project_config import DATA_DIR, configure_runtime_dirs


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_txt(path):
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_model_and_tokenizer(model_name_or_path, device):
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_kwargs).to(device)
    model.eval()
    return tokenizer, model


def generate_predictions(model, tokenizer, inputs, device, batch_size, num_beams):
    predictions = []
    start = time.perf_counter()

    with torch.inference_mode():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Translating"):
            batch = inputs[i:i + batch_size]
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=128,
                num_beams=num_beams,
                early_stopping=(num_beams > 1),
            )
            predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    elapsed = time.perf_counter() - start
    return predictions, elapsed


def main():
    configure_runtime_dirs()

    base_model_id = os.getenv("PHOMT_BASE_MODEL_ID", "Helsinki-NLP/opus-mt-en-vi")
    max_samples = int(os.getenv("PHOMT_COMPARE_MAX_SAMPLES", "1000")) or None
    batch_size = int(os.getenv("PHOMT_COMPARE_BATCH_SIZE", "32" if torch.cuda.is_available() else "8"))
    num_beams = int(os.getenv("PHOMT_COMPARE_NUM_BEAMS", "1"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- THIẾT BỊ ĐANG DÙNG: {device.upper()} ---")
    print(f"📦 Mô hình gốc: {base_model_id}")

    test_en = read_txt(os.path.join(DATA_DIR, "test", "test.en"))
    test_vi = read_txt(os.path.join(DATA_DIR, "test", "test.vi"))

    if not test_en or not test_vi:
        print("❌ Dữ liệu test trống! Hãy kiểm tra lại đường dẫn.")
        raise SystemExit(1)

    if max_samples:
        test_en = test_en[:max_samples]
        test_vi = test_vi[:max_samples]

    print(f"✅ Đã load {len(test_en)} câu test")
    print(f"⚙️ Cấu hình: batch_size={batch_size}, num_beams={num_beams}, max_samples={max_samples or 'full'}")

    tokenizer, model = load_model_and_tokenizer(base_model_id, device)
    predictions, elapsed = generate_predictions(model, tokenizer, test_en, device, batch_size, num_beams)

    bleu = sacrebleu.corpus_bleu(predictions, [test_vi])
    chrf = sacrebleu.corpus_chrf(predictions, [test_vi], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [test_vi])

    print("\n" + "=" * 60)
    print("🎯 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH GỐC")
    print("=" * 60)
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF++: {chrf.score:.2f}")
    print(f"TER: {ter.score:.2f}")
    print(f"Thời gian dịch: {elapsed:.2f} giây")
    print(f"Trung bình mỗi câu: {elapsed / len(test_en):.4f} giây")
    print("=" * 60)

    print("\n📝 VÍ DỤ DỊCH (5 câu đầu tiên):")
    print("-" * 60)
    for i in range(min(5, len(test_en))):
        print(f"\n{i + 1}. EN: {test_en[i]}")
        print(f"   VI (mô hình gốc): {predictions[i]}")
        print(f"   VI (chuẩn): {test_vi[i]}")
    print("-" * 60)


if __name__ == "__main__":
    main()
