# -*- coding: utf-8 -*-
import os
# BƯỚC 1: NHẢY RÀO BẢO MẬT (Phải đặt trước các import khác)
os.environ["HF_SKIP_PROTECTED_LOAD"] = "1" 

import json
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import từ project cá nhân
try:
    from project_config import DATA_DIR, RESULTS_DIR, configure_runtime_dirs
except ImportError:
    # Dự phòng nếu không tìm thấy file config
    DATA_DIR = Path("D:/Users/Admin/Downloads/PhoMT/detokenization")
    RESULTS_DIR = Path("D:/Users/Admin/Downloads/PhoMT/results")

# Cấu hình encoding cho Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def read_txt(path):
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_base_model(model_name_or_path, device):
    print(f"🔍 Đang nạp mô hình gốc: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": {"": device} if device == "cuda" else None,
        "use_safetensors": True,
    }
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return tokenizer, model

def load_lora_model(base_model_id, adapter_path, device):
    print(f"🔍 Đang nạp LoRA Adapter từ: {adapter_path}")
    # Luôn nạp tokenizer từ base model để tránh lỗi config trong folder adapter
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": {"": device} if device == "cuda" else None,
        "use_safetensors": True,
    }
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model

def generate_predictions(model, tokenizer, inputs, device, batch_size, num_beams):
    predictions = []
    start = time.perf_counter()

    with torch.inference_mode():
        for i in tqdm(range(0, len(inputs), batch_size), desc="🚀 Đang dịch", leave=False):
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

def evaluate_predictions(predictions, references, elapsed_seconds):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])
    return {
        "bleu": round(bleu.score, 2),
        "chrf_pp": round(chrf.score, 2),
        "ter": round(ter.score, 2),
        "time_seconds": round(elapsed_seconds, 2),
        "avg_seconds_per_sentence": round(elapsed_seconds / len(references), 4),
    }

def evaluate_model(label, loader_fn, loader_args, inputs, references, device, batch_size, num_beams):
    print(f"\n📦 TRẠNG THÁI: {label}")
    try:
        tokenizer, model = loader_fn(*loader_args, device)
        predictions, elapsed = generate_predictions(model, tokenizer, inputs, device, batch_size, num_beams)
        metrics = evaluate_predictions(predictions, references, elapsed)
        return {"label": label, "metrics": metrics, "predictions": predictions}
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá {label}: {e}")
        return None

def print_summary(base_result, lora_result):
    if not base_result or not lora_result: return
    b_m = base_result["metrics"]
    l_m = lora_result["metrics"]

    print("\n" + "=" * 80)
    print(f"{'CHỈ SỐ ĐÁNH GIÁ':<30}{'BASE MODEL':>15}{'LORA MODEL':>15}{'CẢI THIỆN':>15}")
    print("-" * 80)
    print(f"{'BLEU Score (cao là tốt)':<30}{b_m['bleu']:>15.2f}{l_m['bleu']:>15.2f}{l_m['bleu'] - b_m['bleu']:>15.2f}")
    print(f"{'chrF++':<30}{b_m['chrf_pp']:>15.2f}{l_m['chrf_pp']:>15.2f}{l_m['chrf_pp'] - b_m['chrf_pp']:>15.2f}")
    print(f"{'TER (thấp là tốt)':<30}{b_m['ter']:>15.2f}{l_m['ter']:>15.2f}{l_m['ter'] - b_m['ter']:>15.2f}")
    print(f"{'Thời gian (s)':<30}{b_m['time_seconds']:>15.2f}{l_m['time_seconds']:>15.2f}{l_m['time_seconds'] - b_m['time_seconds']:>15.2f}")
    print("=" * 80)

def main():
    if 'configure_runtime_dirs' in globals(): configure_runtime_dirs()

    # --- TỰ ĐỘNG TÌM ĐƯỜNG DẪN ADAPTER ---
    import glob
    results_base = "D:/Users/Admin/Downloads/PhoMT/results"
    lora_results_base = os.path.join(results_base, "lora_results")
    
    # 1. Tìm tất cả các thư mục checkpoint-xxxx trong lora_results
    checkpoints = glob.glob(os.path.join(lora_results_base, "checkpoint-*"))
    
    if checkpoints:
        # Sắp xếp để lấy cái số to nhất (mới nhất)
        adapter_path = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"✅ Tự động chọn Checkpoint mới nhất: {adapter_path}")
    else:
        # Nếu không có checkpoint thì dùng folder lora_adapter mặc định
        adapter_path = os.path.join(results_base, "lora_adapter")
        print(f"⚠️ Không thấy checkpoint, dùng folder mặc định: {adapter_path}")

    base_model_id = "Helsinki-NLP/opus-mt-en-vi"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- HỆ THỐNG: {device.upper()} ---")
    if device == "cuda":
        print(f"--- GPU: {torch.cuda.get_device_name(0)} ---")

    # Đọc dữ liệu (Đảm bảo DATA_DIR chuẩn)
    test_en = read_txt(os.path.join(DATA_DIR, "test", "test.en"))
    test_vi = read_txt(os.path.join(DATA_DIR, "test", "test.vi"))

    if not test_en: raise SystemExit("❌ Dữ liệu trống hoặc sai đường dẫn DATA_DIR!")

    # Cấu hình so sánh từ môi trường
    max_samples_env = os.getenv("PHOMT_COMPARE_MAX_SAMPLES", "100")
    max_samples = None if max_samples_env.lower() in ("full", "none", "") else int(max_samples_env)
    batch_size = int(os.getenv("PHOMT_COMPARE_BATCH_SIZE", "64" if torch.cuda.is_available() else "16"))
    num_beams = int(os.getenv("PHOMT_COMPARE_NUM_BEAMS", "1"))

    print(f"⚙️ Cấu hình so sánh LoRA: batch_size={batch_size}, num_beams={num_beams}, max_samples={max_samples or 'full'}")

    if max_samples:
        test_en, test_vi = test_en[:max_samples], test_vi[:max_samples]

    base_result = evaluate_model("Mô hình Gốc", load_base_model, (base_model_id,), test_en, test_vi, device, batch_size, num_beams)
    lora_result = evaluate_model("Mô hình LoRA", load_lora_model, (base_model_id, adapter_path), test_en, test_vi, device, batch_size, num_beams)

    print_summary(base_result, lora_result)

    if base_result and lora_result:
        print("\n📝 VÍ DỤ THỰC TẾ:")
        for i in range(min(3, len(test_en))):
            print(f"\n[{i+1}] EN: {test_en[i]}")
            print(f"    Gốc : {base_result['predictions'][i]}")
            print(f"    LoRA: {lora_result['predictions'][i]}")
            print(f"    Target: {test_vi[i]}")

if __name__ == "__main__":
    main()