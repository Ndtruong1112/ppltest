# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from project_config import DATA_DIR, MODEL_DIR, RESULTS_DIR, configure_runtime_dirs


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
        for i in tqdm(range(0, len(inputs), batch_size), desc="Translating", leave=False):
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


def evaluate_model(label, model_name_or_path, inputs, references, device, batch_size, num_beams):
    print(f"\n📦 Đang load: {label}")
    tokenizer, model = load_model_and_tokenizer(model_name_or_path, device)

    print(f"🔄 Đang dịch với {label}...")
    predictions, elapsed = generate_predictions(model, tokenizer, inputs, device, batch_size, num_beams)
    metrics = evaluate_predictions(predictions, references, elapsed)
    return {
        "label": label,
        "model_name_or_path": str(model_name_or_path),
        "metrics": metrics,
        "predictions": predictions,
    }


def print_summary(base_result, finetuned_result):
    base_metrics = base_result["metrics"]
    ft_metrics = finetuned_result["metrics"]

    print("\n" + "=" * 78)
    print("📊 SO SÁNH TRƯỚC FINE-TUNE VÀ SAU FINE-TUNE")
    print("=" * 78)
    print(f"{'Chỉ số':<26}{'Mô hình gốc':>16}{'Mô hình đã train':>20}{'Chênh lệch':>16}")
    print("-" * 78)
    print(f"{'BLEU':<26}{base_metrics['bleu']:>16.2f}{ft_metrics['bleu']:>20.2f}{ft_metrics['bleu'] - base_metrics['bleu']:>16.2f}")
    print(f"{'chrF++':<26}{base_metrics['chrf_pp']:>16.2f}{ft_metrics['chrf_pp']:>20.2f}{ft_metrics['chrf_pp'] - base_metrics['chrf_pp']:>16.2f}")
    print(f"{'TER (thấp hơn tốt hơn)':<26}{base_metrics['ter']:>16.2f}{ft_metrics['ter']:>20.2f}{ft_metrics['ter'] - base_metrics['ter']:>16.2f}")
    print(f"{'Thời gian dịch (giây)':<26}{base_metrics['time_seconds']:>16.2f}{ft_metrics['time_seconds']:>20.2f}{ft_metrics['time_seconds'] - base_metrics['time_seconds']:>16.2f}")
    print("=" * 78)


def save_report(base_result, finetuned_result, references, inputs):
    report_dir = Path(RESULTS_DIR) / "compare_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "before_after_metrics.json"

    payload = {
        "dataset_size": len(inputs),
        "base_model": {
            "label": base_result["label"],
            "model_name_or_path": base_result["model_name_or_path"],
            "metrics": base_result["metrics"],
        },
        "finetuned_model": {
            "label": finetuned_result["label"],
            "model_name_or_path": finetuned_result["model_name_or_path"],
            "metrics": finetuned_result["metrics"],
        },
        "examples": [
            {
                "en": inputs[i],
                "reference_vi": references[i],
                "base_prediction": base_result["predictions"][i],
                "finetuned_prediction": finetuned_result["predictions"][i],
            }
            for i in range(min(5, len(inputs)))
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return report_path


def main():
    configure_runtime_dirs()

    base_model_id = os.getenv("PHOMT_BASE_MODEL_ID", "Helsinki-NLP/opus-mt-en-vi")
    finetuned_model_path = os.getenv("PHOMT_FINETUNED_MODEL_PATH", str(MODEL_DIR))
    max_samples = int(os.getenv("PHOMT_COMPARE_MAX_SAMPLES", "1000")) or None
    batch_size = int(os.getenv("PHOMT_COMPARE_BATCH_SIZE", "32" if torch.cuda.is_available() else "8"))
    num_beams = int(os.getenv("PHOMT_COMPARE_NUM_BEAMS", "1"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- THIẾT BỊ ĐANG DÙNG: {device.upper()} ---")

    test_en = read_txt(os.path.join(DATA_DIR, "test", "test.en"))
    test_vi = read_txt(os.path.join(DATA_DIR, "test", "test.vi"))

    if not test_en or not test_vi:
        print("❌ Dữ liệu test trống! Hãy kiểm tra lại đường dẫn.")
        raise SystemExit(1)

    if max_samples:
        test_en = test_en[:max_samples]
        test_vi = test_vi[:max_samples]

    print(f"✅ Đã load {len(test_en)} câu test")
    print(f"⚙️ Cấu hình so sánh: batch_size={batch_size}, num_beams={num_beams}, max_samples={max_samples or 'full'}")

    base_result = evaluate_model(
        "Mô hình gốc",
        base_model_id,
        test_en,
        test_vi,
        device,
        batch_size,
        num_beams,
    )

    finetuned_result = evaluate_model(
        "Mô hình đã fine-tune",
        finetuned_model_path,
        test_en,
        test_vi,
        device,
        batch_size,
        num_beams,
    )

    print_summary(base_result, finetuned_result)

    print("\n📝 VÍ DỤ SO SÁNH (3 câu đầu tiên):")
    print("-" * 78)
    for i in range(min(3, len(test_en))):
        print(f"\n{i + 1}. EN: {test_en[i]}")
        print(f"   Gốc      : {base_result['predictions'][i]}")
        print(f"   Fine-tune: {finetuned_result['predictions'][i]}")
        print(f"   Chuẩn    : {test_vi[i]}")
    print("-" * 78)

    report_path = save_report(base_result, finetuned_result, test_vi, test_en)
    print(f"\n💾 Đã lưu báo cáo so sánh tại: {report_path}")


if __name__ == "__main__":
    main()
