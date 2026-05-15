# 📊 Project Summary - PhoMT IT Translation

## 🎯 Dự Án

**Hệ Thống Dịch Máy IT: Anh → Việt (PhoMT Fine-tuning)**

Đây là dự án fine-tune mô hình dịch máy chuyên biệt cho lĩnh vực IT, sử dụng:
- **Mô hình base:** Helsinki-NLP/opus-mt-en-vi
- **Framework:** Hugging Face Transformers
- **Kỹ thuật:** Seq2Seq (Sequence-to-Sequence) fine-tuning

---

## 📂 Cấu Trúc Project

```
pplnckh/
├── 📁 .github/workflows/          # GitHub Actions CI/CD
│   ├── evaluate_bleu.yml          # Workflow đánh giá BLEU
│   ├── train_model.yml            # Workflow training
│   └── code-quality.yml           # Workflow kiểm tra code
│
├── 📄 train2.py                   # Script training chính ⭐
├── 📄 final1.py                   # Script đánh giá BLEU ⭐
├── 📄 testdich.py                 # Script inference/dịch
├── 📄 train1.py                   # Script helper (tokenization)
│
├── 📖 README.md                   # Tài liệu chính
├── 📋 CONTRIBUTING.md             # Hướng dẫn đóng góp
├── 📜 LICENSE                     # MIT License
├── 📝 CHANGELOG.md                # Lịch sử thay đổi
├── 📊 PROJECT_SUMMARY.md          # File này
├── 📦 requirements.txt            # Dependencies
├── .gitignore                     # Git ignore rules
└── .git/                          # Git repository
```

---

## 🚀 Workflow & Quy Trình

### 1️⃣ **Training** (`train2.py`)
```
Input Data (train.en + train.vi)
         ↓
Tokenization (parallel, 4 processes)
         ↓
Fine-tuning on GPU/CPU
         ↓
Model saved to: final_model_it/
```

**Configuration:**
- Batch size: 16
- Epochs: 3
- Learning rate: 2e-5
- FP16: Enabled (GPU only)

---

### 2️⃣ **Evaluation** (`final1.py`)
```
Model + Test Data
         ↓
Batch Translation
         ↓
BLEU Score Calculation
         ↓
Metrics & Examples Display
```

**Output:**
- BLEU Score (0-100)
- Individual n-gram precisions
- Brevity penalty
- 5 translation examples

---

### 3️⃣ **Inference** (`testdich.py`)
```
English IT Text
         ↓
Tokenization
         ↓
Model Prediction
         ↓
Vietnamese Output
```

---

## 🔄 GitHub Actions CI/CD

### Workflows Tự Động:

| Workflow | Trigger | Action |
|----------|---------|--------|
| **evaluate_bleu.yml** | Push, PR, Manual | Chạy BLEU evaluation trên Python 3.10, 3.11 |
| **train_model.yml** | Manual dispatch | Validate training script |
| **code-quality.yml** | Push, PR | Syntax check, code style, security |

---

## 📊 Project Statistics

| Metric | Giá Trị |
|--------|--------|
| **Python Files** | 4 files |
| **Lines of Code** | ~350 LOC |
| **Model Base** | 258M parameters |
| **Training Data** | 10,000 sentence pairs |
| **Test Data** | 1,000 sentence pairs |
| **Model Size** | ~300MB (safetensors) |
| **GPU Memory** | 4GB (FP16) / 8GB (FP32) |
| **Training Time** | ~1-2 min (GPU) / ~15-20 min (CPU) |

---

## ✅ Quality Assurance

### Code Quality
- ✅ Python syntax validation
- ✅ Code style checking (Black)
- ✅ Import sorting (isort)
- ✅ UTF-8 encoding
- ✅ Error handling

### Testing & Evaluation
- ✅ BLEU Score metrics
- ✅ Translation quality check
- ✅ Model inference validation
- ✅ GPU/CPU compatibility

---

## 🎯 Key Files Explained

### `train2.py` ⭐ (Chính)
- **Mục đích:** Huấn luyện mô hình
- **Input:** train.en, train.vi
- **Output:** final_model_it/ (checkpoint + model)
- **Runtime:** 1-20 phút (tuỳ GPU/CPU)

### `final1.py` ⭐ (Đánh giá)
- **Mục đích:** Tính BLEU score
- **Input:** test.en, test.vi + trained model
- **Output:** BLEU score + metrics
- **Runtime:** 2-5 phút

### `testdich.py` (Sử dụng)
- **Mục đích:** Dịch câu tiếng Anh
- **Input:** English text
- **Output:** Vietnamese translation
- **Runtime:** Real-time inference

---

## 🔧 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Ndtruong1112/pplnckh.git
cd pplnckh
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
```
Place your data in:
D:/Users/Admin/Downloads/PhoMT/detokenization/
├── train/
│   ├── train.en
│   └── train.vi
└── test/
    ├── test.en
    └── test.vi
```

---

## 🚀 Quick Start

### Train Model
```bash
python train2.py
```

### Evaluate BLEU
```bash
python final1.py
```

### Translate Sentences
```bash
python testdich.py
```

---

## 📈 Performance

**Current BLEU Score:** (To be measured)

**Expected Improvements:**
- Increase data: 10k → 100k (+2-3 BLEU)
- Data augmentation: +1-2 BLEU
- Hyperparameter tuning: +0.5-1 BLEU

---

## 🤝 Contributing

See `CONTRIBUTING.md` for:
- How to contribute
- Development process
- Pull request guidelines
- Code standards

---

## 📜 License

This project is licensed under the **MIT License** - see `LICENSE` file.

---

## 📞 Contact & Support

- **GitHub Issues:** Report bugs or request features
- **GitHub Discussions:** Ask questions
- **Author:** Ndtruong1112

---

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Helsinki-NLP/opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [PhoMT Project](https://github.com/VinAIResearch/PhoMT)

---

**Last Updated:** May 15, 2025  
**Project Status:** 🟢 Active Development
