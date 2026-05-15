# 🎯 Quick Start - GPU Training

## 🚀 3 Bước Nhanh

### Step 1: Tạo virtualenv trên ổ D
```powershell
.\setup_env_d.ps1
```
*Script này tạo môi trường ở `D:\Users\Admin\venvs\pplnckh-1-gpu` và cài PyTorch CUDA*

Nếu bạn đã có sẵn env ở ổ D, set biến môi trường trước khi chạy:
```powershell
$env:PHOMT_VENV_DIR = "D:\Users\Admin\Downloads\PhoMT\.venv"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
```

### Step 2: Chạy Training
**Cách 1 (PowerShell - Recommended):**
```powershell
.\run_training.ps1
```

**Cách 2 (Command Line):**
```bash
python train2.py
```

**Cách 3 (Double-click):**
- Windows: Double-click `run_training.bat`

### Step 3: Đánh Giá BLEU
```bash
python final1.py
```

---

## ✅ Kiểm Tra GPU

### Xem GPU Info:
```bash
nvidia-smi
```

### Verify PyTorch CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 📊 Thông Tin Dữ Liệu Và Cache

Mặc định project sẽ dùng ổ **D** cho:
- Virtualenv: `PHOMT_VENV_DIR`
- Dữ liệu/model/checkpoint: `PHOMT_WORK_ROOT`
- Hugging Face cache: `PHOMT_CACHE_ROOT`
- Temp files: `PHOMT_TMP_DIR`

Ví dụ:
```
D:/Users/Admin/Downloads/PhoMT/
├── detokenization/
│   ├── train/
│   │   ├── train.en  (10,000 câu)
│   │   └── train.vi  (10,000 câu)
│   └── test/
│       ├── test.en   (1,000 câu)
│       └── test.vi   (1,000 câu)
├── results/          (Training checkpoints)
└── final_model_it/   (Output model)
```

---

## 🔥 Expected Performance

| GPU | Training Time | Speed |
|-----|---------------|-------|
| RTX 3060 (6GB) | ~2-3 min | 10x CPU |
| RTX 3080 (10GB) | ~1-2 min | 20x CPU |
| CPU i7 (16GB) | ~20 min | 1x baseline |

---

## 🐛 Troubleshooting

### CUDA not detected?
1. Run: `nvidia-smi` → Check if GPU driver works
2. Run: `.\setup_env_d.ps1` → Recreate GPU env on drive D
3. Restart computer if driver was installed

### Out of Memory?
Edit `train2.py`, change:
```python
per_device_train_batch_size=32  # → Change to 16 or 8
```

### Training too slow?
1. Check GPU usage: `nvidia-smi -l 1` (updates every 1s)
2. If 0%, GPU isn't being used - see GPU Setup Guide
3. If high but slow, your GPU model is old

---

## 📚 Files Explained

| File | Purpose |
|------|---------|
| `train2.py` | Main training script (ổ D) |
| `final1.py` | BLEU evaluation (ổ D) |
| `testdich.py` | Translation inference |
| `setup_env_d.ps1` | Tạo env GPU trên ổ D |
| `setup_pytorch_cuda.py` | GPU setup helper |
| `run_training.ps1` | PowerShell runner |
| `run_training.bat` | Batch runner |
| `GPU_SETUP.md` | Detailed GPU guide |

---

## 💡 Pro Tips

1. **Monitor GPU during training:**
   ```bash
   nvidia-smi -l 1
   ```

2. **Run multiple experiments:**
   ```bash
   # Train multiple times
   python train2.py
   python final1.py
   ```

3. **Custom training settings:**
   Edit `train2.py` lines 65-75 (training_args)

---

## ✨ What's Next?

After training:
1. ✅ Check BLEU score: `python final1.py`
2. ✅ Test translation: `python testdich.py`
3. ✅ Commit & push to GitHub
4. ✅ Increase data size (10k → 100k) for better score

---

**All paths are on Drive D** 🎯

Chúc bạn thành công! 🚀
