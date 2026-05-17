# 🚀 GPU Setup Guide

## Yêu Cầu GPU Training

### 1️⃣ Hardware Requirements
- ✅ NVIDIA GPU (RTX 3060 trở lên recommended)
- ✅ Minimum 6GB VRAM (recommended 8GB+)
- ✅ Latest GPU drivers

### 2️⃣ Software Requirements
- ✅ CUDA Toolkit 11.8 or 12.1
- ✅ cuDNN 8.x
- ✅ Python 3.10+

---

## 📋 Setup Steps

### Step 1: Check GPU
```bash
# Windows PowerShell
nvidia-smi
```

**Expected output:**
```
+-------------------------+
| NVIDIA-SMI X.XX         |
| GPU Name  | Memory      |
| GPU 0     | 8192MB      |
+-------------------------+
```

If `nvidia-smi` not found, install:
- [NVIDIA GPU Driver](https://www.nvidia.com/Download/driverDetails.aspx)

---

### Step 2: Install CUDA Toolkit (if needed)

**Windows:**
1. Download: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Run installer
3. Accept default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

**Verify CUDA:**
```bash
nvcc --version
```

---

### Step 3: Install cuDNN (if needed)

1. Download: https://developer.nvidia.com/cudnn
2. Extract to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
   - Copy files from `bin/`, `lib/`, `include/` folders

**Verify cuDNN:**
```bash
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
cudnn_version.exe
```

---

### Step 4: Setup PyTorch CUDA

**Option A: Auto Setup (Recommended)**
```bash
python setup_pytorch_cuda.py
```

**Option B: Manual Setup**
```bash
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch CUDA:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

### Step 5: Train with GPU

```bash
python train2.py
```

**Expected output:**
```
--- THIẾT BỊ ĐANG DÙNG: CUDA ---
✅ GPU Name: NVIDIA GeForce RTX 3080
✅ CUDA Version: 11.8
✅ GPU Memory: 10.0 GB
```

---

## 🐛 Troubleshooting

### Issue: `CUDA is NOT available`
**Solutions:**
1. Run: `nvidia-smi` to check GPU driver
2. Install latest driver from NVIDIA website
3. Restart computer after driver installation
4. Run: `python setup_pytorch_cuda.py`

### Issue: `Out of Memory`
**Solutions:**
- Reduce batch size: Change `per_device_train_batch_size=2` to `1` in `train2.py` if RTX 4050 6GB still runs out of memory
- Use gradient accumulation: Already enabled
- Use mixed precision: FP16 is enabled automatically

### Issue: `CUDA out of memory but GPU shows free memory`
**Solutions:**
- Run: `nvidia-smi` to check other processes
- Kill background applications using GPU
- Restart Python interpreter

### Issue: `Kernel not compiled with CUDA`
**Solutions:**
1. Uninstall PyTorch: `pip uninstall torch -y`
2. Install CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Restart terminal/IDE

---

## 📊 Performance Comparison

| Config | Time | Speed |
|--------|------|-------|
| CPU (i7 16GB) | ~20 min | 🐢 1x |
| GPU (RTX 3060) | ~2 min | 🚀 10x |
| GPU (RTX 3080) | ~1 min | 🚀 20x |

---

## ✅ Final Verification

Run this to confirm GPU training is ready:

```bash
python -c "
import torch
from transformers import AutoModelForSeq2SeqLM

print('='*50)
print('GPU Training Status')
print('='*50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU ONLY\"}')

if torch.cuda.is_available():
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # Test model loading on GPU
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
    model.to('cuda')
    print('✅ Model loaded successfully on GPU!')
else:
    print('⚠️ GPU not available - will use CPU (SLOW)')
print('='*50)
"
```

---

## 📞 Getting Help

- **NVIDIA CUDA**: https://docs.nvidia.com/cuda/
- **PyTorch Setup**: https://pytorch.org/get-started/locally/
- **GitHub Issues**: Report problems here

---

**Last Updated:** May 15, 2026
