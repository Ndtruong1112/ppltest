# -*- coding: utf-8 -*-
"""
Script để cài đặt PyTorch CUDA nếu chưa có
Chạy: python setup_pytorch_cuda.py
"""

import os
import subprocess
import sys

print("="*60)
print("🔧 PyTorch CUDA Setup")
print("="*60)

print(f"\n🐍 Python env: {sys.executable}")
print("Khuyen nghi: dung env tren o D (vi du D:\\Users\\Admin\\venvs\\pplnckh-1-gpu)")

index_url = os.getenv("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")

print("\n⏳ Installing or upgrading PyTorch with CUDA support...")
print("   (This will take 5-10 minutes...)")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--upgrade",
    "torch", "torchvision", "torchaudio",
    "--index-url", index_url
])

print("\n✅ Installation complete!")
print("\nVerifying installation...")

try:
    import torch
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("\n🎉 GPU is ready for training!")
    else:
        print("⚠️ CUDA not detected. Check your GPU and drivers.")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
