# -*- coding: utf-8 -*-
# PowerShell script để chạy training trên GPU

param(
    [string]$VenvDir = $env:PHOMT_VENV_DIR
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "GPU Training Runner - PhoMT IT Translation" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if (-not $VenvDir) {
    if (Test-Path "D:\Users\Admin\Downloads\PhoMT\.venv\Scripts\Activate.ps1") {
        $VenvDir = "D:\Users\Admin\Downloads\PhoMT\.venv"
    } elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
        $VenvDir = Join-Path $PSScriptRoot ".venv"
    }
}

if (-not $VenvDir) {
    Write-Host "ERROR: Không tìm thấy virtualenv. Hãy chạy .\setup_env_d.ps1 trước." -ForegroundColor Red
    exit 1
}

$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: Không tìm thấy file kích hoạt virtualenv: $activateScript" -ForegroundColor Red
    exit 1
}

# Activate virtual environment on drive D if available
& $activateScript

# Set encoding
$env:PYTHONIOENCODING = 'utf-8'
$env:PHOMT_WORK_ROOT = if ($env:PHOMT_WORK_ROOT) { $env:PHOMT_WORK_ROOT } else { 'D:\Users\Admin\Downloads\PhoMT' }
$env:PHOMT_CACHE_ROOT = if ($env:PHOMT_CACHE_ROOT) { $env:PHOMT_CACHE_ROOT } else { 'D:\Users\Admin\Downloads\PhoMT\.cache' }
$env:PHOMT_TMP_DIR = if ($env:PHOMT_TMP_DIR) { $env:PHOMT_TMP_DIR } else { 'D:\Users\Admin\Downloads\PhoMT\.tmp' }

# Check if training script exists
if (-not (Test-Path "train2.py")) {
    Write-Host "ERROR: train2.py not found!" -ForegroundColor Red
    exit 1
}

# Check GPU availability
Write-Host "🔍 Checking GPU availability..." -ForegroundColor Yellow
python -c @"
import torch
print(f'✅ PyTorch Version: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA Version: {torch.version.cuda}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ GPU NOT available - will use CPU (SLOW!)')
"@

Write-Host ""
Write-Host "📊 Training Configuration:" -ForegroundColor Cyan
Write-Host "   - Model: Helsinki-NLP/opus-mt-en-vi" -ForegroundColor Cyan
Write-Host "   - Epochs: 3" -ForegroundColor Cyan
Write-Host "   - Venv: $VenvDir" -ForegroundColor Cyan
Write-Host "   - Work root: $env:PHOMT_WORK_ROOT" -ForegroundColor Cyan
Write-Host "   - Cache root: $env:PHOMT_CACHE_ROOT" -ForegroundColor Cyan
Write-Host ""

# Ask for confirmation
Read-Host "Press ENTER to start training (CTRL+C to cancel)"

Write-Host ""
Write-Host "🚀 Starting training..." -ForegroundColor Green
Write-Host ""

# Run training
$env:PYTHONIOENCODING = 'utf-8'
python train2.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ Training failed!" -ForegroundColor Red
    Write-Host "Hãy giảm batch size trong train2.py hoặc đóng bớt ứng dụng đang dùng GPU." -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Ask to run evaluation
$response = Read-Host "Run BLEU evaluation? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "🎯 Running BLEU Score Evaluation..." -ForegroundColor Green
    python final1.py
}
