# -*- coding: utf-8 -*-
# PowerShell script to run PhoMT translation pipeline (Train/Eval/Test)

param(
    [string]$VenvDir = $env:PHOMT_VENV_DIR
)

Clear-Host
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   PhoMT IT Translation Pipeline Runner (RTX 4050/6GB)   " -ForegroundColor Cyan -BackgroundColor DarkBlue
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Activate Virtual Environment
if (-not $VenvDir) {
    if (Test-Path "D:\Users\Admin\Downloads\PhoMT\.venv\Scripts\Activate.ps1") {
        $VenvDir = "D:\Users\Admin\Downloads\PhoMT\.venv"
    } elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
        $VenvDir = Join-Path $PSScriptRoot ".venv"
    }
}

if (-not $VenvDir) {
    Write-Host "❌ ERROR: Virtual environment not found. Please run setup_env_d.ps1 first." -ForegroundColor Red
    exit 1
}

$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "❌ ERROR: Activation script not found at: $activateScript" -ForegroundColor Red
    exit 1
}

& $activateScript

# Set environment encoding & path defaults
$env:PYTHONIOENCODING = 'utf-8'
$env:PHOMT_WORK_ROOT = if ($env:PHOMT_WORK_ROOT) { $env:PHOMT_WORK_ROOT } else { 'D:\Users\Admin\Downloads\PhoMT' }
$env:PHOMT_CACHE_ROOT = if ($env:PHOMT_CACHE_ROOT) { $env:PHOMT_CACHE_ROOT } else { 'D:\Users\Admin\Downloads\PhoMT\.cache' }
$env:PHOMT_TMP_DIR = if ($env:PHOMT_TMP_DIR) { $env:PHOMT_TMP_DIR } else { 'D:\Users\Admin\Downloads\PhoMT\.tmp' }

# 2. Check GPU Availability
Write-Host "🔍 Hardware & PyTorch Diagnostics..." -ForegroundColor Yellow
python -c @"
import torch
print(f'   - PyTorch Version: {torch.__version__}')
print(f'   - CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   - Active GPU: {torch.cuda.get_device_name(0)}')
    print(f'   - VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('   - ⚠️ GPU NOT available - running on CPU (Slow)')
"@
Write-Host ""

# 3. Interactive Menu
Write-Host "📋 SELECT AN ACTION TO PERFORM:" -ForegroundColor White
Write-Host "  [1] Train: LoRA Fine-Tuning (Recommended: fast, fits 6GB VRAM easily)" -ForegroundColor Green
Write-Host "  [2] Train: Full Fine-Tuning (Updates all parameters, uses more VRAM)" -ForegroundColor Green
Write-Host "  [3] Eval: Compare Base model vs LoRA Adapter (BLEU/chrF++/TER)" -ForegroundColor Cyan
Write-Host "  [4] Eval: Compare Base model vs Full Fine-Tuned (BLEU/chrF++/TER)" -ForegroundColor Cyan
Write-Host "  [5] Test: Translate individual sentences interactively" -ForegroundColor Magenta
Write-Host "  [6] Exit" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Enter your choice (1-6)"

if ($choice -eq "6" -or -not $choice) {
    Write-Host "👋 Goodbye!" -ForegroundColor Gray
    exit 0
}

# 4. Gather Configurations (for training choices)
if ($choice -eq "1" -or $choice -eq "2") {
    Write-Host ""
    Write-Host "⚙️ CONFIGURING TRAINING HYPERPARAMETERS:" -ForegroundColor Yellow
    
    # Dataset limits
    $train_samples = Read-Host "   - Number of training sentences (default: 500000, type 'full' for all)"
    if (-not $train_samples) { $train_samples = "500000" }
    
    $eval_samples = Read-Host "   - Number of evaluation sentences (default: 10000, type 'full' for all)"
    if (-not $eval_samples) { $eval_samples = "10000" }
    
    # Batch size & accumulation
    $batch_size = Read-Host "   - Batch size (default: 64)"
    if (-not $batch_size) { $batch_size = "64" }
    
    $grad_accum = Read-Host "   - Gradient accumulation steps (default: 2)"
    if (-not $grad_accum) { $grad_accum = "2" }
    
    # Environment config exports
    $env:PHOMT_BATCH_SIZE = $batch_size
    $env:PHOMT_GRADIENT_ACCUMULATION_STEPS = $grad_accum
    
    # Set speed optimization exports
    $env:PHOMT_GRADIENT_CHECKPOINTING = "False" # Set False for ~30% faster execution; fits in 6GB VRAM
    $env:PHOMT_NUM_WORKERS = "0"                 # Set 0 to avoid Windows multi-process spawn delay
    $env:PHOMT_EVAL_STRATEGY = "epoch"           # Set epoch-based eval to avoid step-based pauses
    
    if ($choice -eq "1") {
        $env:PHOMT_LORA_TRAIN_SAMPLES = $train_samples
        $env:PHOMT_LORA_EVAL_SAMPLES = $eval_samples
        Write-Host ""
        Write-Host "🚀 Launching LoRA Training..." -ForegroundColor Green
        python train_lora.py
    } else {
        $env:PHOMT_TRAIN_SAMPLES = $train_samples
        $env:PHOMT_EVAL_SAMPLES = $eval_samples
        Write-Host ""
        Write-Host "🚀 Launching Full Fine-Tuning..." -ForegroundColor Green
        python train2.py
    }
}
elseif ($choice -eq "3") {
    # Compare LoRA
    $max_eval = Read-Host "   - Sentences to evaluate (default: 1000, enter for default)"
    if ($max_eval) { $env:PHOMT_COMPARE_MAX_SAMPLES = $max_eval }
    Write-Host ""
    Write-Host "🎯 Running LoRA Adapter Evaluation..." -ForegroundColor Cyan
    python compare_lora.py
}
elseif ($choice -eq "4") {
    # Compare Full
    $max_eval = Read-Host "   - Sentences to evaluate (default: 1000, enter for default)"
    if ($max_eval) { $env:PHOMT_COMPARE_MAX_SAMPLES = $max_eval }
    Write-Host ""
    Write-Host "🎯 Running Full Fine-Tuning Evaluation..." -ForegroundColor Cyan
    python compare_before_after.py
}
elseif ($choice -eq "5") {
    # Interactive Test
    Write-Host ""
    Write-Host "💬 Running Interactive Translator..." -ForegroundColor Magenta
    python testdich.py
}
else {
    Write-Host "❌ Invalid choice!" -ForegroundColor Red
}

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "✅ Action completed!" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Read-Host "Press ENTER to exit"
