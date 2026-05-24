@echo off
REM -*- coding: utf-8 -*-
REM Batch script to run PhoMT translation pipeline (Train/Eval/Test) - Windows CMD

cd /d %~dp0
cls

echo =========================================================
echo    PhoMT IT Translation Pipeline Runner (RTX 4050/6GB)   
echo =========================================================
echo.

set "VENV_DIR=%PHOMT_VENV_DIR%"
if "%VENV_DIR%"=="" if exist "D:\Users\Admin\Downloads\PhoMT\.venv\Scripts\activate.bat" set "VENV_DIR=D:\Users\Admin\Downloads\PhoMT\.venv"
if "%VENV_DIR%"=="" if exist ".venv\Scripts\activate.bat" set "VENV_DIR=%cd%\.venv"

if "%VENV_DIR%"=="" (
    echo ❌ ERROR: Virtual environment not found. Please run setup_env_d.ps1 first.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ❌ ERROR: Activation script not found at: %VENV_DIR%\Scripts\activate.bat
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM Set environment encoding & path defaults
set PYTHONIOENCODING=utf-8
if "%PHOMT_WORK_ROOT%"=="" set "PHOMT_WORK_ROOT=D:\Users\Admin\Downloads\PhoMT"
if "%PHOMT_CACHE_ROOT%"=="" set "PHOMT_CACHE_ROOT=D:\Users\Admin\Downloads\PhoMT\.cache"
if "%PHOMT_TMP_DIR%"=="" set "PHOMT_TMP_DIR=D:\Users\Admin\Downloads\PhoMT\.tmp"

REM Check GPU Availability
echo 🔍 Hardware ^& PyTorch Diagnostics...
python -c "import torch; print(f'   - PyTorch Version: {torch.__version__}'); print(f'   - CUDA Available: {torch.cuda.is_available()}'); print(f'   - Active GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU - SLOW!\"}') if torch.cuda.is_available() else print('   - ⚠️ GPU NOT available')"
echo.

echo 📋 SELECT AN ACTION TO PERFORM:
echo   [1] Train: LoRA Fine-Tuning (Recommended: fast, fits 6GB VRAM easily)
echo   [2] Train: Full Fine-Tuning (Updates all parameters, uses more VRAM)
echo   [3] Eval: Compare Base model vs LoRA Adapter (BLEU/chrF++/TER)
echo   [4] Eval: Compare Base model vs Full Fine-Tuned (BLEU/chrF++/TER)
echo   [5] Test: Translate individual sentences interactively
echo   [6] Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="6" goto end
if "%choice%"=="" goto end

if "%choice%"=="1" goto train_lora
if "%choice%"=="2" goto train_full
if "%choice%"=="3" goto eval_lora
if "%choice%"=="4" goto eval_full
if "%choice%"=="5" goto test_dich

echo ❌ Invalid choice!
pause
exit /b 1

:train_lora
echo.
echo ⚙️ CONFIGURING TRAINING HYPERPARAMETERS (LoRA):
set /p train_samples="   - Number of training sentences (default: 500000, enter for default): "
if "%train_samples%"=="" set "train_samples=500000"

set /p eval_samples="   - Number of evaluation sentences (default: 10000, enter for default): "
if "%eval_samples%"=="" set "eval_samples=10000"

set /p batch_size="   - Batch size (default: 64): "
if "%batch_size%"=="" set "batch_size=64"

set /p grad_accum="   - Gradient accumulation steps (default: 2): "
if "%grad_accum%"=="" set "grad_accum=2"

set "PHOMT_BATCH_SIZE=%batch_size%"
set "PHOMT_GRADIENT_ACCUMULATION_STEPS=%grad_accum%"
set "PHOMT_GRADIENT_CHECKPOINTING=False"
set "PHOMT_NUM_WORKERS=0"
set "PHOMT_EVAL_STRATEGY=epoch"
set "PHOMT_LORA_TRAIN_SAMPLES=%train_samples%"
set "PHOMT_LORA_EVAL_SAMPLES=%eval_samples%"

echo.
echo 🚀 Launching LoRA Training...
python train_lora.py
goto completed

:train_full
echo.
echo ⚙️ CONFIGURING TRAINING HYPERPARAMETERS (Full Fine-Tune):
set /p train_samples="   - Number of training sentences (default: 500000, enter for default): "
if "%train_samples%"=="" set "train_samples=500000"

set /p eval_samples="   - Number of evaluation sentences (default: 10000, enter for default): "
if "%eval_samples%"=="" set "eval_samples=10000"

set /p batch_size="   - Batch size (default: 64): "
if "%batch_size%"=="" set "batch_size=64"

set /p grad_accum="   - Gradient accumulation steps (default: 2): "
if "%grad_accum%"=="" set "grad_accum=2"

set "PHOMT_BATCH_SIZE=%batch_size%"
set "PHOMT_GRADIENT_ACCUMULATION_STEPS=%grad_accum%"
set "PHOMT_GRADIENT_CHECKPOINTING=False"
set "PHOMT_NUM_WORKERS=0"
set "PHOMT_EVAL_STRATEGY=epoch"
set "PHOMT_TRAIN_SAMPLES=%train_samples%"
set "PHOMT_EVAL_SAMPLES=%eval_samples%"

echo.
echo 🚀 Launching Full Fine-Tuning...
python train2.py
goto completed

:eval_lora
set /p max_eval="   - Sentences to evaluate (default: 1000, enter for default): "
if not "%max_eval%"=="" set "PHOMT_COMPARE_MAX_SAMPLES=%max_eval%"
echo.
echo 🎯 Running LoRA Adapter Evaluation...
python compare_lora.py
goto completed

:eval_full
set /p max_eval="   - Sentences to evaluate (default: 1000, enter for default): "
if not "%max_eval%"=="" set "PHOMT_COMPARE_MAX_SAMPLES=%max_eval%"
echo.
echo 🎯 Running Full Fine-Tuning Evaluation...
python compare_before_after.py
goto completed

:test_dich
echo.
echo 💬 Running Interactive Translator...
python testdich.py
goto completed

:completed
echo.
echo =========================================================
echo ✅ Action completed!
echo =========================================================
pause
exit /b 0

:end
echo 👋 Goodbye!
exit /b 0
