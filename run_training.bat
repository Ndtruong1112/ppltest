@echo off
REM -*- coding: utf-8 -*-
REM Script để chạy training trên GPU - Windows

cd /d %~dp0

set "VENV_DIR=%PHOMT_VENV_DIR%"
if "%VENV_DIR%"=="" if exist "D:\Users\Admin\Downloads\PhoMT\.venv\Scripts\activate.bat" set "VENV_DIR=D:\Users\Admin\Downloads\PhoMT\.venv"
if "%VENV_DIR%"=="" if exist ".venv\Scripts\activate.bat" set "VENV_DIR=%cd%\.venv"

echo ========================================
echo GPU Training Runner - PhoMT IT Translation
echo ========================================
echo.

REM Activate virtual environment
if "%VENV_DIR%"=="" (
    echo ERROR: Khong tim thay virtualenv. Hay chay setup_env_d.ps1 truoc.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Khong tim thay file kich hoat virtualenv: %VENV_DIR%\Scripts\activate.bat
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM Set encoding
set PYTHONIOENCODING=utf-8
if "%PHOMT_WORK_ROOT%"=="" set "PHOMT_WORK_ROOT=D:\Users\Admin\Downloads\PhoMT"
if "%PHOMT_CACHE_ROOT%"=="" set "PHOMT_CACHE_ROOT=D:\Users\Admin\Downloads\PhoMT\.cache"
if "%PHOMT_TMP_DIR%"=="" set "PHOMT_TMP_DIR=D:\Users\Admin\Downloads\PhoMT\.tmp"

REM Check if training script exists
if not exist "train2.py" (
    echo ERROR: train2.py not found!
    pause
    exit /b 1
)

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU - SLOW!\"}') if torch.cuda.is_available() else print('⚠️ GPU NOT available - using CPU')"
echo.
echo Venv: %VENV_DIR%
echo Work root: %PHOMT_WORK_ROOT%
echo Cache root: %PHOMT_CACHE_ROOT%
echo.

REM Ask for confirmation
echo.
echo Press ENTER to start training, or CTRL+C to cancel...
pause

REM Run training
echo.
echo Starting training...
echo.
python train2.py

echo.
echo ========================================
echo Training completed!
echo ========================================
pause
