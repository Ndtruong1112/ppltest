param(
    [string]$VenvDir = "D:\Users\Admin\venvs\pplnckh-1-gpu",
    [string]$WorkRoot = "D:\Users\Admin\Downloads\PhoMT",
    [string]$PyTorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Python GPU Environment On Drive D" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

New-Item -ItemType Directory -Force -Path $VenvDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkRoot ".cache") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkRoot ".tmp") | Out-Null

Write-Host "Creating virtualenv: $VenvDir" -ForegroundColor Cyan
py -3 -m venv $VenvDir

$pythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Khong tao duoc python.exe trong virtualenv: $VenvDir"
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing PyTorch CUDA from: $PyTorchIndexUrl" -ForegroundColor Cyan
& $pythonExe -m pip install torch torchvision torchaudio --index-url $PyTorchIndexUrl

Write-Host "Installing project requirements..." -ForegroundColor Cyan
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host ""
Write-Host "Checking GPU..." -ForegroundColor Yellow
& $pythonExe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

Write-Host ""
Write-Host "Set these variables before running training:" -ForegroundColor Green
Write-Host "  `$env:PHOMT_VENV_DIR='$VenvDir'" -ForegroundColor Green
Write-Host "  `$env:PHOMT_WORK_ROOT='$WorkRoot'" -ForegroundColor Green
Write-Host "  `$env:PHOMT_CACHE_ROOT='$WorkRoot\.cache'" -ForegroundColor Green
Write-Host "  `$env:PHOMT_TMP_DIR='$WorkRoot\.tmp'" -ForegroundColor Green
