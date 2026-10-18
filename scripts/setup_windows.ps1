# One-shot environment setup for a Windows PC with an NVIDIA GPU.
# Run from PowerShell:
#   irm https://raw.githubusercontent.com/kylewkchuan/MemoCMT/main/scripts/setup_windows.ps1 | iex
# Safe to re-run: finished steps are skipped.

$root  = "$env:USERPROFILE\miniconda3"
$conda = "$root\Scripts\conda.exe"
$py    = "$root\envs\MemoCMT-cuda\python.exe"
$repo  = "$env:USERPROFILE\MemoCMT"

if (-not (Test-Path $conda)) {
    Write-Host "=== [1/5] Installing Miniconda ===" -ForegroundColor Cyan
    curl.exe -L -o "$env:TEMP\miniconda.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    Start-Process -Wait -FilePath "$env:TEMP\miniconda.exe" -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$root"
} else { Write-Host "=== [1/5] Miniconda already installed ===" -ForegroundColor Cyan }

if (-not (Test-Path $py)) {
    Write-Host "=== [2/5] Creating Python 3.10 env ===" -ForegroundColor Cyan
    # conda-forge only: avoids the Anaconda Terms-of-Service prompt
    & $conda create -n MemoCMT-cuda -c conda-forge --override-channels python=3.10 -y
} else { Write-Host "=== [2/5] Env already exists ===" -ForegroundColor Cyan }

if (-not (Test-Path $py)) { Write-Host "FAILED: Python env was not created." -ForegroundColor Red; return }

Write-Host "=== [3/5] Installing PyTorch (CUDA 11.8) ===" -ForegroundColor Cyan
# cu118 works with older drivers (e.g. 555.x) on Ampere/Ada laptops
& $py -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

if (-not (Test-Path $repo)) { git clone https://github.com/kylewkchuan/MemoCMT.git $repo }

Write-Host "=== [4/5] Installing other packages ===" -ForegroundColor Cyan
& $py -m pip install "numpy<2" librosa -r "$repo\requirements.txt"

Write-Host "=== [5/5] Checking GPU ===" -ForegroundColor Cyan
& $py -c "import torch; ok = torch.cuda.is_available(); print('RESULT:', torch.__version__, ok, torch.cuda.get_device_name(0) if ok else 'NO GPU')"
Write-Host "Python for this project: $py" -ForegroundColor Green
