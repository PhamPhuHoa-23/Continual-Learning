@echo off
REM Quick setup script with CPU PyTorch (most reliable on Windows)
REM For CUDA version, see INSTALLATION_GUIDE.md

echo ========================================
echo Continual Learning Quick Setup
echo CPU-only PyTorch (reliable on Windows)
echo ========================================
echo.

where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Miniconda or Anaconda.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/5] Creating Python 3.10 environment...
call conda create -n continual-learning python=3.10 -y

echo.
echo [2/5] Activating environment...
call conda activate continual-learning

echo.
echo [3/5] Installing PyTorch (CPU-only)...
pip install torch torchvision torchaudio

echo.
echo [4/5] Installing Avalanche and dependencies...
pip install avalanche-lib numpy pandas scipy matplotlib seaborn pytest pytest-cov tqdm pyyaml

echo.
echo [5/5] Testing installation...
python test_avalanche.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Installation completed!
    echo ========================================
    echo.
    echo Next steps:
    echo   1. conda activate continual-learning
    echo   2. python demo_data_pipeline.py
    echo   3. pytest tests/data/ -v
    echo.
    echo To install CUDA version later:
    echo   See INSTALLATION_GUIDE.md
    echo.
) else (
    echo.
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo.
    echo Please check INSTALLATION_GUIDE.md for troubleshooting.
    echo.
)

pause

