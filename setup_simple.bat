@echo off
REM Simple setup script using pip (more reliable on Windows)
REM Author: Your Team
REM Date: 2026-02-12

echo ========================================
echo Continual Learning Environment Setup
echo Using pip for PyTorch (Windows compatible)
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo [1/6] Creating minimal conda environment...
call conda create -n continual-learning python=3.10 -y

echo.
echo [2/6] Activating environment...
call conda activate continual-learning

echo.
echo [3/6] Installing PyTorch with CUDA 12.1...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/6] Installing Avalanche and other dependencies...
pip install avalanche-lib numpy pandas scipy matplotlib seaborn pytest pytest-cov tqdm tensorboard wandb pyyaml

echo.
echo [5/6] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
python -c "import avalanche; print('Avalanche:', avalanche.__version__)"

echo.
echo [6/6] Testing data pipeline...
python demo_data_pipeline.py

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Environment: continual-learning
echo To activate: conda activate continual-learning
echo.
pause

