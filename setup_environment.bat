@echo off
REM Setup script for Continual Learning environment (Windows)
REM Author: Your Team
REM Date: 2026-02-12

echo ========================================
echo Continual Learning Environment Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo [1/5] Creating conda environment from environment.yml...
call conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment already exists. Updating instead...
    call conda env update -f environment.yml --prune
)

echo.
echo [2/5] Activating environment...
call conda activate continual-learning

echo.
echo [3/5] Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import avalanche; print(f'Avalanche version: {avalanche.__version__}')"

echo.
echo [4/5] Testing CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo [5/5] Running data pipeline tests...
pytest tests/data/test_continual_cifar100.py -v --tb=short

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the environment, run:
echo     conda activate continual-learning
echo.
echo To run tests:
echo     pytest tests/data/ -v
echo.
echo To run demo:
echo     python src/data/continual_cifar100.py
echo.
pause

