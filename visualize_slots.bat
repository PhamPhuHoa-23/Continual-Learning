@echo off
REM Visualize AdaSlot Slots

conda activate continual-learning

REM Update this path to your checkpoint
set CHECKPOINT=checkpoints/adaslot_runs/run_20260224_XXXXXX/adaslot_best.pt

python visualize_adaslot.py ^
    --checkpoint %CHECKPOINT% ^
    --num_images 8 ^
    --output_dir visualizations ^
    --device cuda

echo.
echo ====================================
echo Visualization saved to: visualizations/
echo ====================================
pause
