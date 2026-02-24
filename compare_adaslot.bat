@echo off
REM Compare: With vs Without Primitive Loss

conda activate continual-learning

echo ====================================
echo Training WITHOUT Primitive Loss...
echo ====================================
python train_adaslot.py ^
    --epochs 30 ^
    --batch_size 32 ^
    --device cuda ^
    --workers 0 ^
    --test_mode ^
    --max_samples 200

echo.
echo.
echo ====================================
echo Training WITH Primitive Loss...
echo ====================================
python train_adaslot.py ^
    --epochs 30 ^
    --batch_size 32 ^
    --use_primitive_loss ^
    --primitive_alpha 10.0 ^
    --primitive_temp 10.0 ^
    --device cuda ^
    --workers 0 ^
    --test_mode ^
    --max_samples 200

echo.
echo ====================================
echo Comparison complete!
echo Check both runs in checkpoints/adaslot_runs/
echo Compare final test reconstruction loss
echo ====================================
pause
