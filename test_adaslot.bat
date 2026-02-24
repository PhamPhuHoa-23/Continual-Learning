@echo off
REM Quick Test - Train AdaSlot with Primitive Loss

conda activate continual-learning

python train_adaslot.py ^
    --epochs 20 ^
    --batch_size 32 ^
    --num_slots 7 ^
    --slot_dim 64 ^
    --use_primitive_loss ^
    --primitive_alpha 10.0 ^
    --primitive_temp 10.0 ^
    --device cuda ^
    --workers 0 ^
    --test_mode ^
    --max_samples 200 ^
    --save_interval 5

echo.
echo ====================================
echo Quick test completed!
echo Check: checkpoints/adaslot_runs/
echo ====================================
pause
