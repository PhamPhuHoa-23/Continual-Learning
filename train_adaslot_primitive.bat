@echo off
REM Train AdaSlot with Primitive Loss (Simple Script)

conda activate continual-learning

python train_adaslot.py ^
    --epochs 50 ^
    --batch_size 64 ^
    --num_slots 7 ^
    --slot_dim 64 ^
    --use_primitive_loss ^
    --primitive_alpha 10.0 ^
    --primitive_temp 10.0 ^
    --device cuda ^
    --workers 0

echo.
echo ====================================
echo Training completed!
echo Check: checkpoints/adaslot_runs/
echo ====================================
pause
