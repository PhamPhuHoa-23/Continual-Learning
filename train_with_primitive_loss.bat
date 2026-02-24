@echo off
REM Example: Train AdaSlot with Primitive Loss (CompSLOT paper approach)

conda activate continual-learning

python train_compositional.py ^
    --phase adaslot ^
    --n_tasks 10 ^
    --n_classes_per_task 10 ^
    --adaslot_epochs 50 ^
    --batch_size 64 ^
    --device cuda ^
    --workers 0 ^
    --use_primitive_loss ^
    --primitive_alpha 10.0 ^
    --primitive_temp 10.0 ^
    --test_mode ^
    --max_samples 200

echo.
echo ====================================
echo AdaSlot training with primitive loss completed!
echo Check the output in checkpoints/compositional_runs/
echo ====================================
pause
