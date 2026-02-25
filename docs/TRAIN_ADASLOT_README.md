# AdaSlot Training - Standalone Script

Train AdaSlot chỉ với Task 1, không load compositional pipeline phức tạp.

## Tại sao cần file này?

`train_compositional.py` load cả pipeline (agents, VAEs, SLDA, etc.) → **quá nặng** nếu chỉ muốn train AdaSlot.

`train_adaslot.py` → **gọn nhẹ**, chỉ:
- Load data Task 1
- Train AdaSlot
- Save checkpoint
- Evaluate reconstruction

## Quick Start

### 1. Test nhanh (20 epochs, 200 samples)

```bash
./test_adaslot.bat
```

Hoặc manual:
```bash
python train_adaslot.py \
    --epochs 20 \
    --batch_size 32 \
    --use_primitive_loss \
    --device cuda \
    --test_mode \
    --max_samples 200
```

### 2. Train đầy đủ (50 epochs, full data)

```bash
./train_adaslot_primitive.bat
```

Hoặc manual:
```bash
python train_adaslot.py \
    --epochs 50 \
    --batch_size 64 \
    --use_primitive_loss \
    --primitive_alpha 10.0 \
    --primitive_temp 10.0 \
    --device cuda
```

### 3. So sánh có/không primitive loss

```bash
./compare_adaslot.bat
```

Chạy 2 runs liên tiếp và compare kết quả.

## Arguments Chính

### Model
- `--num_slots`: Số slots (default: 7)
- `--slot_dim`: Dimension của mỗi slot (default: 64)

### Data
- `--n_tasks`: Tổng số tasks (default: 10, chỉ dùng task 1)
- `--n_classes_per_task`: Classes mỗi task (default: 10)
- `--batch_size`: Batch size (default: 64)
- `--adaslot_resolution`: Image resolution (default: 128)

### Training
- `--epochs`: Số epochs (default: 50)
- `--lr`: Learning rate (default: 3e-4)
- `--sparse_weight`: Weight cho sparsity penalty (default: 1.0)
- `--save_interval`: Save checkpoint mỗi N epochs (default: 10)

### Primitive Loss (CompSLOT Paper - RECOMMENDED)
- `--use_primitive_loss`: Bật primitive loss
- `--primitive_alpha`: Weight α cho L_p (default: 10.0)
- `--primitive_temp`: Temperature τ_p (default: 10.0)

### Clustering Loss (Alternative)
- `--use_clustering_loss`: Bật clustering loss
- `--clustering_loss_type`: `contrastive` hoặc `prototype`
- `--clustering_weight`: Weight (default: 0.5)
- `--clustering_temp`: Temperature (default: 0.07)

### System
- `--device`: `cuda` hoặc `cpu`
- `--seed`: Random seed (default: 42)
- `--workers`: Số workers cho DataLoader (default: 4)

### Test Mode
- `--test_mode`: Bật test mode (data giới hạn)
- `--max_samples`: Max samples mỗi task (default: 200)

## Output Structure

```
checkpoints/adaslot_runs/
└── run_20260224_143052/
    ├── train.log              # Training logs
    ├── config.json            # Saved arguments
    ├── adaslot_epoch10.pt     # Checkpoint mỗi 10 epochs
    ├── adaslot_epoch20.pt
    ├── adaslot_epoch50.pt
    ├── adaslot_best.pt        # Best checkpoint (lowest loss)
    └── adaslot_final.pt       # Final checkpoint
```

## Checkpoint Format

```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': {...},
    'primitive_selector_state_dict': {...},  # Nếu dùng primitive loss
    'optimizer_state_dict': {...},
    'losses': {
        'total': 45.2,
        'recon': 40.1,
        'primitive': 0.51,
        'sparse': 0.02
    },
    'args': {...}
}
```

## Load Checkpoint

```python
import torch
from src.models.adaslot.model import AdaSlotModel
from src.losses.primitive import PrimitiveSelector

# Load checkpoint
ckpt = torch.load('checkpoints/adaslot_runs/run_xxx/adaslot_best.pt')

# Reconstruct model
model = AdaSlotModel(
    num_slots=ckpt['args']['num_slots'],
    slot_dim=ckpt['args']['slot_dim']
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Nếu có primitive selector
if 'primitive_selector_state_dict' in ckpt:
    selector = PrimitiveSelector(slot_dim=ckpt['args']['slot_dim'])
    selector.load_state_dict(ckpt['primitive_selector_state_dict'])
    selector.eval()
```

## Usage với Compositional Pipeline

Sau khi train xong, dùng checkpoint này cho `train_compositional.py`:

```bash
python train_compositional.py \
    --phase task1 \
    --adaslot_ckpt checkpoints/adaslot_runs/run_xxx/adaslot_final.pt
```

## Examples

### Example 1: Train với primitive loss (paper's approach)

```bash
python train_adaslot.py \
    --epochs 50 \
    --batch_size 64 \
    --num_slots 10 \
    --slot_dim 128 \
    --use_primitive_loss \
    --primitive_alpha 10.0 \
    --primitive_temp 10.0 \
    --device cuda
```

### Example 2: Train với clustering loss (legacy)

```bash
python train_adaslot.py \
    --epochs 50 \
    --batch_size 64 \
    --use_clustering_loss \
    --clustering_loss_type contrastive \
    --clustering_weight 0.5 \
    --device cuda
```

### Example 3: Test mode - quick experiment

```bash
python train_adaslot.py \
    --epochs 10 \
    --batch_size 32 \
    --use_primitive_loss \
    --test_mode \
    --max_samples 100 \
    --device cuda
```

### Example 4: CPU training (no GPU)

```bash
python train_adaslot.py \
    --epochs 20 \
    --batch_size 16 \
    --use_primitive_loss \
    --device cpu \
    --test_mode
```

## Expected Results

### Without Primitive Loss
```
Epoch 50/50 | total=42.3 | recon=40.1 | sparse=0.02
Test reconstruction loss: 41.5
```

### With Primitive Loss (BETTER)
```
Epoch 50/50 | total=45.2 | recon=40.1 | primitive=0.51 | sparse=0.02
Test reconstruction loss: 40.8
```

✅ **Lower test reconstruction loss** với primitive loss
✅ **More stable slots** across images of same class
✅ **Better concept discovery**

## Troubleshooting

### Out of Memory
```bash
# Giảm batch size
python train_adaslot.py --batch_size 32

# Giảm resolution
python train_adaslot.py --adaslot_resolution 64

# Giảm slots
python train_adaslot.py --num_slots 5
```

### Training too slow
```bash
# Dùng test mode
python train_adaslot.py --test_mode --max_samples 200

# Giảm workers nếu CPU bottleneck
python train_adaslot.py --workers 0
```

### Loss not decreasing
```bash
# Tăng learning rate
python train_adaslot.py --lr 5e-4

# Giảm primitive alpha nếu quá lớn
python train_adaslot.py --primitive_alpha 5.0

# Tăng epochs
python train_adaslot.py --epochs 100
```

## Comparison with train_compositional.py

| Feature | train_compositional.py | train_adaslot.py |
|---------|------------------------|------------------|
| Load data | All tasks | **Task 1 only** |
| Load pipeline | Agents, VAEs, SLDA, etc. | **None (lightweight)** |
| Training | AdaSlot → Task1 → Task2-N | **AdaSlot only** |
| Memory | Large | **Small** |
| Speed | Slow (setup overhead) | **Fast** |
| Use case | Full continual learning | **Just train AdaSlot** |

## Next Steps

1. **Train AdaSlot:**
   ```bash
   ./train_adaslot_primitive.bat
   ```

2. **Check results:**
   ```bash
   ls checkpoints/adaslot_runs/
   ```

3. **Use checkpoint for full pipeline:**
   ```bash
   python train_compositional.py \
       --phase task1 \
       --adaslot_ckpt checkpoints/adaslot_runs/run_xxx/adaslot_final.pt
   ```

## Tips

- ✅ Luôn dùng `--use_primitive_loss` (paper's approach)
- ✅ Dùng `--test_mode` khi experiment nhanh
- ✅ Set `--workers 0` nếu Windows có issue với multiprocessing
- ✅ Check `train.log` để xem training progress chi tiết
- ✅ So sánh `adaslot_best.pt` vs `adaslot_final.pt` (best thường tốt hơn)

## References

- CompSLOT Paper: Section 4.1 (Concept Learning)
- `PRIMITIVE_LOSS_GUIDE.md`: Chi tiết về primitive loss
- `src/losses/primitive.py`: Implementation
