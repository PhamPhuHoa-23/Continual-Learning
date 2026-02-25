# Multi-Phase Training Guide for AdaSlot

Hướng dẫn train AdaSlot theo từng phase cho continual learning.

## 📋 Tổng Quan

Training được chia thành **3 phases**:

1. **Phase 1 - Pretraining**: Học slot representations và primitive concepts
2. **Phase 2 - Continual Learning**: Học từng task tuần tự
3. **Phase 3 - Fine-tuning**: Consolidate tất cả tasks

## 🚀 Cách Sử Dụng

### Option 1: Train All Phases (Khuyến nghị)

```bash
python train_adaslot_multiphase.py --config configs/adaslot_multiphase_full.yaml --phase all
```

Script sẽ tự động train 3 phases liên tục:
- Phase 1: 50 epochs pretraining
- Phase 2: 20 epochs × 10 tasks
- Phase 3: 10 epochs consolidation

**Thời gian dự kiến**: ~12-16 giờ trên GPU

---

### Option 2: Train Từng Phase Riêng

#### Phase 1: Pretraining

```bash
python train_adaslot_multiphase.py \
    --config configs/adaslot_phase1_pretrain.yaml \
    --phase 1
```

**Mục đích**:
- Học slot attention mechanism
- Học primitive concepts thông qua primitive loss
- Reconstruction để đảm bảo slot quality

**Output**: `checkpoints/adaslot_phase1/phase1_final.pth`

---

#### Phase 2: Continual Learning

```bash
python train_adaslot_multiphase.py \
    --config configs/adaslot_phase2_continual.yaml \
    --phase 2 \
    --resume checkpoints/adaslot_phase1/phase1_final.pth
```

**Mục đích**:
- Train classifier trên từng task
- Preserve primitives đã học từ Phase 1
- Evaluate catastrophic forgetting

**Output**: 
- `checkpoints/adaslot_phase2/phase2_task1.pth`
- `checkpoints/adaslot_phase2/phase2_task2.pth`
- ...
- `checkpoints/adaslot_phase2/phase2_final.pth`

---

#### Phase 3: Fine-tuning

```bash
python train_adaslot_multiphase.py \
    --config configs/adaslot_phase3_finetune.yaml \
    --phase 3 \
    --resume checkpoints/adaslot_phase2/phase2_final.pth
```

**Mục đích**:
- Joint training trên tất cả tasks
- Improve overall accuracy
- Reduce forgetting

**Output**: `checkpoints/adaslot_phase3/phase3_final.pth`

---

## 📁 Cấu Trúc Config

Mỗi phase có config riêng:

```yaml
model:
  name: "adaslot_module"
  slot_dim: 64
  num_slots: 7
  num_primitives: 10
  # ...

phase1:  # hoặc phase2, phase3
  epochs: 50
  tasks: null  # null = all tasks
  # ...

training:
  batch_size: 64
  optimizer:
    lr: 0.0003
  # ...

losses:
  reconstruction:
    weight: 1.0
  primitive:
    weight: 10.0
  # ...
```

### Tùy Chỉnh Config

Bạn có thể modify các parameters:

**Model**:
- `slot_dim`: Dimension của mỗi slot (default: 64)
- `num_slots`: Số lượng slots (default: 7)
- `num_primitives`: Số primitive concepts (default: 10)

**Training**:
- `batch_size`: Batch size (default: 64)
- `lr`: Learning rate (Phase 1: 3e-4, Phase 2: 1e-4, Phase 3: 3e-5)

**Losses**:
- `primitive.weight`: Weight cho primitive loss (Phase 1: 10.0, Phase 2: 5.0)
- `sparsity.weight`: Khuyến khích drop slots (Phase 1: 0.1)

---

## 🎯 Expected Results

### Phase 1
- **Reconstruction Loss**: ~0.5-1.0 (MSE)
- **Primitive Loss**: ~5-10 (KL divergence)
- **Slot Selection**: Khoảng 3-5 slots được giữ (adaptive)

### Phase 2
- **Task 1 Accuracy**: ~70-80%
- **Task 10 Accuracy**: ~60-70% (sau khi học 10 tasks)
- **Average Accuracy**: ~65-75%
- **Forgetting**: ~10-20% (normal cho continual learning)

### Phase 3
- **Average Accuracy**: ~75-85% (sau consolidation)
- **Forgetting**: Giảm xuống ~5-10%

---

## 📊 Monitoring Training

Training logs được save tại:
```
checkpoints/adaslot_multiphase/run_YYYYMMDD_HHMMSS/
├── config.yaml               # Config đã dùng
├── phase1_train.log          # Training logs Phase 1
├── phase2_train.log          # Training logs Phase 2
├── phase3_train.log          # Training logs Phase 3
├── phase1_epoch10.pth        # Checkpoints
├── phase1_final.pth
├── phase2_task1.pth
├── ...
└── phase3_final.pth
```

### Real-time Monitoring

Training hiển thị progress bar với:
```
Epoch 1/50 | Task 1: 100%|████████| 782/782 [02:15<00:00, 5.77it/s, loss=0.8234, temp=0.980]
```

- `loss`: Current batch loss
- `temp`: Gumbel temperature (giảm dần từ 1.0 → 0.5)
- `acc`: Accuracy (chỉ hiện trong Phase 2/3)

---

## 🔧 Troubleshooting

### Out of Memory

Giảm batch size:
```yaml
training:
  batch_size: 32  # hoặc 16
```

### Training quá chậm

1. Tăng `num_workers`:
```yaml
training:
  num_workers: 8
```

2. Giảm số epochs:
```yaml
phase1:
  epochs: 20  # thay vì 50
```

### Model không converge

1. Tăng learning rate:
```yaml
training:
  optimizer:
    lr: 0.001  # 10x lớn hơn
```

2. Giảm primitive loss weight:
```yaml
losses:
  primitive:
    weight: 5.0  # thay vì 10.0
```

---

## 🎓 Advanced Usage

### Custom Dataset

Modify `data` section:
```yaml
data:
  dataset: "tiny_imagenet"  # hoặc "cgqa", "cobj"
  n_experiences: 20
```

### Different Phase Orders

Train Phase 2 trước, skip Phase 1:
```bash
python train_adaslot_multiphase.py \
    --config configs/adaslot_phase2_continual.yaml \
    --phase 2
```

### Multiple Runs

Tạo scripts cho sweep:
```bash
# sweep_lr.sh
for lr in 0.0001 0.0003 0.001; do
    python train_adaslot_multiphase.py \
        --config configs/adaslot_multiphase_full.yaml \
        --phase all \
        --output_dir "checkpoints/sweep_lr_${lr}"
done
```

---

## 📚 References

- **AdaSlot Paper**: Adaptive slot attention with Gumbel-Softmax
- **CompSLOT Paper**: Primitive loss for compositional learning
- **Avalanche**: Continual learning benchmark framework

---

## ✅ Quick Test

Test script với ít data:
```bash
# Thêm vào config:
training:
  test_mode: true
  max_samples: 200  # chỉ dùng 200 samples
```

Hoặc tạo simple test:
```bash
python test_adaslot_pipeline.py
```
