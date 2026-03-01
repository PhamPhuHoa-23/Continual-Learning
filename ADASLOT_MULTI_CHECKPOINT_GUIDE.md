# AdaSlot Multi-Checkpoint Training Guide

## 📁 Các file mới

### 1. `cont_src/models/adaslot_configs.py` 
**Registry cho các checkpoint AdaSlot khác nhau**

Định nghĩa config cho 4 checkpoint pretrained:

| Checkpoint | Dataset | Num Slots | Slot Dim | Mô tả |
|-----------|---------|-----------|----------|-------|
| `clevr10` | CLEVR | 11 | 64 | Synthetic 3D scenes, 10 objects max |
| `coco` | MS COCO | 7 | 64 | Real-world photos, complex backgrounds |
| `movic` | MOVi-C | 11 | 64 | Video, 3–10 textured objects |
| `movie` | MOVi-E | 24 | 64 | Video, up to 23 objects (high complexity) |

**Sử dụng:**

```python
from cont_src.models.adaslot_configs import get_adaslot_config, build_adaslot_from_checkpoint

# Xem config của checkpoint
cfg = get_adaslot_config("clevr10")
print(f"Num slots: {cfg.num_slots}, Slot dim: {cfg.slot_dim}")

# Load model + pretrained weights
model = build_adaslot_from_checkpoint(
    checkpoint_name="clevr10",
    ckpt_path="checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt",
    device="cuda",
    reset_gumbel_gate=False,  # ĐỂ False để GIỮ pretrained gate!
)
```

---

### 2. `train_adaslot_no_reset.py`
**Script huấn luyện AdaSlot KHÔNG reset gate**

Dùng script này thay vì `kaggle_train_adaslot.py` khi bạn muốn:
- ✅ **Giữ nguyên pretrained Gumbel gate** (không reset)
- ✅ **Dễ dàng thử các checkpoint khác** (COCO, MOVi-C, MOVi-E)
- ✅ **Train local** trên CIFAR-100 hoặc Tiny-ImageNet

#### Ví dụ sử dụng:

```bash
# 1. Train với CLEVR10 checkpoint (default) - GIỮ GATE
python train_adaslot_no_reset.py \
    --checkpoint clevr10 \
    --dataset cifar100 \
    --epochs 5 \
    --lr 4e-5 \
    --batch_size 64

# 2. Thử COCO checkpoint (7 slots) - phù hợp với real-world images
python train_adaslot_no_reset.py \
    --checkpoint coco \
    --ckpt_path checkpoints/slot_attention/adaslot_real/COCO.ckpt \
    --dataset cifar100 \
    --epochs 5

# 3. MOVi-E checkpoint (24 slots) - phù hợp với scenes phức tạp
python train_adaslot_no_reset.py \
    --checkpoint movie \
    --dataset tiny_imagenet \
    --epochs 3

# 4. RESET gate (nếu domain shift quá lớn) - thêm flag --reset_gate
python train_adaslot_no_reset.py \
    --checkpoint clevr10 \
    --dataset cifar100 \
    --reset_gate \
    --epochs 5
```

#### Arguments quan trọng:

| Arg | Mặc định | Mô tả |
|-----|----------|-------|
| `--checkpoint` | `clevr10` | Chọn: `clevr10`, `coco`, `movic`, `movie` |
| `--ckpt_path` | `None` | Đường dẫn .ckpt (nếu None, dùng default) |
| `--reset_gate` | `False` | Thêm flag này để **RESET gate** (mất pretrained behaviour) |
| `--dataset` | `cifar100` | Dataset: `cifar100`, `tiny_imagenet` |
| `--epochs` | `5` | Số epochs train |
| `--lr` | `4e-5` | Learning rate |
| `--w_recon` | `1.0` | Reconstruction loss weight |
| `--w_sparse` | `10.0` | Sparsity loss weight |
| `--w_prim` | `5.0` | Primitive loss weight |

---

### 3. `test_adaslot_registry.py`
**Test script để xem các checkpoint có sẵn**

```bash
# Xem tất cả configs
python test_adaslot_registry.py

# Test load từng checkpoint (nếu file tồn tại)
python test_adaslot_registry.py --test_load
```

Output:
```
================================================================================
                       AdaSlot Checkpoint Registry                       
================================================================================

Total registered checkpoints: 4
Names: clevr10, coco, movic, movie

────────────────────────────────────────────────────────────────────────────────
Checkpoint: CLEVR10
────────────────────────────────────────────────────────────────────────────────
  Source Dataset : CLEVR
  Description    : Pretrained on CLEVR dataset (synthetic 3D scenes, 10 objects max)

  Architecture:
    Resolution   : 128 × 128
    Num Slots    : 11
    Slot Dim     : 64
    Feature Dim  : 64
    Iterations   : 3
    KVQ Dim      : 128
    Low Bound    : 1

  Default Path   : checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt
  Status         : ✓ Found
  
  Loading checkpoint...
  ✓ Model loaded successfully  (4,567,890 params)
  ✓ Forward pass OK  (avg active slots: 7.3/11)
...
```

---

## 🔧 Cách tích hợp vào `kaggle_train_adaslot.py`

Nếu muốn update file Kaggle hiện tại:

### Bước 1: Thay đổi section CONFIG

**Trước (line 123-135):**
```python
# CHECKPOINT  (upload CLEVR10.ckpt len Kaggle dataset roi dien path)
CKPT_PATH = "/kaggle/input/adaslot-clevr10/CLEVR10.ckpt"
if not _os.path.exists(CKPT_PATH):
    CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"

# MODEL
IMG_SIZE  = 128   # decoder la 4x stride-2 ConvTranspose -> output 128x128
NUM_SLOTS = 7
SLOT_DIM  = 64
D_H       = 64
```

**Sau:**
```python
# CHECKPOINT — chon pretrained AdaSlot checkpoint
# Available: "clevr10" (11 slots), "coco" (7 slots), "movic" (11 slots), "movie" (24 slots)
CKPT_NAME = "clevr10"  # checkpoint name from registry
CKPT_PATH = "/kaggle/input/adaslot-clevr10/CLEVR10.ckpt"
if not _os.path.exists(CKPT_PATH):
    CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"

RESET_GUMBEL_GATE = False  # True = reset gate (discard learned slot selection)
                           # False = preserve gate (keep pretrained behaviour)

# MODEL — auto-loaded from checkpoint config
# IMG_SIZE, NUM_SLOTS, SLOT_DIM are loaded from registry based on CKPT_NAME.
D_H = 64  # agent hidden dim / aggregator output dim
```

### Bước 2: Update imports (line 270)

**Thêm import:**
```python
from cont_src.models.adaslot_configs import (
    get_adaslot_config,
    build_adaslot_from_checkpoint,
    list_available_checkpoints,
)
```

### Bước 3: Thay thế section "Build Model" (line 368-440)

**Xóa code cũ:**
```python
backbone = AdaSlotModel(
    resolution=(IMG_SIZE, IMG_SIZE),
    num_slots=NUM_SLOTS,
    slot_dim=SLOT_DIM,
    num_iterations=3,
    feature_dim=SLOT_DIM,
    kvq_dim=128,
    low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
missing, unexpected = backbone.load_state_dict(ckpt["state_dict"], strict=True)
print(f"Checkpoint loaded: {CKPT_PATH}")
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

# Reset gate: ...
def _reset_gumbel_gate(model: nn.Module) -> None:
    ...

_reset_gumbel_gate(backbone)
print("Gate weights re-initialised")
```

**Thay bằng:**
```python
# ── Load from checkpoint registry (auto-config) ────────────────────────────
print("Available checkpoints:", list_available_checkpoints())
print(f"Selected checkpoint: {CKPT_NAME}")

ckpt_config = get_adaslot_config(CKPT_NAME)
print(f"Config: {ckpt_config.source_dataset}")
print(f"  Resolution: {ckpt_config.resolution}")
print(f"  Num slots : {ckpt_config.num_slots}")
print(f"  Slot dim  : {ckpt_config.slot_dim}")

# Build + load weights (NO gate reset by default)
backbone = build_adaslot_from_checkpoint(
    checkpoint_name=CKPT_NAME,
    ckpt_path=CKPT_PATH,
    device=str(DEVICE),
    strict_load=True,
    reset_gumbel_gate=RESET_GUMBEL_GATE,
)

if RESET_GUMBEL_GATE:
    print("⚠ Gate was RESET — pretrained slot selection discarded")
else:
    print("✓ Gate PRESERVED — using pretrained slot selection behaviour")

# Extract loaded config
IMG_SIZE  = ckpt_config.resolution[0]
NUM_SLOTS = ckpt_config.num_slots
SLOT_DIM  = ckpt_config.slot_dim
```

---

## 🧪 So sánh: Reset vs Không Reset Gate

### ❌ Reset Gate (`RESET_GUMBEL_GATE = True`)

**Khi nào dùng:**
- Domain shift cực lớn (CLEVR synthetic → CIFAR-100 real-world)
- Muốn gate học từ đầu

**Ưu điểm:**
- Gate sẽ học slot selection phù hợp với domain mới

**Nhược điểm:**
- Mất hết pretrained knowledge về slot selection
- Cần nhiều epochs hơn để gate converge
- Có thể unstable ban đầu (keep_prob ~1.0 hoặc ~0.0)

### ✅ Không Reset Gate (`RESET_GUMBEL_GATE = False`) **← RECOMMENDED**

**Khi nào dùng:**
- Muốn giữ pretrained slot selection behaviour
- Domain shift vừa phải (CLEVR → CIFAR hoặc COCO → CIFAR)
- Fine-tuning ngắn (< 5 epochs)

**Ưu điểm:**
- Giữ được knowledge về slot selection từ pretraining
- Stable training từ đầu
- Converge nhanh hơn
- Keep_prob hợp lý ngay lập tức

**Nhược điểm:**
- Nếu domain shift quá lớn, gate có thể không optimal cho domain mới

---

## 📊 Khi nào dùng checkpoint nào?

| Checkpoint | Num Slots | Best For | Example Use Case |
|-----------|-----------|----------|------------------|
| **clevr10** | 11 | Scenes có nhiều objects rời rạc | Object detection, scene decomposition |
| **coco** | 7 | Real-world photos, ít objects | Classification, features extraction |
| **movic** | 11 | Video frames, moderate complexity | Video understanding, continual learning |
| **movie** | 24 | Very complex scenes | Dense object detection, crowd analysis |

**Gợi ý cho CIFAR-100:**
- Bắt đầu với **`coco`** (7 slots) - phù hợp với real-world images, ít background clutter
- Nếu cần thêm capacity: dùng **`clevr10`** (11 slots) hoặc **`movic`** (11 slots)

**Gợi ý cho Tiny-ImageNet:**
- Dùng **`movic`** (11 slots) hoặc **`movie`** (24 slots) - images phức tạp hơn CIFAR

---

## ✅ TODO / Next Steps

- [ ] Test COCO checkpoint trên CIFAR-100
- [ ] Test MOVi-E (24 slots) trên Tiny-ImageNet  
- [ ] Compare performance: reset gate vs preserve gate
- [ ] Add support cho custom checkpoint paths trong Kaggle
- [ ] Viết guide về khi nào nên thay đổi `w_sparse` và `w_prim`

---

## 📝 Code Examples

### Example 1: So sánh 2 checkpoints

```python
# Test CLEVR10 (11 slots, pretrained gate)
!python train_adaslot_no_reset.py \
    --checkpoint clevr10 \
    --epochs 3 \
    --exp_name "clevr10_no_reset"

# Test COCO (7 slots, pretrained gate)
!python train_adaslot_no_reset.py \
    --checkpoint coco \
    --epochs 3 \
    --exp_name "coco_no_reset"

# Compare results
import json
with open("checkpoints/adaslot_finetuned/clevr10_no_reset/history.json") as f:
    h1 = json.load(f)
with open("checkpoints/adaslot_finetuned/coco_no_reset/history.json") as f:
    h2 = json.load(f)

print(f"CLEVR10 final loss: {h1['total_loss'][-1]:.4f}")
print(f"COCO final loss: {h2['total_loss'][-1]:.4f}")
```

### Example 2: Load model trong notebook

```python
from cont_src.models.adaslot_configs import build_adaslot_from_checkpoint

# Load COCO checkpoint
model = build_adaslot_from_checkpoint(
    checkpoint_name="coco",
    device="cuda",
    reset_gumbel_gate=False  # GIỮ pretrained gate
)

# Forward pass
import torch
imgs = torch.randn(4, 3, 128, 128).cuda()
out = model(imgs)

print(f"Active slots: {out['hard_keep_decision'].sum(1).float().mean().item():.1f}/7")
```

---

**Tóm lại:** 
- Dùng `train_adaslot_no_reset.py` để **GIỮ pretrained gate**  
- Dùng `RESET_GUMBEL_GATE = False` trong `kaggle_train_adaslot.py`
- Thử các checkpoint khác nhau: `clevr10`, `coco`, `movic`, `movie`
- Registry tự động load đúng architecture cho mỗi checkpoint
