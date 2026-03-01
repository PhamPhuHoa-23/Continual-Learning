# DINOv2 Training Details (From Official Source Code)

Source: `Setup/dinov2/` (Facebook Research Official Implementation)

## 1. STUDENT-TEACHER ARCHITECTURE ⚠️

### Core Components
```
Input Image
    ↓
┌─────────────────────────────────┐
│  Multi-Crop Augmentation        │
│  - 2 Global Crops (224×224)     │
│  - 8 Local Crops (96×96)        │
└─────────────────────────────────┘
    ↓                    ↓
┌──────────┐        ┌──────────┐
│ Student  │        │ Teacher  │
│ (ViT)    │        │ (ViT)    │
│ Trainable│        │ EMA only │
└──────────┘        └──────────┘
    ↓                    ↓
┌──────────┐        ┌──────────┐
│DINO Head │        │DINO Head │
│(3 layers)│        │(3 layers)│
└──────────┘        └──────────┘
    ↓                    ↓
  Student              Teacher
  Output              Output
     ↓                   ↓
  ┌────────────────────────┐
  │   Cross-Entropy Loss   │
  │ + iBOT Masked Loss     │
  │ + KoLeo Regularization │
  └────────────────────────┘
```

### Critical Details ⚠️

**Teacher Update (EMA):**
```python
# Teacher parameters θ_t are updated by exponential moving average:
θ_teacher = momentum * θ_teacher + (1 - momentum) * θ_student

# Momentum schedule:
# - Start: 0.992
# - End: 1.0 (teacher becomes frozen)
# - Warmup: 30 epochs for temperature
```

**Why This Matters:**
- Teacher provides stable targets (no gradient)
- Student learns to match teacher's output
- Prevents mode collapse in self-supervised learning

---

## 2. MULTI-CROP STRATEGY ⚠️

### Configuration (from `ssl_default_config.yaml`)

```yaml
crops:
  # Global crops: full views of the image
  global_crops_size: 224
  global_crops_scale: [0.32, 1.0]  # Crop 32%-100% of image area
  
  # Local crops: zoomed-in views
  local_crops_number: 8
  local_crops_size: 96
  local_crops_scale: [0.05, 0.32]  # Crop 5%-32% of image area
```

### Training Forward Pass
```python
# For EACH image, generate 10 crops:
# - 2 global crops → passed through BOTH student and teacher
# - 8 local crops → passed through STUDENT ONLY

# Loss computation:
# - Student global crop 1 vs Teacher global crop 2
# - Student global crop 2 vs Teacher global crop 1
# - Student local crop i vs Teacher global crop 1
# - Student local crop i vs Teacher global crop 2
# For i = 1, 2, ..., 8
```

**Why This Matters:**
- Forces network to learn global + local features
- Increases data efficiency (10x augmented views per image)
- Local crops never go to teacher (saves memory)

---

## 3. DATA AUGMENTATION ⚠️

### From `dinov2/data/augmentations.py`

```python
# Global Crop 1:
RandomResizedCrop(224, scale=(0.32, 1.0))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(p=1.0)  # Always blur for global crop 1
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Global Crop 2:
RandomResizedCrop(224, scale=(0.32, 1.0))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(p=0.1)  # Sometimes blur
RandomSolarize(threshold=128, p=0.2)  # Sometimes solarize
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Local Crops (8x):
RandomResizedCrop(96, scale=(0.05, 0.32))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(p=0.5)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Critical Notes:**
- Different augmentations for global crop 1 vs 2
- Always normalize with ImageNet stats
- Strong augmentations prevent shortcuts

---

## 4. DINO LOSS (Cross-Entropy with Centering) ⚠️

### From `dinov2/loss/dino_clstoken_loss.py`

```python
# Student output: raw logits (no centering)
# Teacher output: softmax after centering

# Teacher processing:
teacher_probs = softmax((teacher_logits - center) / teacher_temp)

# Student processing:
student_log_probs = log_softmax(student_logits / student_temp)

# Loss (cross-entropy):
loss = -sum(teacher_probs * student_log_probs)

# Center update (exponential moving average):
center = 0.9 * center + 0.1 * mean(teacher_logits)
```

### Temperature Schedule
```yaml
warmup_teacher_temp: 0.04      # Start temperature
teacher_temp: 0.07              # Final temperature
warmup_teacher_temp_epochs: 30  # Warmup duration
student_temp: 0.1               # Fixed student temperature
```

**Why This Matters:**
- **Centering prevents mode collapse**: Without centering, all outputs could converge to the same value
- **Temperature controls sharpness**: Low temp → sharper distribution
- **Teacher temp warmup**: Gradual sharpening helps stability

---

## 5. DINO HEAD ARCHITECTURE ⚠️

### From `dinov2/layers/dino_head.py` and config

```python
DINOHead(
    in_dim=1024,           # ViT-Large embed_dim
    hidden_dim=2048,       # First projection
    bottleneck_dim=256,    # Bottleneck
    out_dim=65536,         # Number of prototypes (HUGE!)
    nlayers=3              # 3-layer MLP
)

# Architecture:
# input (1024) → Linear → GELU → Linear (2048)
#              → LayerNorm → Linear → GELU → Linear (256)
#              → LayerNorm → Linear (65536)
#              → L2 Normalize → Weight Normalize
```

**Critical Details:**
```yaml
dino:
  head_n_prototypes: 65536        # 65K prototypes!
  head_bottleneck_dim: 256
  head_hidden_dim: 2048
  head_nlayers: 3
  loss_weight: 1.0
  koleo_loss_weight: 0.1
```

**Why 65536 Prototypes?**
- Prevents trivial solutions (too few prototypes → mode collapse)
- Allows fine-grained distinctions
- Requires large batch sizes (Meta trained with 1024+ batch size)

---

## 6. IBOT LOSS (Masked Image Modeling) ⚠️

### From `dinov2/loss/ibot_patch_loss.py`

```python
# Randomly mask 10%-50% of patches in student
# Teacher sees full image (no masking)
# Student predicts teacher's output for masked patches

ibot:
  loss_weight: 1.0
  mask_sample_probability: 0.5      # 50% of images are masked
  mask_ratio_min_max: [0.1, 0.5]    # Mask 10%-50% of patches
  separate_head: false               # Share head with DINO
```

**Why This Matters:**
- Forces network to learn from local patches
- Combines with DINO's global view
- Similar to MAE/BEiT but with teacher

---

## 7. KOLEO LOSS (Regularization) ⚠️

### From `dinov2/loss/koleo_loss.py`

```python
# KoLeo = Kozachenko-Leonenko entropy estimator
# Encourages uniform distribution in embedding space
# Prevents embeddings from collapsing to a few modes

koleo_loss = -log(distance_to_nearest_neighbor)

koleo_loss_weight: 0.1  # Relatively small weight
```

**Why This Matters:**
- Prevents dimensional collapse
- Ensures diversity in learned representations
- Particularly important with large prototype counts

---

## 8. TRAINING HYPERPARAMETERS ⚠️

### From `ssl_default_config.yaml`

```yaml
optim:
  epochs: 100
  base_lr: 0.004                    # For batch size 1024
  lr: 0.0                            # Scaled by rule
  scaling_rule: sqrt_wrt_1024        # lr = base_lr * sqrt(batch_size / 1024)
  warmup_epochs: 10
  min_lr: 1e-6
  
  weight_decay: 0.04                 # Start
  weight_decay_end: 0.4              # End (10x increase!)
  
  clip_grad: 3.0                     # Gradient clipping
  freeze_last_layer_epochs: 1        # Freeze head for stability
  
  adamw_beta1: 0.9
  adamw_beta2: 0.999
  
  # Layer-wise learning rate decay
  layerwise_decay: 0.9               # Each layer gets 0.9x LR of previous
  patch_embed_lr_mult: 0.2           # Patch embed gets 0.2x LR

train:
  batch_size_per_gpu: 64
  # Meta trained with 8-64 GPUs → total batch size 512-4096
```

**Critical Observations:**
1. **Weight decay schedule**: Starts low (0.04), ends high (0.4)
2. **Learning rate scaling**: Must scale with batch size
3. **Gradient clipping**: Prevents instability with large batches
4. **Freeze last layer**: First epoch stabilizes prototypes
5. **Layer-wise decay**: Lower layers learn slower

---

## 9. MIXED PRECISION TRAINING ⚠️

### From `ssl_default_config.yaml`

```yaml
compute_precision:
  grad_scaler: true  # Use automatic mixed precision
  
  teacher:
    backbone:
      mixed_precision:
        param_dtype: fp16    # Parameters in fp16
        reduce_dtype: fp16   # Gradient reduce in fp16
        buffer_dtype: fp32   # Buffers in fp32
  
  student:
    backbone:
      mixed_precision:
        param_dtype: fp16
        reduce_dtype: fp32   # Student reduces in fp32!
        buffer_dtype: fp32
```

**Why This Matters:**
- Faster training (2-3x speedup)
- Lower memory (2x reduction)
- Requires careful setup to avoid numerical issues

---

## 10. WHAT YOU CANNOT SKIP ⚠️⚠️⚠️

### Absolutely Required

1. **Student-Teacher with EMA**
   - Teacher momentum: 0.992 → 1.0
   - Teacher gets no gradients, only EMA updates

2. **Multi-Crop Strategy**
   - 2 global + 8 local crops
   - Local crops only to student

3. **Centering in Loss**
   - Center = EMA of teacher outputs
   - Prevents mode collapse

4. **Temperature Schedule**
   - Teacher temp warmup: 0.04 → 0.07
   - Student temp: fixed at 0.1

5. **Large Prototype Count**
   - At least 4096-8192 prototypes
   - 65536 is ideal but requires large batch

### Highly Recommended

6. **iBOT Masking**
   - Adds local-to-global consistency
   - 10-50% masking ratio

7. **Strong Augmentations**
   - Color jitter, blur, solarize
   - Different augmentations for 2 global crops

8. **Gradient Clipping**
   - Clip norm at 3.0
   - Critical for stability

9. **Weight Decay Schedule**
   - Increase from 0.04 to 0.4
   - Prevents overfitting in later epochs

### Nice to Have

10. **KoLeo Loss** (if you have resources)
11. **Layer-wise LR decay** (improves fine-tuning)
12. **Mixed precision** (faster training)

---

## 11. ADAPTATION FOR YOUR CASE (Continual Learning with Slot Attention)

### Your Architecture
```
Input Image (32×32)
    ↓
Slot Attention (decompose into K slots)
    ↓
For each slot:
    ↓
Select top-k agents (via sub-network estimators)
    ↓
Each agent processes slot → hidden label (like DINO)
    ↓
Aggregate hidden labels → Decision Tree
```

### How to Adapt DINOv2 Training

#### Option 1: Full DINO-style Training (Recommended)
```python
# For each agent (50 agents):
# 1. Use student-teacher architecture
# 2. Train with multi-crop (2 global + 8 local)
# 3. Apply DINO loss with centering
# 4. Use large output dimension (e.g. 4096)

# Training loop:
for image, label in dataloader:
    # Generate multi-crops
    global_crops = [aug1(image), aug2(image)]  # 2 global
    local_crops = [aug_local(image) for _ in range(8)]  # 8 local
    
    # Slot Attention
    slots_global = [slot_attention(crop) for crop in global_crops]
    slots_local = [slot_attention(crop) for crop in local_crops]
    
    # For each slot, select agents
    for slot in slots_global[0]:  # Use first global crop
        # Estimate performance
        scores = [estimator(slot, agent_i) for agent_i in range(50)]
        
        # Select top-k agents
        top_k_indices = topk(scores, k=3)
        
        # Student forward (all crops)
        student_outputs = []
        for agent_idx in top_k_indices:
            student_outputs.append(agents[agent_idx](slot))
        
        # Teacher forward (only global crops)
        with torch.no_grad():
            teacher_outputs = []
            for agent_idx in top_k_indices:
                teacher_outputs.append(teacher_agents[agent_idx](slot))
        
        # DINO loss
        loss += dino_loss(student_outputs, teacher_outputs)
    
    # Update student
    loss.backward()
    optimizer.step()
    
    # Update teacher (EMA)
    update_teacher(student_agents, teacher_agents, momentum=0.996)
```

#### Option 2: Simplified (If Resources Limited)
```python
# Single network (no teacher), supervised training
# Use DINO-style embeddings but train with labels
# Use contrastive loss instead of DINO loss

# Training loop:
for image, label in dataloader:
    slots = slot_attention(image)
    
    # For each slot
    embeddings = []
    for slot in slots:
        # Select agents
        scores = [estimator(slot, i) for i in range(50)]
        top_k = topk(scores, k=3)
        
        # Process slot
        slot_embeddings = [agents[i](slot) for i in top_k]
        embeddings.append(concat(slot_embeddings))
    
    # Aggregate
    final_embedding = concat(embeddings)
    
    # Decision tree or classifier
    prediction = tree.predict(final_embedding)
    
    # Supervised loss
    loss = cross_entropy(prediction, label)
    loss.backward()
    optimizer.step()
```

### Key Considerations

1. **Hidden Label Dimension**
   - DINOv2 uses 1024-1536 (ViT embedding)
   - Your agents output configurable `hidden_dim` (e.g. 128-512)
   - Decision tree input: `num_slots × k × hidden_dim`

2. **Training Phases**
   - Phase 1: Pretrain agents with DINO-style SSL (no labels)
   - Phase 2: Fine-tune with supervised + continual learning

3. **Continual Learning Adaptations**
   - Freeze some agents, fine-tune others
   - Use Hoeffding Tree for incremental label learning
   - Replay buffer for old tasks

---

## 12. REFERENCES

- Official Code: https://github.com/facebookresearch/dinov2
- Paper: "DINOv2: Learning Robust Visual Features without Supervision"
- Key Files:
  - `dinov2/train/ssl_meta_arch.py` - Main training logic
  - `dinov2/loss/dino_clstoken_loss.py` - DINO loss
  - `dinov2/configs/ssl_default_config.yaml` - Hyperparameters
  - `dinov2/data/augmentations.py` - Data augmentation

---

## Summary Checklist ✅

Before implementing, make sure you understand:

- [ ] Student-Teacher architecture with EMA updates
- [ ] Multi-crop strategy (2 global + 8 local)
- [ ] DINO loss with centering mechanism
- [ ] Temperature schedules (warmup + fixed)
- [ ] Large prototype count (4K-65K)
- [ ] Strong data augmentations (color, blur, solarize)
- [ ] Training hyperparameters (LR scaling, weight decay schedule)
- [ ] Mixed precision training (fp16)
- [ ] iBOT masking (optional but recommended)
- [ ] KoLeo regularization (optional)

**Good luck with your implementation! 🚀**


