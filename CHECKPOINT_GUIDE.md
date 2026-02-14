# Checkpoint Guide 💾

Complete guide to loading and saving checkpoints in the Slot Multi-Agent System.

---

## Overview 📋

The system supports checkpoints for:
1. **Slot Attention** - Pretrained AdaSlot or custom weights
2. **Agents** - After Phase 1 DINO training
3. **Estimators** - Performance estimators
4. **Full System** - All components together

---

## Config Options ⚙️

### 1. Slot Attention Checkpoint

```yaml
slot_attention:
  pretrained:
    enabled: false               # Enable pretrained weights
    checkpoint_path: null        # Path to checkpoint
    strict_load: false           # Ignore missing keys
    freeze: false                # Freeze weights during training
```

**Example:**

```yaml
slot_attention:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/slot_attention/CLEVR10.ckpt"
    strict_load: false  # AdaSlot checkpoint may have shape differences
    freeze: false       # Fine-tune on your dataset
```

---

### 2. Agent Checkpoint (Phase 1 → Phase 2)

```yaml
agents:
  pretrained:
    enabled: false               # Load pretrained agents
    checkpoint_path: null        # Path to agent checkpoint
    load_teacher: true           # Also load teacher weights
    strict_load: false           # Ignore missing keys
    freeze: false                # Freeze agents (skip Phase 1)
```

**Example:**

```yaml
agents:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/agents/agents_phase1_epoch10.pth"
    load_teacher: true
    freeze: true  # Freeze agents, only train tree

training:
  phase1_agents:
    enabled: false  # Skip Phase 1 since agents are pretrained
```

---

## Available Checkpoints 📦

### 1. AdaSlot CLEVR10 (Pretrained)

- **Source:** [AdaSlot GitHub](https://github.com/amazon-science/AdaSlot)
- **Dataset:** CLEVR (synthetic objects)
- **Download:** Google Drive link in `Setup/adaslot/`
- **Path:** `./checkpoints/slot_attention/CLEVR10.ckpt`
- **Size:** ~50MB
- **Format:** PyTorch Lightning (.ckpt)

**Specifications:**
```python
{
  'num_slots': adaptive (3-10),
  'slot_dim': 64,
  'input_size': 128x128,
  'trained_on': 'CLEVR dataset'
}
```

**How to use:**
```yaml
# config_variants/07_pretrained_slot_attention.yaml
slot_attention:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/slot_attention/CLEVR10.ckpt"
```

---

### 2. Your Custom Checkpoints

After training Phase 1, you'll have:

```
checkpoints/
├── agents/
│   ├── agents_phase1_epoch05.pth
│   ├── agents_phase1_epoch10.pth  ← Use this for Phase 2
│   └── agents_phase1_best.pth
├── estimators/
│   └── estimators_phase1_epoch10.pth
└── slot_attention/
    └── slot_attention_epoch10.pth (if trained from scratch)
```

---

## Usage Examples 🚀

### Example 1: Train from Scratch (No Checkpoints)

```yaml
# config.yaml
slot_attention:
  pretrained:
    enabled: false  # Train from scratch

agents:
  pretrained:
    enabled: false  # Train from scratch
```

**Training:**
```bash
# Phase 1: Train agents (DINO SSL)
python train_phase1.py --config config.yaml

# Phase 2: Train tree (incremental)
python train_phase2.py --config config.yaml
```

---

### Example 2: Use Pretrained Slot Attention

```yaml
# config_variants/07_pretrained_slot_attention.yaml
slot_attention:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/slot_attention/CLEVR10.ckpt"
    freeze: false  # Fine-tune
```

**Training:**
```bash
# Phase 1: Train agents (Slot Attention loaded from checkpoint)
python train_phase1.py --config config_variants/07_pretrained_slot_attention.yaml

# Phase 2: Train tree
python train_phase2.py --config config_variants/07_pretrained_slot_attention.yaml
```

**Benefits:**
- ✅ Better slot decomposition (pretrained on CLEVR)
- ✅ Faster convergence (good initialization)
- ✅ Can freeze to save compute

---

### Example 3: Skip Phase 1 (Load Pretrained Agents)

After training Phase 1 once, you can skip it:

```yaml
# config_variants/08_pretrained_agents_phase2.yaml
agents:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/agents/agents_phase1_epoch10.pth"
    freeze: true

training:
  phase1_agents:
    enabled: false  # Skip!
  phase2_tree:
    enabled: true
```

**Training:**
```bash
# Step 1: Train Phase 1 ONCE
python train_phase1.py --config config.yaml
# Saves to: ./checkpoints/agents/agents_phase1_epoch10.pth

# Step 2: Experiment with different Phase 2 configs
python train_phase2.py --config config_variants/08_pretrained_agents_phase2.yaml

# Try different selection strategies without retraining agents:
python train_phase2.py --config config_variants/02_ucb_bandit_exploration.yaml \
    --override agents.pretrained.enabled=true \
    --override agents.pretrained.checkpoint_path=./checkpoints/agents/agents_phase1_epoch10.pth
```

**Benefits:**
- ✅ Experiment with Phase 2 (tree, bandit) without retraining agents
- ✅ Faster iteration
- ✅ Save compute

---

## Programmatic Usage 🐍

### Load Checkpoints

```python
from src.utils.checkpoint import (
    load_slot_attention_checkpoint,
    load_agent_checkpoint,
    load_estimator_checkpoint
)
from src.models.slot_attention import SlotAttentionAutoEncoder
from src.slot_multi_agent import create_agent_pool, create_estimator_pool

# 1. Load Slot Attention
slot_model = SlotAttentionAutoEncoder(num_slots=7, slot_dim=64)
slot_model = load_slot_attention_checkpoint(
    model=slot_model,
    checkpoint_path='./checkpoints/CLEVR10.ckpt',
    strict=False,
    device='cuda'
)
# Output:
# Loading Slot Attention checkpoint from: ./checkpoints/CLEVR10.ckpt
# ✓ Checkpoint loaded successfully
# ⚠ Missing keys in checkpoint: 15
# ✓ Weights loaded successfully (strict=False)

# 2. Load agents
students, teachers = create_agent_pool(50, 64, 256)
students, teachers = load_agent_checkpoint(
    student_agents=students,
    teacher_agents=teachers,
    checkpoint_path='./checkpoints/agents_phase1_epoch10.pth',
    strict=False,
    device='cuda'
)
# Output:
# Loading agent checkpoint from: ./checkpoints/agents_phase1_epoch10.pth
# ✓ Loaded 50 student agents
# ✓ Loaded 50 teacher agents
# ✓ DINO loss centers available (50 agents)

# 3. Load estimators
estimators = create_estimator_pool(50, 'vae', 64)
estimators = load_estimator_checkpoint(
    estimators=estimators,
    checkpoint_path='./checkpoints/estimators_phase1.pth',
    strict=False,
    device='cuda'
)
```

---

### Save Checkpoints

```python
from src.utils.checkpoint import (
    save_agent_checkpoint,
    save_estimator_checkpoint,
    save_full_checkpoint
)

# 1. Save agents (after Phase 1 training)
save_agent_checkpoint(
    student_agents=students,
    teacher_agents=teachers,
    dino_losses=dino_losses,  # List of DINOLoss objects
    save_path='./checkpoints/agents/agents_phase1_epoch10.pth',
    epoch=10,
    metadata={'dataset': 'cifar100', 'num_prototypes': 256}
)
# Output:
# ✓ Agent checkpoint saved to: ./checkpoints/agents/agents_phase1_epoch10.pth

# 2. Save estimators
save_estimator_checkpoint(
    estimators=estimators,
    save_path='./checkpoints/estimators/estimators_phase1.pth',
    metadata={'type': 'vae', 'latent_dim': 16}
)

# 3. Save full system
save_full_checkpoint(
    slot_attention=slot_model,
    student_agents=students,
    teacher_agents=teachers,
    estimators=estimators,
    dino_losses=dino_losses,
    save_path='./checkpoints/full/full_system_epoch10.pth',
    epoch=10,
    metadata={'experiment': 'baseline'}
)
```

---

### List Available Checkpoints

```python
from src.utils.checkpoint import list_checkpoints, get_latest_checkpoint

# List all agent checkpoints
checkpoints = list_checkpoints('./checkpoints/agents/', pattern='*.pth')
for ckpt in checkpoints:
    print(f"  - {ckpt}")

# Get latest checkpoint
latest = get_latest_checkpoint('./checkpoints/agents/', pattern='agents_phase1_*.pth')
print(f"Latest: {latest}")
```

---

## Checkpoint Format 📄

### Agent Checkpoint Structure

```python
{
    'student_agents': state_dict,        # Student agent weights
    'teacher_agents': state_dict,        # Teacher agent weights (EMA)
    'dino_loss_centers': [tensor, ...],  # DINO loss centers (50 agents)
    'epoch': int,                        # Training epoch
    'metadata': {                        # Custom metadata
        'dataset': 'cifar100',
        'num_prototypes': 256,
        'learning_rate': 1e-3,
        ...
    }
}
```

### Slot Attention Checkpoint Structure

```python
{
    'model_state_dict': state_dict,      # Model weights
    'metadata': {                        # Custom metadata
        'num_slots': 7,
        'slot_dim': 64,
        ...
    }
}
```

Or PyTorch Lightning format:
```python
{
    'state_dict': state_dict,            # Model weights (with 'model.' prefix)
    'hyper_parameters': {...},           # Hyperparameters
    'epoch': int,
    'global_step': int,
    ...
}
```

---

## Troubleshooting 🔧

### Problem 1: Checkpoint Not Found

```
FileNotFoundError: Checkpoint not found: ./checkpoints/CLEVR10.ckpt
```

**Solution:**
```bash
# Create checkpoint directory
mkdir -p checkpoints/slot_attention

# Download AdaSlot checkpoint
# See: Setup/adaslot/README.md for download link
```

---

### Problem 2: Shape Mismatch

```
RuntimeError: size mismatch for encoder.conv1.weight: 
copying a param with shape torch.Size([64, 3, 5, 5]) 
from checkpoint, the shape in current model is torch.Size([64, 3, 3, 3])
```

**Solution:**
```yaml
# Use strict_load=false to ignore mismatches
slot_attention:
  pretrained:
    strict_load: false  # ← This!
```

Or adjust model architecture to match checkpoint:
```yaml
slot_attention:
  encoder_kernel_size: 5  # Match checkpoint
```

---

### Problem 3: Missing Keys

```
⚠ Missing keys in checkpoint: 127
```

**This is usually OK!** Means:
- Checkpoint was trained on different architecture
- Some layers don't exist in checkpoint
- `strict_load=false` will load what matches

**Solution:**
```yaml
# Already handled with strict_load=false
slot_attention:
  pretrained:
    strict_load: false
```

---

### Problem 4: Out of Memory with Large Checkpoints

```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Load on CPU first
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

# Then move to GPU after loading
model = model.to('cuda')
```

Or in config:
```yaml
device:
  type: "cpu"  # Load first, then switch to cuda manually
```

---

## Best Practices ✅

### 1. Always Use `strict_load=false` for Pretrained Models

```yaml
pretrained:
  strict_load: false  # Ignore shape/key mismatches
```

**Why?** Pretrained models often have slight architecture differences.

---

### 2. Save Checkpoints Regularly

```python
# Save every 5 epochs
if epoch % 5 == 0:
    save_agent_checkpoint(...)
```

Or in config:
```yaml
checkpointing:
  save_every_n_epochs: 5
```

---

### 3. Keep Metadata

```python
save_agent_checkpoint(
    ...,
    metadata={
        'dataset': 'cifar100',
        'num_prototypes': 256,
        'learning_rate': 1e-3,
        'num_agents': 50,
        'experiment_name': 'baseline'
    }
)
```

**Why?** Helps track experiments!

---

### 4. Freeze Pretrained Weights (If Appropriate)

```yaml
slot_attention:
  pretrained:
    freeze: true  # Don't train Slot Attention
```

**When to freeze:**
- ✅ Pretrained model is good enough
- ✅ Want to save compute
- ✅ Only care about agents/tree

**When NOT to freeze:**
- ❌ Pretrained on very different domain (CLEVR vs real images)
- ❌ Want to fine-tune for your data

---

### 5. Use Latest Checkpoint Utility

```python
from src.utils.checkpoint import get_latest_checkpoint

latest = get_latest_checkpoint('./checkpoints/agents/')
if latest:
    students, teachers = load_agent_checkpoint(students, teachers, latest)
```

---

## Checkpoint Directory Structure 📂

Recommended organization:

```
checkpoints/
├── slot_attention/
│   ├── CLEVR10.ckpt                    # Pretrained AdaSlot
│   ├── slot_attention_scratch_epoch10.pth
│   └── slot_attention_finetuned.pth
│
├── agents/
│   ├── baseline/
│   │   ├── agents_phase1_epoch05.pth
│   │   ├── agents_phase1_epoch10.pth
│   │   └── agents_phase1_best.pth
│   │
│   ├── large_prototypes/
│   │   └── agents_512proto_epoch10.pth
│   │
│   └── ucb_bandit/
│       └── agents_ucb_epoch10.pth
│
├── estimators/
│   ├── vae_estimators.pth
│   └── mlp_estimators.pth
│
└── full/
    ├── full_system_baseline_epoch10.pth
    └── full_system_ucb_epoch10.pth
```

---

## Summary Checklist ✅

Before running training:

- [ ] Check if pretrained Slot Attention is available
- [ ] Decide: train from scratch or load checkpoint?
- [ ] Set `pretrained.enabled` in config
- [ ] Set checkpoint paths correctly
- [ ] Use `strict_load=false` for pretrained models
- [ ] Decide: freeze or fine-tune?
- [ ] Verify checkpoint exists before training

After Phase 1:

- [ ] Checkpoint saved to `./checkpoints/agents/`
- [ ] Verify checkpoint can be loaded
- [ ] Use checkpoint for Phase 2 experiments

---

## Next Steps 🚀

1. **Download AdaSlot checkpoint:**
   ```bash
   # See Setup/adaslot/ for instructions
   ```

2. **Try pretrained Slot Attention:**
   ```bash
   python train_phase1.py --config config_variants/07_pretrained_slot_attention.yaml
   ```

3. **Train Phase 1, save checkpoint:**
   ```bash
   python train_phase1.py --config config.yaml
   # Saves to: ./checkpoints/agents/agents_phase1_epoch10.pth
   ```

4. **Skip Phase 1, load checkpoint for Phase 2:**
   ```bash
   python train_phase2.py --config config_variants/08_pretrained_agents_phase2.yaml
   ```

---

**Questions?** See `CONFIG_GUIDE.md` for full configuration options!

**Last Updated:** 2026-02-13


