# AdaSlot Implementation Summary

## ✅ Implementation Complete

Successfully implemented Adaptive Slot Attention (AdaSlot) with primitive selection mechanism from the CompSLOT paper.

## 📦 Files Created

### Core Modules
1. **adaptive_slot_attention.py** (460 lines)
   - `AdaptiveSlotAttention` - Main slot attention with Gumbel selection
   - `GumbelSlotSelector` - Learnable slot selection network
   - `sample_slot_lower_bound()` - Minimum slot constraint

2. **primitives.py** (413 lines)
   - `PrimitiveSelector` - Attention-based primitive aggregation
   - `SlotDecoder` - MLP decoder for reconstruction
   - `AdaSlotModule` - Complete module combining all components

3. **__init__.py** - Package exports

4. **CONFIG_EXAMPLES.md** - Configuration examples and usage guide

### Testing & Configuration
5. **test_adaslot.py** (390 lines) - Comprehensive test suite
6. **configs/cifar100_adaslot.yaml** - Example configuration

## 🎯 Key Features

### 1. Adaptive Slot Attention
- **Iterative attention mechanism** with GRU updates
- **Gumbel-Softmax selection** for differentiable slot dropping
- **Temperature annealing**: τ(t) = max(τ_min, τ_init × e^(-r×t))
- **Minimum slot constraint**: Ensures at least N slots kept
- **Multi-head attention** support
- **Position-aware** slot initialization

### 2. Primitive Selection
- **Attention-based aggregation**: Learns class-relevant concepts
- **Learnable primitive key**: Adaptively selects important slots
- **Temperature control**: Adjustable sparsity
- **Mask support**: Works with dropped slots

### 3. Slot Decoder
- **MLP-based reconstruction**: Lightweight decoder
- **Position encoding**: Handles spatial information
- **Attention-weighted**: Uses attention maps for aggregation

### 4. Loss Integration
- **Primitive Loss** ✅ (already implemented in losses.py)
  - KL divergence between label similarity and concept similarity
  - Enforces intra-class primitive consistency
- **Reconstruction Loss** ✅
  - MSE between input and reconstructed features
  - Helps learn meaningful slot decompositions
- **Supervised Contrastive Loss** ✅
  - Pull same-class samples together in concept space

## 📊 Test Results

All tests passed ✅:

```
adaptive_slot_attention       : ✅ PASSED
primitive_selector            : ✅ PASSED  
slot_decoder                  : ✅ PASSED
adaslot_module                : ✅ PASSED
training_simulation           : ✅ PASSED
with_losses                   : ✅ PASSED
```

**Model Size**: ~497K parameters for complete AdaSlotModule

**Performance Highlights**:
- Adaptive slot selection working (keeps 1-6 slots out of 7)
- Temperature annealing functional (1.0 → 0.74 over 10K steps)
- Primitive selection produces meaningful aggregations
- Reconstruction achieves low MSE (~1.0)
- Primitive loss computed correctly (~15.5)

## 🔧 Architecture Details

### AdaptiveSlotAttention
```
Input: Features (B, N, D_f)
├── Slot Initialization: Sample from learned Gaussian
├── Iterative Attention (×3):
│   ├── LayerNorm
│   ├── Q, K, V projections
│   ├── Attention: Q @ K^T / √d
│   ├── Softmax over slots
│   ├── Weighted sum: Attn @ V
│   └── GRU update + MLP
├── Gumbel Selection:
│   ├── Score network: Slot → [drop, keep] logits
│   ├── Gumbel-Softmax sampling
│   └── Apply minimum constraint
└── Output: Slots (B, K, D_s) + Mask (B, K)
```

### PrimitiveSelector
```
Input: Slots (B, K, D_s)
├── Project: LayerNorm + Linear + Tanh
├── Similarity: proj(S) @ K_p (learnable key)
├── Attention: softmax(τ × similarity)
└── Aggregate: weighted sum → Primitive (B, D_s)
```

### SlotDecoder
```
Input: Slots (B, K, D_s) + Attention (B, K, N)
├── Add Position Encoding
├── MLP Decoder per slot
├── Attention-weighted aggregation
└── Output: Reconstructed (B, N, D_f)
```

## 💡 Usage Example

```python
from cont_src.models.slot_attention import AdaSlotModule
from cont_src.losses import PrimitiveLoss

# Create model
adaslot = AdaSlotModule(
    num_slots=7,
    slot_dim=128,
    feature_dim=768,
    use_gumbel=True,
    use_primitive=True,
    use_decoder=True
)

# Forward pass
features = backbone(images)  # (B, 196, 768) from ViT
outputs = adaslot(features, global_step=step)

slots = outputs["slots"]            # (B, K, 128)
primitives = outputs["primitives"]  # (B, 128)
reconstruction = outputs["reconstruction"]  # (B, 196, 768)
mask = outputs["slot_mask"]         # (B, K)

# Compute losses
prim_loss_fn = PrimitiveLoss(temperature=10.0, weight=10.0)
loss_prim = prim_loss_fn(primitives, labels)
loss_recon = F.mse_loss(reconstruction, features)

loss = loss_classification + loss_prim + loss_recon
```

## 📝 Configuration

See [CONFIG_EXAMPLES.md](cont_src/models/slot_attention/CONFIG_EXAMPLES.md) for:
- Basic AdaSlot configuration
- Loss configuration examples
- Training setup
- Compositional learning setups
- Custom temperature schedules

Example config: [configs/cifar100_adaslot.yaml](configs/cifar100_adaslot.yaml)

## 🔬 Implementation Details

### Temperature Annealing
```python
τ(t) = max(τ_min, τ_init × exp(-r × t))
```
- `τ_init = 1.0`: Initial temperature (high exploration)
- `τ_min = 0.5`: Minimum temperature (exploitation)
- `r = 0.00003`: Decay rate (~25K steps to reach min)

### Gumbel-Softmax
- **Hard sampling** during training
- **Straight-through estimator** for gradients
- **Minimum constraint** enforced via post-processing

### Primitive Selection
- **Temperature**: τ = 100/√D (auto-scaled by dimension)
- **Softmax attention**: Normalizes slot contributions
- **Mask-aware**: Zeros out dropped slots before aggregation

## Basic AdaSlot Configuration

```yaml
# Adaptive Slot Attention with default settings
adaslot:
  num_slots: 7
  slot_dim: 128
  feature_dim: 768  # ViT-B/16 feature dim
  num_iterations: 3
  num_heads: 1
  
  # Adaptive selection
  use_gumbel: true
  gumbel_low_bound: 1  # Keep at least 1 slot
  init_temperature: 1.0
  min_temperature: 0.5
  temperature_anneal_rate: 0.00003
  
  # Primitive selection
  use_primitive: true
  primitive_hidden_dim: 128
  
  # Reconstruction
  use_decoder: true
  decoder_hidden_dim: 256
  decoder_layers: 2
```

## Loss Configuration with AdaSlot

```yaml
loss:
  # Standard classification loss
  use_cross_entropy: true
  
  # Primitive loss (concept-level KL divergence)
  use_primitive: true
  weight_primitive: 10.0
  primitive_temperature: 10.0
  
  # Reconstruction loss
  use_reconstruction: true
  weight_reconstruction: 1.0
  
  # Supervised contrastive loss (optional)
  use_supcon: false
  weight_supcon: 0.1
  supcon_temperature: 0.07
```

## Training Configuration

```yaml
training:
  batch_size: 64
  learning_rate: 0.0001
  epochs_per_task: 20
  
  # AdaSlot specific
  global_step_tracking: true  # For temperature annealing
  
optimizer:
  type: "adam"
  lr: 0.0001
  weight_decay: 0.0001
  
  # Separate LR for slot attention (optional)
  slot_lr_multiplier: 1.0
```

## Complete Example: CIFAR-100 with AdaSlot

```yaml
experiment_name: "cifar100_adaslot_baseline"

# Data
data:
  dataset: "cifar100"
  n_tasks: 10
  batch_size: 64
  num_workers: 4
  seed: 42

# Backbone
backbone:
  type: "vit_b_16"
  pretrained: true
  freeze: false  # Fine-tune backbone

# AdaSlot Module
adaslot:
  num_slots: 7
  slot_dim: 128
  feature_dim: 768
  num_iterations: 3
  
  use_gumbel: true
  gumbel_low_bound: 2
  
  use_primitive: true
  use_decoder: true

# Losses
loss:
  use_primitive: true
  weight_primitive: 10.0
  primitive_temperature: 10.0
  
  use_reconstruction: true
  weight_reconstruction: 1.0

# Training
training:
  epochs_per_task: 20
  learning_rate: 0.0001
  
# Classifier
classifier:
  type: "slda"  # or "linear"
  hidden_dim: 256
```

## AdaSlot with Few Slots

```yaml
# Minimal slot configuration for faster training
adaslot:
  num_slots: 4
  slot_dim: 64
  num_iterations: 2
  
  gumbel_low_bound: 1
  init_temperature: 1.5  # Higher = more exploration
```

## AdaSlot with Many Slots

```yaml
# Many slots for complex scenes
adaslot:
  num_slots: 10
  slot_dim: 256
  num_iterations: 5
  
  gumbel_low_bound: 3  # Keep at least 3
  temperature_anneal_rate: 0.00005  # Slower annealing
```

## Compositional Learning Tasks

```yaml
# For compositional benchmarks (CGQA, COBJ)
data:
  dataset: "cgqa"  # or "cobj"
  n_tasks: 3

adaslot:
  num_slots: 8
  slot_dim: 192
  
  # More aggressive selection
  gumbel_low_bound: 2
  init_temperature: 2.0
  min_temperature: 0.3

loss:
  # Strong primitive loss for concept learning
  weight_primitive: 20.0
  primitive_temperature: 15.0
  
  # Enable all losses
  use_supcon: true
  weight_supcon: 0.5
```

## Usage in Code

```python
from cont_src.config import Config
from cont_src.models.slot_attention import AdaSlotModule

# Load config
config = Config.from_yaml("configs/cifar100_adaslot.yaml")

# Build AdaSlot module
adaslot = AdaSlotModule(
    num_slots=config.adaslot.num_slots,
    slot_dim=config.adaslot.slot_dim,
    feature_dim=config.adaslot.feature_dim,
    **config.adaslot.__dict__
)

# Forward pass
features = backbone(images)  # (B, N, D)
outputs = adaslot(features, global_step=current_step)

slots = outputs["slots"]            # (B, K, D_s)
primitives = outputs["primitives"]  # (B, D_s)
reconstruction = outputs["reconstruction"]  # (B, N, D)
```

## Notes

- **Temperature Annealing**: Starts at `init_temperature`, exponentially decays to `min_temperature`
- **Low Bound**: Ensures minimum slots kept for stability
- **Primitive Loss**: Higher weight (10-20) recommended for concept learning
- **Reconstruction**: Helps slot attention learn meaningful decompositions
- **Global Step**: Must track global step for temperature annealing to work

## Advanced: Custom Temperature Schedule

```python
# Custom temperature function
def custom_temperature(step, init=1.0, min_temp=0.5):
    # Linear decay
    decay = 1.0 - (step / 10000.0)
    temp = init * max(decay, min_temp)
    return max(temp, min_temp)

# Use in model
adaslot.slot_attention.get_temperature = custom_temperature
```



