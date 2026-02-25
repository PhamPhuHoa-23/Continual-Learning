# AdaSlot Configuration Examples

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
