# Continual Learning Framework (cont-src/)

A modular, config-driven framework for continual learning research with compositional sub-concept routing.

## 🎯 Key Features

✅ **Config-Driven**: Change any component via YAML config - no code changes needed  
✅ **Registry Pattern**: Easy plugin system for datasets, models, losses  
✅ **Modular Design**: Swap backbones, agents, routers, aggregators independently  
✅ **Scalable**: Clean separation of concerns, extensible architecture  
✅ **Reproducible**: Comprehensive config tracking and checkpointing

## 📁 Structure

```
cont-src/
├── config/          # Configuration system
│   ├── base.py     # Config dataclasses
│   ├── defaults.py # Preset configurations
│   └── schema.py   # Validation
├── core/           # Core framework
│   ├── registry.py # Component registration
│   ├── base_module.py # Base classes
│   └── trainer.py  # Training logic
├── models/         # Model components
│   ├── agents/     # Agent networks (MLP, etc.)
│   ├── routers/    # Routing mechanisms (VAE, etc.)
│   ├── aggregators/ # Aggregation (attention, average)
│   └── classifiers/ # Classification heads (SLDA, etc.)
├── losses/         # Loss functions
│   └── losses.py   # Primitive, SupCon, etc.
├── data/           # Dataset loaders
└── training/       # Training pipelines
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from cont_src.config import get_config, Config
from cont_src.core.registry import AGENT_REGISTRY, LOSS_REGISTRY

# Load default config
config = get_config("cifar100")

# Or create custom config
config = Config()
config.agent.type = "mlp"
config.agent.hidden_dim = 512
config.loss.use_primitive = True

# Build components from config
agent = AGENT_REGISTRY.build(
    config.agent.type,
    input_dim=config.agent.input_dim,
    hidden_dim=config.agent.hidden_dim,
    output_dim=config.agent.output_dim
)
```

### 2. Using YAML Config

```yaml
# configs/my_experiment.yaml
experiment_name: "my_cifar100_experiment"

agent:
  type: "mlp"
  input_dim: 64
  hidden_dim: 256
  output_dim: 256
  num_layers: 3
  activation: "relu"

loss:
  use_primitive: true
  use_supcon: true
  weight_primitive: 10.0
  primitive_temperature: 10.0

data:
  dataset: "cifar100"
  n_tasks: 10
  batch_size: 64
```

```python
from cont_src.config import Config

# Load from YAML
config = Config.from_yaml("configs/my_experiment.yaml")

# Validate
from cont_src.config import validate_config
validate_config(config)
```

### 3. Register Custom Components

```python
from cont_src.core import AGENT_REGISTRY, BaseAgent
import torch.nn as nn

@AGENT_REGISTRY.register("my_custom_agent")
class MyCustomAgent(BaseAgent):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.net = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, slots):
        return self.net(slots)

# Now use in config:
# agent:
#   type: "my_custom_agent"
```

### 4. Composite Loss

```python
from cont_src.losses import CompositeLoss

# Create composite loss from config
loss_fn = CompositeLoss(config.loss.to_dict())

# Compute losses
outputs = loss_fn(
    hidden=hidden_states,
    labels=labels,
    slots=slots,
    # ... other inputs
)

print(outputs["total"])  # Total loss
print(outputs["primitive"])  # Individual losses
print(outputs["supcon"])
```

## 📝 Configuration Reference

### Agent Configuration

```python
agent:
  type: str                # "mlp", "transformer", "identity"
  input_dim: int          # Slot dimension
  hidden_dim: int         # Internal hidden dim
  output_dim: int         # Output dimension (d_h)
  num_layers: int         # Number of layers
  activation: str         # "relu", "gelu", "leaky_relu"
  dropout: float          # Dropout probability
  use_layer_norm: bool    # Use layer normalization
```

### Router Configuration

```python
router:
  type: str               # "vae", "cosine"
  latent_dim: int        # VAE latent dimension
  hidden_dims: list      # VAE hidden layers
  freeze_network: bool   # Freeze VAE network
  update_stats: bool     # Update latent statistics
  threshold_match: float # Matching threshold
  threshold_novel: float # Novelty threshold
```

### Loss Configuration

```python
loss:
  # Enable/disable losses
  use_primitive: bool
  use_supcon: bool
  use_agent_recon: bool
  use_reconstruction: bool
  use_local_geometry: bool
  
  # Loss weights
  weight_primitive: float
  weight_supcon: float
  weight_agent_recon: float
  
  # Loss parameters
  primitive_temperature: float
  supcon_temperature: float
  local_k_neighbors: int
```

## 🔧 Extending the Framework

### Add New Dataset

```python
from cont_src.core.registry import DATASET_REGISTRY
from torch.utils.data import Dataset

@DATASET_REGISTRY.register("my_dataset")
class MyDataset(Dataset):
    def __init__(self, root, train=True, **kwargs):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        return image, label
    
    def __len__(self):
        return len(self.data)
```

### Add New Loss

```python
from cont_src.losses import BaseLoss
from cont_src.core.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("my_loss")
class MyCustomLoss(BaseLoss):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight)
        # Initialize your loss
    
    def forward(self, predictions, targets):
        # Compute loss
        loss = ...
        return self.weight * loss
```

### Add New Aggregator

```python
from cont_src.core import BaseAggregator, AGGREGATOR_REGISTRY

@AGGREGATOR_REGISTRY.register("my_aggregator")
class MyAggregator(BaseAggregator):
    def forward(self, hidden_states, agent_assignments):
        # Aggregate hidden states
        aggregated = ...
        return {"aggregated": aggregated}
```

## 📊 Example: Complete Training Pipeline

```python
from cont_src.config import get_config
from cont_src.core.registry import AGENT_REGISTRY, ROUTER_REGISTRY, AGGREGATOR_REGISTRY
from cont_src.losses import CompositeLoss
import torch

# 1. Load config
config = get_config("cifar100")

# 2. Build models
agent = AGENT_REGISTRY.build(config.agent.type, **config.agent.to_dict())
router = ROUTER_REGISTRY.build(config.router.type, **config.router.to_dict())
aggregator = AGGREGATOR_REGISTRY.build(config.aggregator.type, **config.aggregator.to_dict())

# 3. Create loss
loss_fn = CompositeLoss(config.loss.to_dict())

# 4. Training loop
optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.task1_lr)

for images, labels in train_loader:
    # Get slots from slot attention
    slots = slot_attention(images)  # (B, K, D_slot)
    
    # Agent processing
    hidden = agent(slots)  # (B, K, D_h)
    
    # Aggregate
    agent_ids = torch.zeros(B, K, dtype=torch.long)  # Dummy assignments
    aggregated = aggregator(hidden, agent_ids)["aggregated"]  # (B, D_h)
    
    # Compute loss
    losses = loss_fn(hidden=aggregated, labels=labels)
    
    # Backward
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    
    print(f"Loss: {losses['total'].item():.4f}")
```

## 🎓 Training Pipeline (from Paper)

The framework implements the training pipeline from the paper:

**Task 1 (Warm-up):**
1. Train Slot Attention with L_recon → Freeze
2. Cluster slots with K-Means → Spawn M₀ agents
3. Train agents with L_p + L_SupCon + L_agent → Freeze agents
4. Train VAE routers → Freeze networks, init stats
5. Update SLDA classifier

**Task t > 1 (Incremental):**
1. Route slots using VAE (Mahalanobis distance)
2. Update VAE statistics (network frozen)
3. Collect novel slots → Cluster → Spawn new agents
4. Train new agents only → Freeze
5. Update SLDA with new classes only (old stats unchanged)

## 📖 Documentation

- [Configuration Guide](config/README.md) - Detailed config options
- [Loss Functions](losses/README.md) - Loss function documentation
- [Model Components](models/README.md) - Component specifications
- [Training Guide](training/README.md) - Training pipeline details

## 🔍 Registry System

All components are registered and can be listed:

```python
from cont_src.core.registry import REGISTRIES

# List all registered components
for registry_name, registry in REGISTRIES.items():
    print(f"{registry_name}: {registry.list()}")

# Output:
# agent: ['mlp', 'mlp_with_decoder', 'identity']
# router: ['vae']
# aggregator: ['attention', 'average']
# loss: ['primitive', 'supcon', 'agent_reconstruction', ...]
```

## 🛠️ Best Practices

1. **Always validate configs**: Use `validate_config()` before training
2. **Save configs with checkpoints**: Use `config.save_yaml()` to track experiments
3. **Use registry for all components**: Makes switching components trivial
4. **Follow naming conventions**: Use snake_case for registration names
5. **Document custom components**: Add docstrings with Args/Returns

## 🎯 Design Principles

1. **Config-Driven**: Change behavior through config, not code
2. **Separation of Concerns**: Each module has single responsibility
3. **Open/Closed**: Open for extension, closed for modification
4. **Dependency Injection**: Components receive dependencies, don't create them
5. **Reproducibility**: Track all hyperparameters and random seeds

## 📚 References

- Training pipeline: `paper/prototype_training-pipeline.tex`
- CompSLOT paper: "Plug-and-play compositionality for CL"
- Slot Attention: Locatello et al. (NeurIPS 2020)
- SupCon: Khosla et al. (NeurIPS 2020)
- Local Geometry: Gao et al. (CVPR 2023)

## 🤝 Contributing

Follow [RULE.md](../RULE.md) for code quality standards:
- No dummy code
- Research before implementing
- Document everything
- Test thoroughly
- Report if you can't implement something

---

**Built with modularity and extensibility in mind. Happy researching! 🚀**
