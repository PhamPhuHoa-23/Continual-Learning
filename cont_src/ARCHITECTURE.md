# Continual Learning Framework Architecture

## 📐 cont-src/ Architecture Overview

The `cont-src/` framework is designed with **modularity**, **config-driven development**, and **easy extensibility** as core principles.

---

## 🏗️ Core Design Patterns

### 1. Registry Pattern

All components are registered and instantiated through a central registry:

```python
from cont_src.core.registry import AGENT_REGISTRY

@AGENT_REGISTRY.register("my_agent")
class MyAgent(BaseAgent):
    pass

# Later, instantiate from config
agent = AGENT_REGISTRY.build("my_agent", hidden_dim=256)
```

**Benefits:**
- No hardcoding of component types
- Easy to add new components without modifying existing code
- Config can specify any registered component by name

### 2. Config-Driven Architecture

Everything is configured through structured dataclasses:

```python
@dataclass
class AgentConfig:
    type: str = "mlp"
    hidden_dim: int = 256
    # ...

config = Config.from_yaml("configs/experiment.yaml")
agent = AGENT_REGISTRY.build(config.agent.type, **vars(config.agent))
```

**Benefits:**
- Change model architecture without code changes
- Easy experiment tracking (save config with checkpoints)
- Type-safe configurations with validation

### 3. Base Classes

All components inherit from base classes that define common interface:

```python
class BaseAgent(BaseModule):
    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        pass
```

**Benefits:**
- Consistent API across all components
- Easy to swap implementations
- Built-in utilities (freeze, count_parameters, etc.)

---

## 📦 Component Structure

### Core Components (`core/`)

**registry.py** - Component registration system
- `Registry` class for managing component classes
- Global registries: MODEL, AGENT, ROUTER, LOSS, etc.
- `register()` decorator for easy registration

**base_module.py** - Base classes for all components
- `BaseModule` - Common interface for all PyTorch modules
- `BaseAgent` - Agent network interface
- `BaseRouter` - Routing mechanism interface
- `BaseAggregator` - Aggregation interface
- `BaseClassifier` - Classification interface

**trainer.py** - Training logic (TODO)
- `BaseTrainer` - Training loop abstraction
- `Task1Trainer` - Warm-up phase training
- `IncrementalTrainer` - Incremental task training

### Configuration (`config/`)

**base.py** - Configuration dataclasses
- `Config` - Main config container
- Component-specific configs (AgentConfig, LossConfig, etc.)
- Methods: `from_yaml()`, `save_yaml()`, `to_dict()`

**defaults.py** - Preset configurations
- `get_config("cifar100")` - Get CIFAR-100 defaults
- `get_config("tiny_imagenet")` - Get Tiny-ImageNet defaults
- Easy starting points for experiments

**schema.py** - Configuration validation
- `validate_config()` - Check config consistency
- `auto_fix_config()` - Auto-fix common issues (e.g., dimension matching)

### Models (`models/`)

**agents/** - Agent networks (s_k → h_k)
- `MLPAgent` - Multi-layer perceptron agent
- `MLPAgentWithDecoder` - MLP with decoder for L_agent
- `IdentityAgent` - Pass-through for debugging

**routers/** - Routing mechanisms
- `VAERouter` - VAE-based routing with Mahalanobis distance
  - Frozen network, updated statistics
  - Welford's algorithm for incremental stats

**aggregators/** - Aggregation mechanisms
- `AttentionAggregator` - Block-diagonal attention
  - Per-agent attention keys
  - Constant output dimension
- `AverageAggregator` - Simple average pooling

**classifiers/** - Classification heads (TODO)
- `SLDAClassifier` - Streaming Linear Discriminant Analysis
- `LinearClassifier` - Standard linear classifier

### Losses (`losses/`)

**losses.py** - All loss functions
- `PrimitiveLoss` - Matrix-level KL divergence (L_p)
- `SupervisedContrastiveLoss` - Pair-level metric learning (L_SupCon)
- `AgentReconstructionLoss` - Anti-collapse loss (L_agent)
- `ReconstructionLoss` - Slot attention reconstruction (L_recon)
- `LocalGeometryLoss` - Neighborhood preservation (L_local)
- `CompositeLoss` - Combines multiple losses based on config

### Data (`data/`) (TODO)

**base_dataset.py** - Base dataset interface
**cifar100.py** - CIFAR-100 continual learning splits
**tiny_imagenet.py** - Tiny-ImageNet splits

### Training (`training/`) (TODO)

**task1_trainer.py** - Task 1 warm-up training
**incremental_trainer.py** - Incremental task training
**callbacks.py** - Training callbacks (logging, checkpointing, etc.)

---

## 🔄 Data Flow

### Task 1 (Warm-up)

```
Images → Backbone (ViT) → Slot Attention → Slots
                                          ↓
                                    [Cluster Slots]
                                          ↓
Slots → Agent (MLP) → Hidden → Aggregator → H → SLDA → Predictions
        ↓
        VAE Router (train) → Freeze network, init stats
        ↓
Loss: L_p + L_SupCon + L_agent → Update agents → Freeze
```

### Task t > 1 (Incremental)

```
Images → Backbone (frozen) → Slot Attention (frozen) → Slots
                                                       ↓
                        Route via VAE (Mahalanobis distance)
                                ↓                      ↓
                        Existing Agent          Novel Slots
                        (frozen)                      ↓
                             ↓                   [Cluster]
                        Update stats                  ↓
                                              Spawn New Agents
                                                     ↓
                        Hidden States → Aggregator (add new keys)
                                           ↓
                                          H → SLDA (update new classes only)
```

---

## 🎛️ Configuration System

### Hierarchical Config Structure

```yaml
experiment_name: "my_experiment"

agent:
  type: "mlp"
  hidden_dim: 256
  
router:
  type: "vae"
  latent_dim: 32

loss:
  use_primitive: true
  use_supcon: true
  weight_primitive: 10.0

data:
  dataset: "cifar100"
  batch_size: 64

training:
  task1_lr: 0.0003
  incremental_lr: 0.0001
```

### Config → Component Mapping

```python
# Config specifies component type
config.agent.type = "mlp"

# Registry builds component
agent = AGENT_REGISTRY.build(
    config.agent.type,              # "mlp"
    input_dim=config.agent.input_dim,
    hidden_dim=config.agent.hidden_dim,
    # ...
)
```

---

## 🔌 Extension Points

### Adding a New Agent

```python
from cont_src.core import BaseAgent, AGENT_REGISTRY

@AGENT_REGISTRY.register("transformer_agent")
class TransformerAgent(BaseAgent):
    def __init__(self, input_dim, hidden_dim, n_heads=4, **kwargs):
        super().__init__(config={"input_dim": input_dim, ...})
        self.transformer = nn.TransformerEncoder(...)
    
    def forward(self, slots):
        return self.transformer(slots)
```

### Adding a New Loss

```python
from cont_src.losses import BaseLoss
from cont_src.core.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("my_loss")
class MyCustomLoss(BaseLoss):
    def forward(self, predictions, targets):
        loss = ...
        return self.weight * loss
```

### Adding a New Dataset

```python
from cont_src.core.registry import DATASET_REGISTRY
from torch.utils.data import Dataset

@DATASET_REGISTRY.register("my_dataset")
class MyDataset(Dataset):
    def __init__(self, root, train=True, **kwargs):
        # Load data
        pass
    
    def __getitem__(self, idx):
        return image, label
```

---

## 📊 Component Interaction Diagram

```
┌─────────────────────────────────────────────────┐
│                   Config                        │
│  (YAML → Python Dataclasses)                    │
└────────────┬────────────────────────────────────┘
             │
             ├──→ Registry → Build Components
             │
┌────────────┴────────────────────────────────────┐
│               Training Pipeline                 │
│                                                 │
│  ┌──────────┐     ┌──────────┐     ┌────────┐ │
│  │ Backbone │ ──→ │  Slots   │ ──→ │ Router │ │
│  └──────────┘     └──────────┘     └────┬───┘ │
│                         │                 │     │
│                    ┌────▼─────┐      [Existing│
│                    │  Agent   │       / Novel] │
│                    └────┬─────┘           │    │
│                         │            ┌────▼───┐│
│                    ┌────▼──────┐     │  New   ││
│                    │Aggregator │◄────┤ Agent  ││
│                    └────┬──────┘     └────────┘│
│                         │                       │
│                    ┌────▼──────┐               │
│                    │   SLDA    │               │
│                    └───────────┘               │
│                                                 │
│  Loss: CompositeLoss(L_p + L_SupCon + ...)    │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Design Benefits

### 1. Modularity
- Each component is independent
- Easy to test components in isolation
- Swap implementations without affecting others

### 2. Extensibility
- Add new components by registering them
- No need to modify existing code
- Follows Open/Closed Principle

### 3. Config-Driven
- Experiment entirely through config changes
- No code recompilation needed
- Easy to track experiments

### 4. Reproducibility
- Configs saved with checkpoints
- All hyperparameters tracked
- Seed control at every level

### 5. Scalability
- Clean separation of concerns
- Easy to parallelize components
- Supports complex pipelines

---

## 📚 Key Principles

1. **Single Responsibility**: Each module does one thing well
2. **Dependency Injection**: Components receive dependencies, don't create them
3. **Interface Segregation**: Base classes define minimal required interface
4. **Open/Closed**: Open for extension, closed for modification
5. **Don't Repeat Yourself**: Common functionality in base classes

---

## 🔍 Comparison: Old vs New Structure

| Aspect | Old `src/` | New `cont-src/` |
|--------|-----------|----------------|
| **Component Discovery** | Hardcoded imports | Registry pattern |
| **Configuration** | Scattered params | Centralized config |
| **Extensibility** | Modify existing code | Register new components |
| **Swapping Components** | Code changes | Config changes |
| **Dimension Consistency** | Manual tracking | Auto-validation |
| **Loss Composition** | Hardcoded sum | Config-driven composite |
| **Training Pipeline** | Monolithic script | Modular trainers |

---

## 🚀 Future Enhancements

1. **Automatic Hyperparameter Tuning** - Ray Tune integration
2. **Distributed Training** - DDP support
3. **Experiment Tracking** - W&B/MLflow integration
4. **Model Zoo** - Pre-trained component library
5. **AutoML** - Neural architecture search for agents
6. **Visualization** - Interactive component graphs

---

**Built for research flexibility and production scalability! 🎓**
