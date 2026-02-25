# Data Module Usage Guide

## Overview

The `cont-src/data/` module provides continual learning datasets with **Avalanche framework integration**. All datasets follow a unified interface and can be swapped via configuration.

## Quick Start

### Using Registry (Recommended)

```python
from cont_src.core.registry import DATASET_REGISTRY

# Build from config
dataset = DATASET_REGISTRY.build(
    "cifar100",
    n_tasks=10,
    batch_size=64,
    seed=42
)

# Iterate over tasks
for task_id in range(len(dataset)):
    task = dataset[task_id]
    
    print(f"Task {task_id}:")
    print(f"  Classes: {task.classes}")
    print(f"  Total classes seen: {task.n_classes_total}")
    
    # Training
    # NOTE: Avalanche returns (images, labels, task_ids) - 3 values!
    for batch in task.train_loader:
        images, labels, task_ids = batch  # Unpack all 3
        # images: (B, C, H, W)
        # labels: (B,)
        # task_ids: (B,)
        pass
    
    # Testing (on all classes seen so far)
    for batch in task.test_loader:
        images, labels, task_ids = batch
        pass
```

**⚠️ Important:** Avalanche DataLoaders return 3 values: `(images, labels, task_ids)`, not just 2. Always unpack all 3 values or handle variable-length tuples.

### Direct Import

```python
from cont_src.data import CIFAR100ContinualDataset

dataset = CIFAR100ContinualDataset(
    n_tasks=10,
    batch_size=64
)
```

## Supported Datasets

### CIFAR-10

```python
dataset = DATASET_REGISTRY.build(
    "cifar10",
    n_tasks=5,  # Must divide 10 evenly: 2, 5, or 10
    batch_size=64,
    seed=42
)

# Details:
# - 10 classes
# - 32x32 RGB images
# - 50,000 training, 10,000 test
```

### CIFAR-100

```python
dataset = DATASET_REGISTRY.build(
    "cifar100",
    n_tasks=10,  # Must divide 100 evenly: 2, 5, 10, 20, 25, 50, 100
    batch_size=64,
    seed=42,
    shuffle=True,  # Randomize class order
    fixed_class_order=None  # Or provide custom order: [3, 7, 1, ...]
)

# Details:
# - 100 classes
# - 32x32 RGB images
# - 50,000 training, 10,000 test
# - n_tasks=10 → 10 classes per task
```

### MNIST

```python
dataset = DATASET_REGISTRY.build(
    "mnist",
    n_tasks=5,  # Must divide 10 evenly: 2, 5, or 10
    batch_size=128,
    seed=42
)

# Details:
# - 10 classes (digits 0-9)
# - 28x28 grayscale images
# - 60,000 training, 10,000 test
```

### Tiny-ImageNet

```python
dataset = DATASET_REGISTRY.build(
    "tiny_imagenet",
    n_tasks=20,  # Must divide 200 evenly: 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200
    batch_size=64,
    seed=42
)

# Details:
# - 200 classes from ImageNet
# - 64x64 RGB images
# - 100,000 training, 10,000 test
# - 500 training images per class
```

## Task Data Structure

Each task returns a `TaskData` object:

```python
@dataclass
class TaskData:
    task_id: int                # Task index (0-based)
    train_loader: DataLoader    # Training data for this task only
    test_loader: DataLoader     # Test data for ALL classes seen so far
    classes: List[int]          # Class indices in this task
    n_classes_total: int        # Total classes seen up to this task
```

### Example:

```python
dataset = DATASET_REGISTRY.build("cifar100", n_tasks=10)

# Task 0: classes [23, 45, 12, ...] (10 classes)
task_0 = dataset[0]
print(task_0.classes)          # [23, 45, 12, 67, 89, 5, 34, 78, 91, 2]
print(task_0.n_classes_total)  # 10

# Task 1: classes [88, 3, 56, ...] (10 more classes)
task_1 = dataset[1]
print(task_1.classes)          # [88, 3, 56, 14, 72, 41, 9, 63, 28, 95]
print(task_1.n_classes_total)  # 20

# Testing: task_1.test_loader tests on all 20 classes (task 0 + task 1)
```

## Custom Datasets

For datasets not in Avalanche, use the custom template:

```python
from cont_src.data.base_dataset import BaseContinualDataset
from cont_src.core.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("my_dataset")
class MyCustomDataset(BaseContinualDataset):
    def __init__(self, n_tasks, batch_size=64, **kwargs):
        super().__init__(n_tasks, batch_size, **kwargs)
        # Load your data
        self._load_data()
    
    def _load_data(self):
        # Implement data loading
        pass
    
    def get_task_data(self, task_id: int) -> TaskData:
        # Return TaskData for this task
        pass
    
    def get_task_classes(self, task_id: int) -> List[int]:
        # Return class indices for this task
        pass
```

See [custom_template.py](custom_template.py) for complete example.

## Configuration

Add to your YAML config:

```yaml
data:
  dataset: "cifar100"
  n_tasks: 10
  batch_size: 64
  num_workers: 4
  seed: 42
  data_root: "./data"
  
  # Dataset-specific options
  shuffle: true
  fixed_class_order: null
  return_task_id: false
```

Load in code:

```python
from cont_src.config import Config

config = Config.from_yaml("config.yaml")
dataset = DATASET_REGISTRY.build(
    config.data.dataset,
    **config.data.__dict__
)
```

## Avalanche Environment

The datasets require Avalanche to be installed:

```bash
# Activate conda environment with Avalanche
conda activate continual-learning

# Or install Avalanche
pip install avalanche-lib
```

## Dataset Registry

All datasets are registered automatically on import:

```python
from cont_src.core.registry import DATASET_REGISTRY

# List available datasets
print(DATASET_REGISTRY.list())
# ['cifar10', 'cifar100', 'mnist', 'tiny_imagenet', 'custom_example']

# Get dataset info
print(DATASET_REGISTRY.registry["cifar100"])
# <class 'cont_src.data.cifar.CIFAR100ContinualDataset'>
```

## Best Practices

1. **Use Registry**: Better for config-driven experiments
2. **Cache Tasks**: `get_task_data()` caches results automatically
3. **Test Incrementally**: `test_loader` includes all classes seen so far
4. **Consistent Seeds**: Use same seed for reproducibility
5. **Custom Datasets**: Follow the template for new datasets

## Examples

### Training Loop

```python
dataset = DATASET_REGISTRY.build("cifar100", n_tasks=10)

for task_id in range(len(dataset)):
    task = dataset[task_id]
    
    # Train on new classes
    model.train()
    for batch in task.train_loader:
        images, labels, task_ids = batch  # Avalanche returns 3 values
        
        # Training code
        loss = model(images, labels)
        loss.backward()
    
    # Test on all classes seen so far
    model.eval()
    correct = 0
    total = 0
    for batch in task.test_loader:
        images, labels, _ = batch  # Ignore task_ids
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum()
        total += len(labels)
    
    accuracy = correct / total
    print(f"Task {task_id} - Accuracy: {accuracy:.2%}")
```

### Multi-Dataset Experiment

```yaml
# configs/cifar100.yaml
data:
  dataset: "cifar100"
  n_tasks: 10
  batch_size: 64

# configs/tiny_imagenet.yaml
data:
  dataset: "tiny_imagenet"
  n_tasks: 20
  batch_size: 64
```

```python
# Run on both datasets
for config_path in ["configs/cifar100.yaml", "configs/tiny_imagenet.yaml"]:
    config = Config.from_yaml(config_path)
    dataset = DATASET_REGISTRY.build(config.data.dataset, **config.data.__dict__)
    train_and_evaluate(dataset)
```

## Future Additions

Planned datasets:
- CORe50 (continual object recognition)
- DomainNet (domain incremental)
- Custom video datasets
- Task-incremental variants

To add a new dataset, see [custom_template.py](custom_template.py).
