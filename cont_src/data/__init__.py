"""
Data loading and processing utilities for continual learning.

Provides continual learning datasets with Avalanche integration.
All datasets are accessible via the DATASET_REGISTRY.

Supported datasets:
- CIFAR-10: 10 classes, 32x32 RGB
- CIFAR-100: 100 classes, 32x32 RGB
- MNIST: 10 classes, 28x28 grayscale
- Tiny-ImageNet: 200 classes, 64x64 RGB

Example:
    >>> from cont_src.core.registry import DATASET_REGISTRY
    >>> 
    >>> # Build from registry
    >>> dataset = DATASET_REGISTRY.build(
    ...     "cifar100",
    ...     n_tasks=10,
    ...     batch_size=64
    ... )
    >>> 
    >>> # Or import directly
    >>> from cont_src.data import CIFAR100ContinualDataset
    >>> dataset = CIFAR100ContinualDataset(n_tasks=10)
"""

from typing import Dict, Any

# Base classes
from cont_src.data.base_dataset import (
    BaseContinualDataset,
    AvalancheDatasetWrapper,
    TaskData,
    DatasetInfo,
    DATASET_INFO,
)

# CIFAR datasets
from cont_src.data.cifar import (
    CIFAR10ContinualDataset,
    CIFAR100ContinualDataset,
    get_cifar10_continual,
    get_cifar100_continual,
)

# MNIST
from cont_src.data.mnist import (
    MNISTContinualDataset,
    get_mnist_continual,
)

# ImageNet variants
from cont_src.data.imagenet import (
    TinyImageNetContinualDataset,
    get_tiny_imagenet_continual,
)

__all__ = [
    # Base classes
    "BaseContinualDataset",
    "AvalancheDatasetWrapper",
    "TaskData",
    "DatasetInfo",
    "DATASET_INFO",
    
    # CIFAR
    "CIFAR10ContinualDataset",
    "CIFAR100ContinualDataset",
    "get_cifar10_continual",
    "get_cifar100_continual",
    
    # MNIST
    "MNISTContinualDataset",
    "get_mnist_continual",
    
    # ImageNet
    "TinyImageNetContinualDataset",
    "get_tiny_imagenet_continual",
]
