"""
Data Pipeline for Continual Learning

This module provides data loaders and benchmarks for continual learning experiments.

Two implementations available:
1. Custom implementation (continual_cifar100.py) - for learning/understanding
2. Avalanche implementation (continual_cifar100_avalanche.py) - recommended for production
"""

# Custom implementation (coded from scratch for understanding)
from .continual_cifar100 import (
    ContinualCIFAR100Dataset,
    get_continual_cifar100_loaders,
    ClassIncrementalSplit,
)

# Avalanche implementation (uses built-in datasets)
from .continual_cifar100_avalanche import (
    get_avalanche_cifar100_benchmark,
    get_avalanche_loaders_from_benchmark,
)
from .continual_tinyimagenet import (
    get_tinyimagenet_benchmark,
)

__all__ = [
    # Custom implementation
    "ContinualCIFAR100Dataset",
    "get_continual_cifar100_loaders",
    "ClassIncrementalSplit",
    # Avalanche CIFAR-100 (32x32, 100 classes)
    "get_avalanche_cifar100_benchmark",
    "get_avalanche_loaders_from_benchmark",
    # Avalanche Tiny-ImageNet (64x64, 200 classes) - recommended for larger images
    "get_tinyimagenet_benchmark",
]

