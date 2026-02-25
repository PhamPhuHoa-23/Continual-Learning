"""
Data Pipeline for Continual Learning

This module provides data loaders and benchmarks for continual learning experiments.
Uses standalone implementations (no Avalanche dependency).
"""

# Standalone TinyImageNet (no Avalanche required) — primary dataset
from .continual_tinyimagenet import get_continual_tinyimagenet_loaders

__all__ = [
    "get_continual_tinyimagenet_loaders",
]

