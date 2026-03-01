"""
Loss functions for continual learning pipeline.
"""

from .contrastive import (
    SupervisedContrastiveLoss,
    PrototypeLoss,
    SlotClusteringLoss
)

__all__ = [
    'SupervisedContrastiveLoss',
    'PrototypeLoss',
    'SlotClusteringLoss',
]
