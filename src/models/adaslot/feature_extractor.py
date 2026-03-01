"""
Feature Extractor for SlotAttention / AdaSlot.

Mirrors: ocl.feature_extractors.SlotAttentionFeatureExtractor
Checkpoint key prefix: models.feature_extractor.layers.*
"""

import torch
import torch.nn as nn
from typing import Tuple


def cnn_compute_positions_and_flatten(features: torch.Tensor):
    """Flatten CNN output and compute normalized positions."""
    spatial_dims = features.shape[2:]  # (H, W)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # (B, C, H, W) -> (B, H*W, C)
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return positions, flattened


class SlotAttentionFeatureExtractor(nn.Module):
    """
    CNN feature extractor as used in the original Slot Attention paper.

    4 layers of Conv2d(kernel=5, padding=same, stride=1), 64 channels, ReLU.
    Output: (B, H*W, 64) features + (H*W, 2) positions.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

    @property
    def feature_dim(self):
        return 64

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            positions: (H*W, 2) normalized positions in [0, 1]
            features:  (B, H*W, 64) flattened feature maps
        """
        feat = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(feat)
        return positions, flattened
