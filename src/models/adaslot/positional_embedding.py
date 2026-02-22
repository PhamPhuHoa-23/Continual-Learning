"""
Positional Embedding for AdaSlot.

Mirrors: ocl.neural_networks.positional_embedding.SoftPositionEmbed
Checkpoint key prefix:
  - models.perceptual_grouping.positional_embedding.layers.*
  - models.object_decoder.positional_embedding.dense.*
"""

import torch
import torch.nn as nn


class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding using a linear projection of positions.

    For standard mode: positions are concatenated with (1-positions) -> 4D input.
    For SAVi style: positions are rescaled to [-1, 1] -> 2D input.

    Args:
        n_spatial_dims: Number of spatial dimensions (typically 2).
        feature_dim: Feature dimension to project into.
        cnn_channel_order: If True, permute output for CNN channel order (C,H,W).
        savi_style: If True, use SAVi-style positional encoding.
    """

    def __init__(
        self,
        n_spatial_dims: int,
        feature_dim: int,
        cnn_channel_order: bool = False,
        savi_style: bool = False,
    ):
        super().__init__()
        self.savi_style = savi_style
        n_features = n_spatial_dims if savi_style else 2 * n_spatial_dims
        self.dense = nn.Linear(in_features=n_features, out_features=feature_dim)
        self.cnn_channel_order = cnn_channel_order

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to inputs.

        Args:
            inputs: Features tensor.
            positions: Position tensor with values in [0, 1].

        Returns:
            inputs + positional_embedding
        """
        if self.savi_style:
            positions = (positions - 0.5) * 2
        else:
            positions = torch.cat([positions, 1 - positions], axis=-1)
        emb_proj = self.dense(positions)
        if self.cnn_channel_order:
            emb_proj = emb_proj.permute(*range(inputs.ndim - 3), -1, -3, -2)
        return inputs + emb_proj
