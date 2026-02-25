"""
Decoder for AdaSlot.

Mirrors: ocl.decoding.SlotAttentionDecoder with get_slotattention_decoder_backbone
Checkpoint key prefix: models.object_decoder.*
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple, Union


def build_grid_of_positions(resolution: Tuple[int, int]) -> torch.Tensor:
    """Build grid of positions for positional embeddings."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid


def get_slotattention_decoder_backbone(
    object_dim: int, output_dim: int = 4, n_upsample: int = 4
) -> nn.Sequential:
    """
    CNN decoder backbone from the original Slot Attention paper.

    Broadcasts slots from an 8×8 base grid up to ``8 * 2**n_upsample`` pixels.
    Default n_upsample=4 → 128×128 output (matches original paper + pretrained ckpt).
    Use n_upsample=3 for 64×64, n_upsample=5 for 256×256, etc.

    Architecture: n_upsample × ConvTranspose2d(stride=2)
                  + 1 × ConvTranspose2d(stride=1)  (refinement)
                  + 1 × final Conv
    """
    import math
    layers: list = []
    in_ch = object_dim
    for _ in range(n_upsample):
        layers += [
            nn.ConvTranspose2d(in_ch, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
        ]
        in_ch = 64
    # Refinement (stride=1)
    layers += [
        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim, 3, stride=1, padding=1, output_padding=0),
    ]
    return nn.Sequential(*layers)


def get_activation_fn(name: Union[str, Callable]):
    """Get activation function by name."""
    if callable(name):
        return name
    if name == "identity" or name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class SlotAttentionDecoder(nn.Module):
    """
    Decoder used in the original Slot Attention paper.

    Broadcasts each slot to an 8x8 spatial grid, optionally adds positional
    embedding, then applies a deconvolution network. Splits output into
    RGB (3ch) and alpha (1ch), normalizes alpha via softmax across slots.
    """

    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = "identity",
        positional_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer(
                "grid", build_grid_of_positions(self.initial_conv_size)
            )

    def forward(self, object_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode slot features into image reconstructions.

        Args:
            object_features: (B, num_slots, slot_dim) slot representations.

        Returns:
            Dict with 'reconstruction', 'object_reconstructions', 'masks'.
        """
        assert object_features.dim() >= 3
        initial_shape = object_features.shape[:-1]  # (B, num_slots)
        object_features = object_features.flatten(0, -2)  # (B*num_slots, slot_dim)

        # Broadcast to spatial grid
        object_features = (
            object_features.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, *self.initial_conv_size)
        )  # (B*num_slots, slot_dim, 8, 8)

        if self.positional_embedding:
            object_features = self.positional_embedding(
                object_features, self.grid.unsqueeze(0)
            )

        # Apply deconvolution
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split RGB and alpha
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)
        alpha = alpha.softmax(dim=-4)  # Normalize across slots

        return {
            "reconstruction": (rgb * alpha).sum(-4),
            "object_reconstructions": rgb,
            "masks": alpha.squeeze(-3),
        }
