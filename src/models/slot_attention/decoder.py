"""
Decoder for Slot Attention Model

Implements decoders that reconstruct images from slot representations.
"""

from typing import Tuple
import torch
import torch.nn as nn


class BroadcastDecoder(nn.Module):
    """
    Decoder that broadcasts each slot to a spatial grid and decodes independently.
    
    This decoder takes each slot representation, broadcasts it to a 2D grid,
    and decodes it to produce per-slot reconstructions and masks.
    The final reconstruction is a weighted sum of slot reconstructions.
    
    Args:
        slot_dim (int): Dimension of slot representations.
        hidden_dims (Tuple[int, ...]): Hidden dimensions for deconv layers.
            Default: (64, 64, 64, 64).
        out_channels (int): Number of output channels (3 for RGB). Default: 3.
        resolution (Tuple[int, int]): Output resolution (height, width). Default: (64, 64).
        kernel_size (int): Kernel size for deconv layers. Default: 5.
        
    Input:
        slots: torch.Tensor of shape (batch_size, num_slots, slot_dim)
        
    Output:
        reconstruction: torch.Tensor of shape (batch_size, out_channels, height, width)
            Reconstructed image.
        masks: torch.Tensor of shape (batch_size, num_slots, height, width)
            Attention masks showing which pixels each slot explains.
        slot_recons: torch.Tensor of shape (batch_size, num_slots, out_channels, height, width)
            Per-slot reconstructions before combining.
    """
    
    def __init__(
        self,
        slot_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        out_channels: int = 3,
        resolution: Tuple[int, int] = (64, 64),
        kernel_size: int = 5,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.out_channels = out_channels
        self.resolution = resolution
        
        # Initial spatial size (will be upsampled)
        self.initial_size = (8, 8)
        
        # Build decoder layers
        layers = []
        prev_dim = slot_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(
                    prev_dim, hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                ),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Final layer: output RGB + alpha mask
        # Output channels: out_channels (RGB) + 1 (alpha mask)
        layers.append(
            nn.ConvTranspose2d(
                prev_dim, out_channels + 1,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        
        self.decoder = nn.Sequential(*layers)
        
        # Position grid for broadcasting slots
        self.register_buffer(
            "pos_grid",
            self._build_position_grid(self.initial_size),
        )
    
    def _build_position_grid(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Build position grid for broadcasting slots.
        
        Args:
            size: Grid size (height, width).
            
        Returns:
            Position grid of shape (1, height, width, 4) containing
            normalized x, y coordinates in both linear and sinusoidal form.
        """
        h, w = size
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        
        # Create meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Stack coordinates
        grid = torch.stack([xx, yy], dim=-1)  # (h, w, 2)
        grid = grid.unsqueeze(0)  # (1, h, w, 2)
        
        return grid
    
    def forward(
        self, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode slots to reconstructed image.
        
        Args:
            slots: Slot representations of shape (batch_size, num_slots, slot_dim).
            
        Returns:
            reconstruction: Reconstructed image (batch_size, out_channels, height, width).
            masks: Attention masks (batch_size, num_slots, height, width).
            slot_recons: Per-slot reconstructions (batch_size, num_slots, out_channels, height, width).
        """
        batch_size, num_slots, slot_dim = slots.shape
        
        # Broadcast slots to spatial dimensions
        # slots: (B, num_slots, slot_dim) -> (B*num_slots, slot_dim, h, w)
        h, w = self.initial_size
        slots_spatial = slots.reshape(batch_size * num_slots, slot_dim, 1, 1)
        slots_spatial = slots_spatial.expand(-1, -1, h, w)
        
        # Decode each slot
        # (B*num_slots, slot_dim, h, w) -> (B*num_slots, out_channels+1, H, W)
        decoded = self.decoder(slots_spatial)
        
        # Resize to target resolution if needed
        if decoded.shape[2:] != self.resolution:
            decoded = nn.functional.interpolate(
                decoded, size=self.resolution, mode='bilinear', align_corners=False
            )
        
        # Split into RGB and alpha
        # decoded: (B*num_slots, out_channels+1, H, W)
        decoded = decoded.reshape(
            batch_size, num_slots, self.out_channels + 1,
            self.resolution[0], self.resolution[1]
        )
        
        slot_recons = decoded[:, :, :self.out_channels, :, :]  # (B, num_slots, out_channels, H, W)
        alpha_logits = decoded[:, :, -1:, :, :]  # (B, num_slots, 1, H, W)
        
        # Apply softmax to alpha masks across slots
        masks = torch.softmax(alpha_logits, dim=1)  # (B, num_slots, 1, H, W)
        masks = masks.squeeze(2)  # (B, num_slots, H, W)
        
        # Compute weighted reconstruction
        # (B, num_slots, out_channels, H, W) * (B, num_slots, 1, H, W)
        reconstruction = (slot_recons * masks.unsqueeze(2)).sum(dim=1)
        
        return reconstruction, masks, slot_recons


class MLPDecoder(nn.Module):
    """
    Simple MLP-based decoder for slots.
    
    This decoder uses MLPs to decode each slot independently, then combines them.
    Simpler but less powerful than BroadcastDecoder.
    
    Args:
        slot_dim (int): Dimension of slot representations.
        hidden_dim (int): Hidden dimension for MLP. Default: 256.
        out_channels (int): Number of output channels. Default: 3.
        resolution (Tuple[int, int]): Output resolution (height, width). Default: (64, 64).
    """
    
    def __init__(
        self,
        slot_dim: int,
        hidden_dim: int = 256,
        out_channels: int = 3,
        resolution: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.out_channels = out_channels
        self.resolution = resolution
        
        output_size = out_channels * resolution[0] * resolution[1]
        
        # MLP for each slot
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size + resolution[0] * resolution[1]),
            # +resolution for alpha mask
        )
    
    def forward(
        self, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode slots using MLP.
        
        Args:
            slots: Slot representations of shape (batch_size, num_slots, slot_dim).
            
        Returns:
            reconstruction: Reconstructed image.
            masks: Attention masks.
            slot_recons: Per-slot reconstructions.
        """
        batch_size, num_slots, _ = slots.shape
        h, w = self.resolution
        
        # Decode each slot
        # slots: (B, num_slots, slot_dim) -> (B*num_slots, slot_dim)
        decoded = self.mlp(slots.reshape(batch_size * num_slots, self.slot_dim))
        
        # Reshape output
        rgb_size = self.out_channels * h * w
        alpha_size = h * w
        
        decoded = decoded.reshape(batch_size, num_slots, rgb_size + alpha_size)
        
        # Split RGB and alpha
        slot_recons = decoded[:, :, :rgb_size]
        alpha_logits = decoded[:, :, rgb_size:]
        
        # Reshape
        slot_recons = slot_recons.reshape(batch_size, num_slots, self.out_channels, h, w)
        alpha_logits = alpha_logits.reshape(batch_size, num_slots, h, w)
        
        # Softmax for masks
        masks = torch.softmax(alpha_logits, dim=1)
        
        # Weighted reconstruction
        reconstruction = (slot_recons * masks.unsqueeze(2)).sum(dim=1)
        
        return reconstruction, masks, slot_recons


if __name__ == "__main__":
    # Test decoders
    print("Testing Broadcast Decoder...")
    
    batch_size = 4
    num_slots = 7
    slot_dim = 64
    resolution = (64, 64)
    
    # Broadcast Decoder
    decoder = BroadcastDecoder(
        slot_dim=slot_dim,
        resolution=resolution,
        out_channels=3,
    )
    
    slots = torch.randn(batch_size, num_slots, slot_dim)
    reconstruction, masks, slot_recons = decoder(slots)
    
    print(f"Broadcast Decoder:")
    print(f"  Slots shape: {slots.shape}")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Slot reconstructions shape: {slot_recons.shape}")
    print(f"  Masks sum (should be ~1): {masks.sum(dim=1).mean():.4f}")
    
    # MLP Decoder
    print("\nTesting MLP Decoder...")
    mlp_decoder = MLPDecoder(
        slot_dim=slot_dim,
        resolution=resolution,
        out_channels=3,
    )
    
    reconstruction, masks, slot_recons = mlp_decoder(slots)
    
    print(f"MLP Decoder:")
    print(f"  Slots shape: {slots.shape}")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Slot reconstructions shape: {slot_recons.shape}")
    
    print("\nDecoder tests passed!")

