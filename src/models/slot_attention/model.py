"""
Complete Slot Attention Model

Combines encoder, slot attention, and decoder into a full model for
object-centric learning.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .encoder import CNNEncoder, ResNetEncoder
from .slot_attention import SlotAttention, PositionalEmbedding
from .decoder import BroadcastDecoder, MLPDecoder


class SlotAttentionAutoEncoder(nn.Module):
    """
    Complete Slot Attention Auto-Encoder.
    
    This model:
    1. Encodes input images to features
    2. Applies Slot Attention to discover objects
    3. Decodes slots to reconstruct the image
    
    Args:
        resolution (Tuple[int, int]): Image resolution (height, width).
        num_slots (int): Number of slots (max objects).
        num_iterations (int): Number of slot attention iterations.
        in_channels (int): Number of input channels (3 for RGB).
        slot_dim (int): Dimension of slot representations.
        hidden_dim (int): Hidden dimension for encoder/decoder.
        encoder_type (str): Type of encoder ('cnn' or 'resnet'). Default: 'cnn'.
        decoder_type (str): Type of decoder ('broadcast' or 'mlp'). Default: 'broadcast'.
        
    Input:
        image: torch.Tensor of shape (batch_size, in_channels, height, width)
        
    Output (dict):
        - 'reconstruction': Reconstructed image
        - 'slots': Slot representations
        - 'masks': Attention masks
        - 'attn_weights': Feature attention weights
        - 'slot_recons': Per-slot reconstructions
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (64, 64),
        num_slots: int = 7,
        num_iterations: int = 3,
        in_channels: int = 3,
        slot_dim: int = 64,
        hidden_dim: int = 64,
        encoder_type: str = 'cnn',
        decoder_type: str = 'broadcast',
    ):
        super().__init__()
        
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        
        # Build encoder
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(
                in_channels=in_channels,
                out_dim=slot_dim,
                hidden_dims=(hidden_dim,) * 4,
            )
        elif encoder_type == 'resnet':
            self.encoder = ResNetEncoder(
                model_name='resnet18',
                out_dim=slot_dim,
                pretrained=True,
                freeze_backbone=False,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Compute encoder output resolution
        # For CNN encoder with stride=1, resolution stays the same
        # For ResNet, resolution is downsampled by 32
        if encoder_type == 'cnn':
            encoder_resolution = resolution
        else:  # resnet
            encoder_resolution = (resolution[0] // 32, resolution[1] // 32)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(
            resolution=encoder_resolution,
            feature_dim=slot_dim,
        )
        
        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=slot_dim,
            n_iters=num_iterations,
            hidden_dim=hidden_dim * 2,
        )
        
        # Build decoder
        if decoder_type == 'broadcast':
            self.decoder = BroadcastDecoder(
                slot_dim=slot_dim,
                out_channels=in_channels,
                resolution=resolution,
                hidden_dims=(hidden_dim,) * 4,
            )
        elif decoder_type == 'mlp':
            self.decoder = MLPDecoder(
                slot_dim=slot_dim,
                out_channels=in_channels,
                resolution=resolution,
                hidden_dim=hidden_dim * 4,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(
        self, image: torch.Tensor, slots_init: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Slot Attention AutoEncoder.
        
        Args:
            image: Input image of shape (batch_size, in_channels, height, width).
            slots_init: Optional initial slots. If None, initialized randomly.
            
        Returns:
            Dictionary containing:
            - 'reconstruction': Reconstructed image
            - 'slots': Slot representations
            - 'masks': Attention masks
            - 'attn_weights': Feature attention weights
            - 'slot_recons': Per-slot reconstructions
        """
        # Encode image to features
        features = self.encoder(image)  # (B, num_features, slot_dim)
        
        # Add positional embeddings
        features = self.pos_embed(features)
        
        # Apply slot attention
        slots, attn_weights = self.slot_attention(features, slots_init)
        
        # Decode slots to reconstruction
        reconstruction, masks, slot_recons = self.decoder(slots)
        
        return {
            'reconstruction': reconstruction,
            'slots': slots,
            'masks': masks,
            'attn_weights': attn_weights,
            'slot_recons': slot_recons,
        }
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to slot representations.
        
        Args:
            image: Input image.
            
        Returns:
            slots: Slot representations.
        """
        features = self.encoder(image)
        features = self.pos_embed(features)
        slots, _ = self.slot_attention(features)
        return slots
    
    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Decode slots to reconstructed image.
        
        Args:
            slots: Slot representations.
            
        Returns:
            reconstruction: Reconstructed image.
        """
        reconstruction, _, _ = self.decoder(slots)
        return reconstruction


def build_slot_attention_model(
    resolution: Tuple[int, int] = (64, 64),
    num_slots: int = 7,
    num_iterations: int = 3,
    **kwargs
) -> SlotAttentionAutoEncoder:
    """
    Builder function for Slot Attention model.
    
    Args:
        resolution: Image resolution.
        num_slots: Number of slots.
        num_iterations: Number of slot attention iterations.
        **kwargs: Additional arguments for SlotAttentionAutoEncoder.
        
    Returns:
        Slot Attention model.
    """
    return SlotAttentionAutoEncoder(
        resolution=resolution,
        num_slots=num_slots,
        num_iterations=num_iterations,
        **kwargs
    )


if __name__ == "__main__":
    # Test complete model
    print("Testing Slot Attention AutoEncoder...")
    
    batch_size = 4
    resolution = (64, 64)
    num_slots = 7
    
    # Build model
    model = SlotAttentionAutoEncoder(
        resolution=resolution,
        num_slots=num_slots,
        num_iterations=3,
        encoder_type='cnn',
        decoder_type='broadcast',
    )
    
    # Create random input
    image = torch.randn(batch_size, 3, *resolution)
    
    # Forward pass
    outputs = model(image)
    
    print(f"\nModel test:")
    print(f"  Input shape: {image.shape}")
    print(f"  Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"  Slots shape: {outputs['slots'].shape}")
    print(f"  Masks shape: {outputs['masks'].shape}")
    print(f"  Attention weights shape: {outputs['attn_weights'].shape}")
    print(f"  Slot reconstructions shape: {outputs['slot_recons'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\nSlot Attention AutoEncoder test passed!")

