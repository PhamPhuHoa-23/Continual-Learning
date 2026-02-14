"""
Slot Attention Module Implementation

This module implements the Slot Attention mechanism from the paper:
"Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
https://arxiv.org/abs/2006.15055

Adapted from:
- Amazon AdaSlot: https://github.com/amazon-science/AdaSlot
- Phil Wang's implementation: https://github.com/lucidrains/slot-attention
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric learning.

    Slot Attention iteratively refines a set of slot representations by attending
    to input features. Each slot competes to explain different parts of the input,
    enabling unsupervised object discovery.

    Args:
        num_slots (int): Number of slots (object representations).
        slot_dim (int): Dimensionality of each slot.
        feature_dim (int): Dimensionality of input features.
        kvq_dim (Optional[int]): Dimensionality for key, value, query projections.
            If None, uses slot_dim.
        n_heads (int): Number of attention heads. Default: 1.
        n_iters (int): Number of slot attention iterations. Default: 3.
        eps (float): Small constant for numerical stability. Default: 1e-8.
        hidden_dim (int): Hidden dimension for MLP. If 0, no MLP is used. Default: 128.
        use_projection_bias (bool): Whether to use bias in K, V, Q projections. Default: False.

    Input:
        features: torch.Tensor of shape (batch_size, num_features, feature_dim)
            Input features from encoder.
        slots_init: Optional[torch.Tensor] of shape (batch_size, num_slots, slot_dim)
            Initial slot values. If None, slots are initialized randomly.

    Output:
        slots: torch.Tensor of shape (batch_size, num_slots, slot_dim)
            Refined slot representations.
        attn_weights: torch.Tensor of shape (batch_size, num_slots, num_features)
            Attention weights showing which features each slot attends to.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        n_iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        use_projection_bias: bool = False,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.n_iters = n_iters
        self.eps = eps
        
        # Use slot_dim for KVQ if not specified
        self.kvq_dim = kvq_dim if kvq_dim is not None else slot_dim
        
        # Ensure kvq_dim is divisible by number of heads
        if self.kvq_dim % self.n_heads != 0:
            raise ValueError(
                f"kvq_dim ({self.kvq_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head ** -0.5  # Attention scaling factor
        
        # Linear projections for queries, keys, values
        self.to_q = nn.Linear(slot_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        
        # GRU for updating slots
        self.gru = nn.GRUCell(self.kvq_dim, slot_dim)
        
        # Layer normalization
        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        
        # Optional MLP applied to slots after GRU update
        if hidden_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, slot_dim)
            )
        else:
            self.mlp = None
        
        # Learnable slot initialization (will be used if slots_init is None)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
    
    def init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slots from a learned Gaussian distribution.
        
        Args:
            batch_size (int): Batch size.
            device (torch.device): Device for tensors.
            
        Returns:
            torch.Tensor of shape (batch_size, num_slots, slot_dim)
        """
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots
    
    def attention_step(
        self,
        slots: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single iteration of slot attention.
        
        Args:
            slots: Current slot representations (batch_size, num_slots, slot_dim).
            k: Key projections of features (batch_size, num_features, n_heads, dims_per_head).
            v: Value projections of features (batch_size, num_features, n_heads, dims_per_head).
            
        Returns:
            updated_slots: Updated slot representations.
            attn_weights: Attention weights (batch_size, num_slots, num_features).
        """
        batch_size, num_slots, _ = slots.shape
        
        # Normalize slots and compute queries
        slots_normalized = self.norm_slots(slots)
        q = self.to_q(slots_normalized).view(
            batch_size, num_slots, self.n_heads, self.dims_per_head
        )
        
        # Compute attention logits
        attn_logits = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        
        # Softmax over slots and heads to get attention weights
        attn = attn_logits.flatten(1, 2).softmax(dim=1)
        attn = attn.view(batch_size, num_slots, self.n_heads, -1)
        
        # Store attention before normalization
        attn_weights = attn.mean(dim=2)  # Average over heads
        
        # Normalize attention weights
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Compute weighted sum of values
        updates = torch.einsum("bjhd,bihj->bihd", v, attn)
        updates = updates.reshape(batch_size * num_slots, self.kvq_dim)
        
        # Update slots with GRU
        slots_flat = slots.reshape(batch_size * num_slots, self.slot_dim)
        updated_slots = self.gru(updates, slots_flat)
        updated_slots = updated_slots.reshape(batch_size, num_slots, self.slot_dim)
        
        # Apply MLP if present
        if self.mlp is not None:
            updated_slots = updated_slots + self.mlp(updated_slots)
        
        return updated_slots, attn_weights
    
    def forward(
        self,
        features: torch.Tensor,
        slots_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Slot Attention.
        
        Args:
            features: Input features of shape (batch_size, num_features, feature_dim).
            slots_init: Optional initial slots. If None, slots are initialized randomly.
            
        Returns:
            slots: Final slot representations (batch_size, num_slots, slot_dim).
            attn_weights: Final attention weights (batch_size, num_slots, num_features).
        """
        batch_size, num_features, _ = features.shape
        
        # Initialize slots if not provided
        if slots_init is None:
            slots = self.init_slots(batch_size, features.device)
        else:
            slots = slots_init
        
        # Normalize input features and compute keys and values
        features_normalized = self.norm_input(features)
        k = self.to_k(features_normalized).view(
            batch_size, num_features, self.n_heads, self.dims_per_head
        )
        v = self.to_v(features_normalized).view(
            batch_size, num_features, self.n_heads, self.dims_per_head
        )
        
        # Iterate slot attention
        for _ in range(self.n_iters):
            slots, attn_weights = self.attention_step(slots, k, v)
        
        return slots, attn_weights


class PositionalEmbedding(nn.Module):
    """
    Adds learnable positional embeddings to features.
    
    This is commonly used to add spatial information to CNN features
    before passing them to Slot Attention.
    
    Args:
        resolution (Tuple[int, int]): Spatial resolution (height, width).
        feature_dim (int): Feature dimension.
    """
    
    def __init__(self, resolution: Tuple[int, int], feature_dim: int):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, resolution[0] * resolution[1], feature_dim) * 0.02
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to features.
        
        Args:
            features: Features of shape (batch_size, num_features, feature_dim).
            
        Returns:
            Features with added positional embeddings.
        """
        return features + self.pos_embed


if __name__ == "__main__":
    # Simple test
    print("Testing Slot Attention module...")
    
    batch_size = 4
    num_features = 256  # e.g., 16x16 spatial locations
    feature_dim = 64
    num_slots = 7
    slot_dim = 64
    
    # Create module
    slot_attn = SlotAttention(
        num_slots=num_slots,
        slot_dim=slot_dim,
        feature_dim=feature_dim,
        n_iters=3,
    )
    
    # Create random features
    features = torch.randn(batch_size, num_features, feature_dim)
    
    # Forward pass
    slots, attn_weights = slot_attn(features)
    
    print(f"Input features shape: {features.shape}")
    print(f"Output slots shape: {slots.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1): {attn_weights.sum(dim=-1).mean():.4f}")
    
    print("\nSlot Attention module test passed!")

