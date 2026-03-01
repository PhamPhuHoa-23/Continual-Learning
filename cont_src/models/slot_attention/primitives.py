"""
Primitive Selection and Slot Decoding

Implements primitive selection mechanism and slot decoder from CompSLOT paper.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.core.base_module import BaseModule
from cont_src.core.registry import MODEL_REGISTRY


class PrimitiveSelector(BaseModule):
    """
    Primitive Selection Mechanism from CompSLOT.

    Aggregates slots into a single primitive representation using
    attention-based weighting. Learns which slots are class-relevant.

    Process:
    1. Map slots to similarity space
    2. Compute similarity to learnable primitive key
    3. Use similarity as attention weights
    4. Weighted sum of slots → primitive

    Reference: CompSLOT paper Section 4.1, Equation (2)
    """

    def __init__(
        self,
        slot_dim: int,
        hidden_dim: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Args:
            slot_dim: Dimension of slot representations (D_s)
            hidden_dim: Hidden dimension for projection (default: slot_dim)
            temperature: Temperature for attention softmax (auto if None)
        """
        super().__init__()

        self.slot_dim = slot_dim
        hidden_dim = hidden_dim or slot_dim

        # Projection to similarity space
        self.projection = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.Tanh()
        )

        # Learnable primitive key (K_p in paper)
        self.primitive_key = nn.Parameter(torch.randn(hidden_dim))

        # Temperature (τ_t in paper)
        if temperature is None:
            # Auto temperature: 100/√D
            self.temperature = 100.0 / (hidden_dim ** 0.5)
        else:
            self.temperature = temperature

    def forward(
        self,
        slots: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Select primitive from slots.

        Args:
            slots: Slot representations (B, K, D_s)
            slot_mask: Optional binary mask (B, K) - 0 for dropped slots
            return_weights: If True, also return attention weights

        Returns:
            primitives: Primitive representations (B, D_s)
            weights: Attention weights (B, K) if return_weights=True
        """
        B, K, D_s = slots.shape

        # Project slots to similarity space
        projected = self.projection(slots)  # (B, K, D_h)

        # Compute similarity to primitive key
        # sim = ¯S @ K_p
        similarity = torch.matmul(
            projected,
            self.primitive_key
        )  # (B, K)

        # Apply temperature and softmax
        # w_p = softmax(τ_t * ¯S @ K_p)
        weights = F.softmax(self.temperature * similarity, dim=-1)  # (B, K)

        # Apply slot mask if provided (zero out dropped slots)
        if slot_mask is not None:
            weights = weights * slot_mask
            # Renormalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum: s_p = w_p^T @ ¯S
        primitives = torch.einsum('bk,bkd->bd', weights, projected)  # (B, D_h)

        if return_weights:
            return primitives, weights
        return primitives


class SlotDecoder(BaseModule):
    """
    Slot Decoder for reconstruction.

    Decodes slot representations back to patch feature space.
    Used for reconstruction loss in training.

    Architecture:
    - MLP decoder per slot
    - Position encoding added
    - Attention-weighted aggregation
    """

    def __init__(
        self,
        slot_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        """
        Args:
            slot_dim: Dimension of slot representations
            output_dim: Output dimension (patch feature dim)
            hidden_dim: Hidden dimension (default: slot_dim)
            num_layers: Number of MLP layers
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.output_dim = output_dim

        hidden_dim = hidden_dim or slot_dim

        # MLP decoder
        layers = []
        in_dim = slot_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(
        self,
        slots: torch.Tensor,
        attention: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode slots to patch features.

        Args:
            slots: Slot representations (B, K, D_s)
            attention: Attention weights (B, K, N)
            positions: Optional position encodings (N, D_s)

        Returns:
            reconstructed: Reconstructed patch features (B, N, D_f)
        """
        B, K, D_s = slots.shape
        _, _, N = attention.shape

        # Add position encoding if provided
        if positions is not None:
            # Expand positions: (N, D_s) -> (B, N, K, D_s)
            pos_expanded = positions.unsqueeze(0).unsqueeze(2)
            pos_expanded = pos_expanded.expand(B, -1, K, -1)

            # Add to slots: (B, K, D_s) -> (B, N, K, D_s)
            slots_with_pos = slots.unsqueeze(1) + pos_expanded
        else:
            slots_with_pos = slots.unsqueeze(1).expand(B, N, K, D_s)

        # Decode each slot at each position
        decoded = self.decoder(slots_with_pos)  # (B, N, K, D_f)

        # Weighted sum with attention
        # attention: (B, K, N) -> (B, N, K)
        attn_permuted = attention.permute(0, 2, 1)  # (B, N, K)

        # Aggregate: (B, N, K) @ (B, N, K, D_f) -> (B, N, D_f)
        reconstructed = torch.einsum('bnk,bnkd->bnd', attn_permuted, decoded)

        return reconstructed


class AdaSlotModule(BaseModule):
    """
    Complete AdaSlot module with primitive selection and decoding.

    Combines:
    - Adaptive Slot Attention
    - Primitive Selection
    - Slot Decoder

    Suitable for continual learning with reconstruction and primitive losses.
    """

    def __init__(
        self,
        # Slot attention params
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        num_iterations: int = 3,
        num_heads: int = 1,

        # Adaptive params
        use_gumbel: bool = True,
        gumbel_low_bound: int = 1,

        # Primitive selection params
        use_primitive: bool = True,
        primitive_hidden_dim: Optional[int] = None,

        # Decoder params
        use_decoder: bool = True,
        decoder_hidden_dim: Optional[int] = None,
        decoder_layers: int = 2,

        **kwargs
    ):
        """
        Args:
            num_slots: Maximum number of slots
            slot_dim: Slot dimension
            feature_dim: Input feature dimension
            num_iterations: Slot attention iterations
            num_heads: Number of attention heads
            use_gumbel: Enable adaptive slot selection
            gumbel_low_bound: Minimum slots to keep
            use_primitive: Enable primitive selection
            primitive_hidden_dim: Primitive projection dimension
            use_decoder: Enable reconstruction decoder
            decoder_hidden_dim: Decoder hidden dimension
            decoder_layers: Number of decoder layers
            **kwargs: Additional arguments for AdaptiveSlotAttention
        """
        super().__init__()

        # Import here to avoid circular dependency
        from cont_src.models.slot_attention import AdaptiveSlotAttention

        # Slot attention
        self.slot_attention = AdaptiveSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
            num_iterations=num_iterations,
            num_heads=num_heads,
            use_gumbel=use_gumbel,
            gumbel_low_bound=gumbel_low_bound,
            **kwargs
        )

        # Primitive selector
        self.use_primitive = use_primitive
        if use_primitive:
            self.primitive_selector = PrimitiveSelector(
                slot_dim=slot_dim,
                hidden_dim=primitive_hidden_dim
            )

        # Decoder
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = SlotDecoder(
                slot_dim=slot_dim,
                output_dim=feature_dim,
                hidden_dim=decoder_hidden_dim or slot_dim,
                num_layers=decoder_layers
            )

            # Learnable position encoding for decoder
            # Assume max 256 patches (16x16 grid)
            self.register_buffer(
                'pos_embed',
                torch.randn(256, slot_dim) * (slot_dim ** -0.5)
            )

    def forward(
        self,
        features: torch.Tensor,
        global_step: Optional[int] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of complete AdaSlot module.

        Args:
            features: Input features (B, N, D_f)
            global_step: Training step for temperature annealing
            return_all: Return all intermediate outputs

        Returns:
            Dictionary with:
                - slots: Refined slots (B, K, D_s)
                - attention: Attention maps (B, K, N)
                - slot_mask: Binary slot mask (B, K)
                - slot_probs: Soft slot probabilities (B, K)
                - primitives: Selected primitives (B, D_s) if enabled
                - primitive_weights: Primitive attention weights (B, K) if enabled
                - reconstruction: Reconstructed features (B, N, D_f) if enabled
        """
        B, N, D_f = features.shape

        # Slot attention
        outputs = self.slot_attention(
            features,
            global_step=global_step,
            return_all=return_all
        )

        slots = outputs["slots"]
        attention = outputs["attention"]
        slot_mask = outputs.get("slot_mask", None)

        # Primitive selection
        if self.use_primitive:
            primitives, prim_weights = self.primitive_selector(
                slots,
                slot_mask=slot_mask,
                return_weights=True
            )
            outputs["primitives"] = primitives
            outputs["primitive_weights"] = prim_weights

        # Reconstruction
        if self.use_decoder:
            # Get position embeddings
            pos = self.pos_embed[:N]  # (N, D_s)

            reconstructed = self.decoder(
                slots,
                attention,
                positions=pos
            )
            outputs["reconstruction"] = reconstructed

        return outputs


# Register modules
MODEL_REGISTRY.register("primitive_selector")(PrimitiveSelector)
MODEL_REGISTRY.register("slot_decoder")(SlotDecoder)
MODEL_REGISTRY.register("adaslot_module")(AdaSlotModule)
