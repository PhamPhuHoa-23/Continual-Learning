"""
Adaptive Slot Attention Module

Implementation of AdaSlot (Adaptive Slot Attention) with Gumbel-Softmax selection.
Dynamically selects relevant slots during training using differentiable sampling.

Reference:
- AdaSlot paper and implementation
- CompSLOT framework (ICLR 2026)
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.core.base_module import BaseModule
from cont_src.core.registry import MODEL_REGISTRY


def sample_slot_lower_bound(mask: torch.Tensor, lower_bound: int = 1) -> torch.Tensor:
    """
    Ensure at least `lower_bound` slots are kept by sampling from dropped slots.

    Args:
        mask: Binary slot mask (B, K), 0=drop, 1=keep
        lower_bound: Minimum number of slots to keep

    Returns:
        Additional mask to combine with original (B, K)
    """
    B, K = mask.shape
    device = mask.device

    additional_mask = torch.zeros_like(mask)
    slots_kept = (mask != 0).sum(dim=1)  # (B,)

    # Find batches with fewer than lower_bound slots
    need_more = slots_kept < lower_bound

    if need_more.any():
        for b in torch.where(need_more)[0]:
            # Find dropped slots
            dropped_indices = torch.where(mask[b] == 0)[0]

            if len(dropped_indices) > 0:
                # Sample additional slots needed
                n_needed = lower_bound - slots_kept[b].item()
                n_sample = min(n_needed, len(dropped_indices))

                # Random sampling
                perm = torch.randperm(len(dropped_indices))[:n_sample]
                sampled_indices = dropped_indices[perm]

                additional_mask[b, sampled_indices] = 1

    return additional_mask


class GumbelSlotSelector(nn.Module):
    """
    Gumbel-Softmax based slot selection network.

    Learns to score each slot and use Gumbel-Softmax for differentiable
    binary selection (keep/drop).
    """

    def __init__(
        self,
        slot_dim: int,
        hidden_dim: Optional[int] = None,
        low_bound: int = 1,
    ):
        """
        Args:
            slot_dim: Dimension of slot representations
            hidden_dim: Hidden layer dimension (default: slot_dim)
            low_bound: Minimum number of slots to keep
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.low_bound = low_bound

        hidden_dim = hidden_dim or slot_dim

        # Score network: slot -> binary logits (keep/drop)
        self.score_net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [drop_logit, keep_logit]
        )

    def forward(
        self,
        slots: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select slots using Gumbel-Softmax.

        Args:
            slots: Slot representations (B, K, D)
            temperature: Gumbel temperature (lower = more discrete)
            hard: If True, return hard binary mask

        Returns:
            hard_mask: Binary mask (B, K) - 1=keep, 0=drop
            soft_prob: Soft keeping probability (B, K)
        """
        B, K, D = slots.shape

        # Get selection logits
        logits = self.score_net(slots)  # (B, K, 2)

        # Gumbel-Softmax sampling
        gumbel_out = F.gumbel_softmax(
            logits, tau=temperature, hard=hard, dim=-1)

        # Extract keep decision (index 1)
        hard_mask = gumbel_out[..., 1]  # (B, K)

        # Ensure minimum slots kept
        if self.low_bound > 0:
            additional = sample_slot_lower_bound(hard_mask, self.low_bound)
            hard_mask = hard_mask + additional
            hard_mask = torch.clamp(hard_mask, 0, 1)

        # Soft probability (for logging/analysis)
        soft_prob = F.softmax(logits, dim=-1)[..., 1]  # (B, K)

        return hard_mask, soft_prob


class AdaptiveSlotAttention(BaseModule):
    """
    Adaptive Slot Attention with Gumbel-Softmax selection.

    Standard Slot Attention + learnable slot selection mechanism.
    During training, adaptively determines which slots to keep using
    Gumbel-Softmax reparameterization for differentiability.

    Features:
    - Iterative attention mechanism
    - GRU-based slot updates
    - Gumbel-Softmax slot selection
    - Temperature annealing
    - Minimum slot constraint

    Example:
        >>> slot_attn = AdaptiveSlotAttention(
        ...     num_slots=7,
        ...     slot_dim=128,
        ...     feature_dim=768,
        ...     num_iterations=3
        ... )
        >>> features = torch.randn(2, 196, 768)  # (B, N, D)
        >>> slots, attn, mask = slot_attn(features, global_step=100)
        >>> print(slots.shape)  # (2, K_kept, 128) - K_kept <= 7
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        num_iterations: int = 3,
        num_heads: int = 1,
        kvq_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        epsilon: float = 1e-8,
        use_gumbel: bool = True,
        gumbel_low_bound: int = 1,
        init_temperature: float = 1.0,
        min_temperature: float = 0.5,
        temperature_anneal_rate: float = 0.00003,
    ):
        """
        Args:
            num_slots: Maximum number of slots (K)
            slot_dim: Dimension of each slot (D_s) - called 'dim' in AdaSlot source
            feature_dim: Dimension of input features (D_f)
            num_iterations: Number of slot attention iterations - called 'iters' in source
            num_heads: Number of attention heads - called 'n_heads' in source
            kvq_dim: Attention projection dimension (default: slot_dim)
            mlp_hidden_dim: Hidden dim for slot MLP (default: slot_dim)
            epsilon: Small constant for numerical stability - called 'eps' in source
            use_gumbel: Whether to use Gumbel selection
            gumbel_low_bound: Minimum slots to keep - called 'low_bound' in source
            init_temperature: Initial Gumbel temperature
            min_temperature: Minimum temperature (annealing target)
            temperature_anneal_rate: Temperature decay rate per step
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim  # 'dim' in source
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations  # 'iters' in source
        self.num_heads = num_heads  # 'n_heads' in source
        self.epsilon = epsilon  # 'eps' in source

        # kvq_dim defaults to slot_dim (like source)
        self.kvq_dim = kvq_dim if kvq_dim is not None else slot_dim

        self.use_gumbel = use_gumbel
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.temperature_anneal_rate = temperature_anneal_rate
        self.gumbel_low_bound = gumbel_low_bound  # 'low_bound' in source

        # Validate dimensions
        if self.kvq_dim % num_heads != 0:
            raise ValueError(
                f"kvq_dim ({self.kvq_dim}) must be divisible by num_heads ({num_heads})")

        self.dims_per_head = self.kvq_dim // num_heads
        self.scale = self.dims_per_head ** -0.5

        # Slot initialization (learnable Gaussian)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Layer norms - ĐÚNG THEO SOURCE
        self.norm_input = nn.LayerNorm(feature_dim)  # Normalize FEATURES
        self.norm_slots = nn.LayerNorm(slot_dim)      # Normalize SLOTS

        # Key, Query, Value projections - ĐÚNG THEO SOURCE
        self.to_q = nn.Linear(slot_dim, self.kvq_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=False)

        # GRU for slot updates - ĐÚNG THEO SOURCE
        self.gru = nn.GRUCell(self.kvq_dim, slot_dim)

        # Optional MLP after GRU - ĐÚNG THEO SOURCE (called ff_mlp)
        mlp_hidden_dim = mlp_hidden_dim or slot_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )

        # Gumbel selector - ĐÚNG THEO SOURCE
        if use_gumbel:
            # Checkpoint uses 4x slot_dim for gumbel hidden (256 for slot_dim=64)
            gumbel_hidden_dim = slot_dim * 4
            self.gumbel_selector = GumbelSlotSelector(
                slot_dim=slot_dim,
                hidden_dim=gumbel_hidden_dim,
                low_bound=gumbel_low_bound
            )
        else:
            self.gumbel_selector = None

    def get_temperature(self, global_step: Optional[int] = None) -> float:
        """
        Compute current Gumbel temperature with annealing.

        τ(t) = max(τ_min, τ_init * exp(-r * t))

        Args:
            global_step: Current training step

        Returns:
            Current temperature
        """
        if global_step is None:
            return self.init_temperature

        temperature = self.init_temperature * math.exp(
            -self.temperature_anneal_rate * global_step
        )
        return max(self.min_temperature, temperature)

    def initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize slots from learned Gaussian distribution.

        Args:
            batch_size: Batch size
            device: Device to create slots on

        Returns:
            Initial slots (B, K, D_s)
        """
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)

        # Sample from Gaussian
        slots = mu + sigma * torch.randn_like(mu)

        return slots

    def attention_step(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single iteration of slot attention.

        Args:
            slots: Current slots (B, K, D_s)
            keys: Feature keys (B, N, H, D_h)
            values: Feature values (B, N, H, D_h)
            slot_mask: Optional mask for slots (B, K) to exclude

        Returns:
            updated_slots: Updated slots (B, K, D_s)
            attention: Attention weights (B, K, N)
        """
        B, K, D_s = slots.shape
        _, N, H, D_h = keys.shape

        slots_prev = slots

        # Normalize slots
        slots = self.norm_slots(slots)

        # Queries from slots
        queries = self.to_q(slots).view(B, K, H, D_h)  # (B, K, H, D_h)

        # Compute attention: Q @ K^T
        dots = torch.einsum('bkhd,bnhd->bkhn', queries, keys) * self.scale

        # Apply slot mask if provided (masked slots don't compete)
        if slot_mask is not None:
            mask_value = float('-inf')
            dots = dots.masked_fill(
                slot_mask.view(B, K, 1, 1).bool(),
                mask_value
            )

        # Softmax over slots (competition for features)
        attn = dots.flatten(1, 2).softmax(dim=1)  # (B, K*H, N)
        attn = attn.view(B, K, H, N)

        # Store attention before reweighting
        attn_vis = attn.mean(dim=2)  # (B, K, N) - average over heads

        # Normalize attention (weighted mean)
        attn_norm = attn + self.epsilon
        attn_norm = attn_norm / attn_norm.sum(dim=-1, keepdim=True)

        # Aggregate features: weighted sum - ĐÚNG THEO SOURCE
        updates = torch.einsum('bnhd,bkhn->bkhd', values, attn_norm)
        # Flatten to (B*K, kvq_dim)
        updates = updates.reshape(B * K, self.kvq_dim)

        # GRU update - ĐÚNG THEO SOURCE
        slots_flat = slots_prev.reshape(
            B * K, self.slot_dim)  # (B*K, slot_dim)
        # GRUCell(kvq_dim, slot_dim)
        slots_updated = self.gru(updates, slots_flat)
        slots_updated = slots_updated.reshape(B, K, self.slot_dim)

        # MLP - ĐÚNG THEO SOURCE
        slots_updated = slots_updated + self.mlp(slots_updated)

        return slots_updated, attn_vis

    def iterate_attention(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run multiple iterations of slot attention.

        Args:
            slots: Initial slots
            keys: Feature keys
            values: Feature values
            slot_mask: Optional slot mask

        Returns:
            final_slots: Refined slots
            final_attention: Final attention weights
        """
        for i in range(self.num_iterations):
            slots, attn = self.attention_step(slots, keys, values, slot_mask)

        return slots, attn

    def forward(
        self,
        features: torch.Tensor,
        global_step: Optional[int] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of adaptive slot attention.

        Args:
            features: Input features (B, N, D_f) from ViT patches
            global_step: Current training step (for temperature annealing)
            return_all: If True, return all outputs including masks

        Returns:
            Dictionary containing:
                - slots: Refined slot representations (B, K, D_s)
                - attention: Attention weights (B, K, N)
                - slot_mask: Binary mask of kept slots (B, K) if Gumbel used
                - slot_probs: Soft keeping probabilities (B, K) if Gumbel used
        """
        B, N, D_f = features.shape
        device = features.device

        # Normalize features - ĐÚNG THEO SOURCE
        features = self.norm_input(features)

        # Project to keys and values - ĐÚNG THEO SOURCE
        keys = self.to_k(features).view(
            B, N, self.num_heads, self.dims_per_head)
        values = self.to_v(features).view(
            B, N, self.num_heads, self.dims_per_head)

        # Initialize slots
        slots = self.initialize_slots(B, device)

        # Initialize slots
        slots = self.initialize_slots(B, device)

        # Run slot attention iterations
        slots, attention = self.iterate_attention(slots, keys, values)

        # Gumbel selection (if enabled)
        output = {
            "slots": slots,
            "attention": attention,
        }

        if self.use_gumbel and self.gumbel_selector is not None:
            temperature = self.get_temperature(global_step)
            hard_mask, soft_probs = self.gumbel_selector(
                slots, temperature=temperature, hard=True
            )

            output["slot_mask"] = hard_mask  # (B, K)
            output["slot_probs"] = soft_probs  # (B, K)

            # Apply mask to slots (zero out dropped slots)
            if self.training:
                output["slots"] = slots * hard_mask.unsqueeze(-1)

        return output

    def extra_repr(self) -> str:
        return (
            f"num_slots={self.num_slots}, slot_dim={self.slot_dim}, "
            f"feature_dim={self.feature_dim}, iters={self.num_iterations}, "
            f"heads={self.num_heads}, gumbel={self.use_gumbel}"
        )


# Register AdaptiveSlotAttention
# Note: Can't use multiple decorators, so we register the main class
# and create an alias manually
MODEL_REGISTRY.register("adaptive_slot_attention")(AdaptiveSlotAttention)
MODEL_REGISTRY._registry["adaslot"] = AdaptiveSlotAttention  # Alias
