"""
Concat Aggregator — fixed-size slot concatenation for downstream classifiers.

═══════════════════════════════════════════════════════════════════════════════
TWO AGGREGATION STRATEGIES (both implemented here)
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  AttentionSumAggregator  (see attention_aggregator.py)                      │
│                                                                             │
│  H = Σ_k α_k h_k  ∈ R^{d_h}   ← constant dim, SLDA-compatible             │
│                                                                             │
│  • Output dim = d_h, FIXED regardless of agent count                       │
│  • Old-class H identical across tasks (frozen keys → zero forgetting)      │
│  • Gradient flows through α_k and h_k simultaneously                       │
│  • Use with: SLDA, Mahalanobis classifier, any fixed-dim head              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  ConcatAggregator  (THIS FILE)                                              │
│                                                                             │
│  H_cat = [h_1 ‖ h_2 ‖ ... ‖ h_{K_max}]  ∈ R^{K_max × d_h}               │
│  (unassigned / dropped slots padded with zeros)                             │
│                                                                             │
│  • Preserves slot-level structure → richer for learned classifiers          │
│  • Output dim = K_max × d_h  (fixed if K_max fixed at init)                │
│  • Incompatible with analytic SLDA (dim changes if K_max grows)             │
│  • Use with: collaborator's MLP/Transformer classifier trained per-task    │
└─────────────────────────────────────────────────────────────────────────────┘

Trade-off summary:
    ┌──────────────┬──────────────────┬────────────────────────────────────┐
    │              │ attention_sum    │ concat                             │
    ├──────────────┼──────────────────┼────────────────────────────────────┤
    │ Output dim   │ fixed d_h        │ fixed K_max × d_h                  │
    │ SLDA compat  │ ✅               │ ❌ (requires classifier retraining) │
    │ Slot order   │ permutation-inv  │ sorted by agent_id (deterministic)  │
    │ Zero forget  │ ✅ (frozen keys) │ depends on downstream classifier    │
    │ Expressivity │ moderate         │ higher (slot interactions possible) │
    └──────────────┴──────────────────┴────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.core.base_module import BaseAggregator
from cont_src.core.registry import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("concat")
class ConcatAggregator(BaseAggregator):
    """
    Fixed-slot concatenation aggregator.

    Pads unassigned slots with zeros, sorts by agent_id, concatenates.

    Output shape: (B, K_max, d_h)  — ready for Transformer, MLP, or flatten.

    Example::

        agg = ConcatAggregator(hidden_dim=64, max_slots=11)
        H_cat = agg(hidden_states, agent_assignments)["aggregated"]
        # shape: (B, 11, 64)
        # flatten → (B, 704) for MLP classifier

    Unassigned slots (agent_id = -1) are placed at the end and zero-padded.
    This ensures deterministic slot ordering across batches.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_slots: int = 11,
        pad_value: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            hidden_dim: Hidden dimension d_h (must match agent output_dim).
            max_slots:  K_max — maximum number of slots per image.
                        Output dim = max_slots × hidden_dim.
            pad_value:  Fill value for unassigned / missing slots.
        """
        super().__init__(config={
            "hidden_dim": hidden_dim,
            "max_slots": max_slots,
            "pad_value": pad_value,
        })
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots
        self.pad_value = pad_value
        self.output_dim = max_slots * hidden_dim   # for classifier input sizing

    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_assignments: torch.Tensor,
        return_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate agent hidden states into fixed-size representation.

        Args:
            hidden_states:     (B, K, d_h) — per-slot hidden vectors from agents.
            agent_assignments: (B, K)      — agent ID for each slot (-1 = unassigned).
            return_mask:       If True, return boolean mask of active slots.

        Returns:
            Dict::
                "aggregated": (B, K_max, d_h) — padded, sorted slot tensor
                "flat":       (B, K_max * d_h) — flattened version
                "mask":       (B, K_max) bool  — True where slot is active (if return_mask)
        """
        B, K, D = hidden_states.shape
        device = hidden_states.device
        K_max = self.max_slots

        # Output buffer — fills pad_value for missing/unassigned slots
        output = torch.full(
            (B, K_max, D), fill_value=self.pad_value,
            dtype=hidden_states.dtype, device=device
        )
        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)

        for b in range(B):
            # Separate assigned vs unassigned
            assigned_mask = agent_assignments[b] >= 0          # (K,)
            assigned_h = hidden_states[b][assigned_mask]    # (n_assigned, D)
            n_assigned = assigned_h.shape[0]

            # Sort assigned slots by agent_id for deterministic ordering
            assigned_ids = agent_assignments[b][assigned_mask]
            sort_order = assigned_ids.argsort()
            assigned_h = assigned_h[sort_order]

            # Fill into output (up to K_max)
            n_fill = min(n_assigned, K_max)
            output[b, :n_fill] = assigned_h[:n_fill]
            mask[b, :n_fill] = True

        out = {
            "aggregated": output,                            # (B, K_max, D)
            "flat":       output.flatten(start_dim=1),      # (B, K_max * D)
        }
        if return_mask:
            out["mask"] = mask
        return out


@AGGREGATOR_REGISTRY.register("concat_pool")
class ConcatPoolAggregator(ConcatAggregator):
    """
    ConcatAggregator with optional pooling head.

    Concatenates slots then applies mean/max/attention pooling to produce
    a fixed (B, d_h) vector — bridge between concat expressivity and
    attention_sum's fixed output dim, useful for comparing both strategies.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_slots: int = 11,
        pool_mode: str = "mean",   # "mean" | "max" | "attention"
        **kwargs,
    ):
        """
        Args:
            pool_mode: How to pool K_max slots → 1 vector.
                       "mean":      average over slots (ignores mask padding).
                       "max":       max over slots.
                       "attention": learnable attention over slots.
        """
        super().__init__(hidden_dim=hidden_dim, max_slots=max_slots, **kwargs)
        self.pool_mode = pool_mode

        if pool_mode == "attention":
            self.pool_key = nn.Parameter(
                torch.randn(hidden_dim) / hidden_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_assignments: torch.Tensor,
        return_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        base = super().forward(hidden_states, agent_assignments, return_mask=True)
        slots = base["aggregated"]   # (B, K_max, D)
        mask = base["mask"]         # (B, K_max) bool

        if self.pool_mode == "mean":
            # Mean over active slots only
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)
            pooled = (slots * mask.unsqueeze(-1).float()
                      ).sum(dim=1) / denom  # (B, D)

        elif self.pool_mode == "max":
            # Max, masked out positions set to -inf
            fill = slots.clone()
            fill[~mask] = float("-inf")
            pooled = fill.max(dim=1).values   # (B, D)

        elif self.pool_mode == "attention":
            # Learned attention over active slots
            logits = torch.einsum("bkd,d->bk", slots,
                                  self.pool_key)   # (B, K_max)
            logits[~mask] = float("-inf")
            # (B, K_max)
            attn = F.softmax(logits, dim=1)
            pooled = torch.einsum("bk,bkd->bd", attn, slots)           # (B, D)

        else:
            raise ValueError(f"Unknown pool_mode '{self.pool_mode}'")

        out = {**base, "aggregated": pooled}   # override with (B, D) pooled
        if not return_mask:
            out.pop("mask", None)
        return out
