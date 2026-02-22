"""
Perceptual Grouping with Gumbel Slot Selection (AdaSlot).

Mirrors: ocl.perceptual_grouping.SlotAttentionGumbelV1 and SlotAttentionGroupingGumbelV1
Checkpoint key prefix: models.perceptual_grouping.slot_attention.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def sample_slot_lower_bound(A: torch.Tensor, lower_bound: int = 1) -> torch.Tensor:
    """
    Ensure at least `lower_bound` slots are kept.

    Args:
        A: (B, K) binary keep decisions.
        lower_bound: Minimum slots to keep.

    Returns:
        B: (B, K) additional slots to activate.
    """
    B_out = torch.zeros_like(A, device=A.device)
    batch_slot_leftnum = (A != 0).sum(-1)
    lesser_column_idx = torch.nonzero(batch_slot_leftnum < lower_bound).reshape(-1)
    for j in lesser_column_idx:
        left_slot_mask = A[j]
        sample_slot_zero_idx = torch.nonzero(left_slot_mask == 0).reshape(-1)
        sampled_indices = torch.randperm(sample_slot_zero_idx.size(0))[
            : lower_bound - batch_slot_leftnum[j]
        ]
        sampled_elements = sample_slot_zero_idx[sampled_indices]
        B_out[j][sampled_elements] += 1
    return B_out


class SlotAttentionGumbelV1(nn.Module):
    """
    Slot Attention with Gumbel Selection Module.

    After standard iterative slot attention, applies a Gumbel-Softmax based
    selection network to adaptively choose which slots to keep.
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        single_gumbel_score_network: Optional[nn.Module] = None,
        low_bound: int = 0,
        temporature_function=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError(
                "Key, value, query dimensions must be divisible by number of heads."
            )
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head ** -0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp
        self.single_gumbel_score_network = single_gumbel_score_network
        self.low_bound = low_bound
        self.temporature_function = temporature_function

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(
                masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf")
            )

        attn = dots.flatten(1, 2).softmax(dim=1)
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(
            updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim)
        )
        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self,
        inputs: torch.Tensor,
        conditioning: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        global_step=None,
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        # Gumbel selection
        _, k_slots, _ = conditioning.shape
        prev_decision = torch.ones(b, k_slots, dtype=slots.dtype, device=slots.device)

        slots_keep_prob = self.single_gumbel_score_network(slots)
        if global_step is None:
            tau = 1
        else:
            tau = (
                self.temporature_function(global_step)
                if self.temporature_function is not None
                else 1
            )
        current_keep_decision = F.gumbel_softmax(slots_keep_prob, hard=True, tau=tau)[
            ..., 1
        ]
        if self.low_bound > 0:
            current_keep_decision = current_keep_decision + sample_slot_lower_bound(
                current_keep_decision, self.low_bound
            )
        hard_keep_decision = current_keep_decision * prev_decision
        slots_keep_prob = F.softmax(slots_keep_prob, dim=-1)[..., 1]

        return slots, attn, slots_keep_prob, hard_keep_decision


class SlotAttentionGroupingGumbelV1(nn.Module):
    """
    Perceptual grouping module that wraps SlotAttentionGumbelV1
    with positional embedding.

    Checkpoint key prefix: models.perceptual_grouping.*
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        single_gumbel_score_network: Optional[nn.Module] = None,
        low_bound: int = 0,
        temporature_function=None,
    ):
        super().__init__()

        self._object_dim = object_dim
        self.slot_attention = SlotAttentionGumbelV1(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            single_gumbel_score_network=single_gumbel_score_network,
            low_bound=low_bound,
            temporature_function=temporature_function,
        )

        self.positional_embedding = positional_embedding

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        conditioning: torch.Tensor,
        slot_masks: Optional[torch.Tensor] = None,
        global_step=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, feature_dim) extracted features.
            positions: (N, 2) normalized positions.
            conditioning: (B, num_slots, object_dim) initial slot embeddings.
            slot_masks: Optional mask for slots.
            global_step: Optional global step for temperature scheduling.

        Returns:
            Dict with 'objects', 'is_empty', 'feature_attributions',
            'slots_keep_prob', 'hard_keep_decision'.
        """
        if self.positional_embedding:
            features = self.positional_embedding(features, positions)

        slots, attn, slots_keep_prob, hard_keep_decision = self.slot_attention(
            features, conditioning, slot_masks, global_step=global_step
        )

        return {
            "objects": slots,
            "is_empty": slot_masks,
            "feature_attributions": attn,
            "slots_keep_prob": slots_keep_prob,
            "hard_keep_decision": hard_keep_decision,
        }
