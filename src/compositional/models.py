"""
Compositional Sub-Concept Routing — Agent and Aggregator models.

AgentNet:
    Maps slot s_k → hidden representation h_k ∈ R^{d_h}.
    Uses ResidualMLP backbone (from atomic_agent.py) for richer representations.
    Includes a lightweight decoder for L_agent (anti-collapse loss).

BlockDiagonalAggregator:
    Computes H = Σ_k α_k h_k with per-agent attention keys w_i (constant d_h output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.slot_multi_agent.atomic_agent import ResidualBlock


# ─── Agent ────────────────────────────────────────────────────────────────────

class AgentNet(nn.Module):
    """
    Agent a_i: maps slot s_k → hidden representation h_k ∈ R^{d_h}.

    Encoder uses a ResidualMLP backbone for better representations.
    Decoder is lightweight (2-layer MLP) for L_agent anti-collapse loss.

    Args:
        slot_dim: Input slot dimension (e.g. 64 for AdaSlot)
        d_h:      Output hidden-label dimension
        hidden_dim: Internal width of residual blocks
        num_blocks: Number of residual blocks
    """

    def __init__(
        self,
        slot_dim: int = 64,
        d_h: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.d_h = d_h

        # --- Encoder: linear projection + residual blocks + output projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, d_h)

        # --- Decoder: lightweight 2-layer MLP (for L_agent) ---
        self.decoder = nn.Sequential(
            nn.Linear(d_h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Encode slot to hidden representation.

        Args:
            s: (B, slot_dim)
        Returns:
            h: (B, d_h)
        """
        x = self.input_proj(s)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation back to slot space (for L_agent).

        Args:
            h: (B, d_h)
        Returns:
            s_hat: (B, slot_dim)
        """
        return self.decoder(h)


# ─── Aggregator ───────────────────────────────────────────────────────────────

class BlockDiagonalAggregator(nn.Module):
    """
    Constant-dimension aggregator (paper Section 1, eq. 1–2).

    H = Σ_k α_k h_k,   α_k = softmax_k( w_{σ(k)}^T h_k )

    Each agent i owns a key w_i ∈ R^{d_h} frozen when agent i freezes.
    Output dimension d_h is constant regardless of how many agents exist.

    Args:
        d_h: Hidden-label dimension
    """

    def __init__(self, d_h: int = 128):
        super().__init__()
        self.d_h = d_h
        self.keys = nn.ParameterDict()   # {agent_id: Parameter(d_h,)}

    def add_agent_key(self, agent_id: str, device=None) -> None:
        """Register a new per-agent attention key (initialised small)."""
        dev = torch.device(device) if device is not None else torch.device('cpu')
        key = nn.Parameter(torch.empty(self.d_h, device=dev))
        nn.init.normal_(key, std=0.01)
        self.keys[agent_id] = key

    def freeze_key(self, agent_id: str) -> None:
        """Freeze key when its agent is frozen."""
        if agent_id in self.keys:
            self.keys[agent_id].requires_grad_(False)

    def forward(
        self,
        h: torch.Tensor,
        sigma: list,          # list[list[str]] — agent_id per slot per sample
    ) -> torch.Tensor:
        """
        Args:
            h:     (B, K, d_h)  — encoded slot representations
            sigma: (B, K)       — agent assignment strings ('unassigned' or agent_id)
        Returns:
            H: (B, d_h)
        """
        B, K, _ = h.shape

        # Build logits via stacking (NOT in-place) so grads flow through self.keys
        neg_inf = torch.tensor(-1e9, device=h.device)
        logit_rows = []
        for b in range(B):
            logit_cols = []
            for k in range(K):
                aid = sigma[b][k]
                if aid != "unassigned" and aid in self.keys:
                    logit_cols.append(torch.dot(self.keys[aid], h[b, k]))
                else:
                    logit_cols.append(neg_inf.expand([]))
            logit_rows.append(torch.stack(logit_cols))     # (K,)
        logits = torch.stack(logit_rows)                   # (B, K)

        alpha = F.softmax(logits, dim=1).unsqueeze(2)      # (B, K, 1)
        return (alpha * h).sum(dim=1)                      # (B, d_h)
