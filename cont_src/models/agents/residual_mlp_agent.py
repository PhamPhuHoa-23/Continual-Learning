"""
Residual MLP Agent with optional reconstruction decoder.

═══════════════════════════════════════════════════════════════════════════════
DESIGN
═══════════════════════════════════════════════════════════════════════════════

Each agent maps one slot → one hidden vector:

    s_k  ∈ R^{D_s}  →  [ResidualMLP]  →  h_k  ∈ R^{d_h}

The residual structure prevents gradient vanishing when agents are deep,
and is better suited than plain MLP for learning incremental feature
transformations (important when fine-tuning on new tasks).

Residual block:
    h' = h + W_2 · GELU(LN(W_1 · LN(h)))

Agent decoder (for L_agent anti-collapse loss):
    h_k  →  [small MLP]  →  ŝ_k  ≈  s_k

    L_agent = (1/K) Σ_k ||ŝ_k - s_k||²

    This loss forces h_k to encode enough information to reconstruct s_k,
    preventing the agent from collapsing to a trivial constant mapping.

═══════════════════════════════════════════════════════════════════════════════
TRAINING PHASES (for L_agent)
═══════════════════════════════════════════════════════════════════════════════

Phase A (warm-up):   L = γ · L_agent only
    → establish meaningful h_k before label loss is applied

Phase B (main):      L = γ · L_agent + α · L_p + β · L_SupCon
    → label signal shapes H = aggregated(h_k) into discriminative space

Phase C (freeze):    agent.freeze()  →  only SLDA / new agents train

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from cont_src.core.base_module import BaseAgent
from cont_src.core.registry import AGENT_REGISTRY


# ── Residual block ────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """
    Pre-LN residual block:
        h' = h + W_2 · GELU(LN(W_1 · h))

    Pre-LN (LayerNorm before linear) stabilises training for small agents
    that receive gradients from multiple loss terms.
    """

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ── Main agent ────────────────────────────────────────────────────────────────

@AGENT_REGISTRY.register("residual_mlp")
class ResidualMLPAgent(BaseAgent):
    """
    Residual MLP agent:  s_k → h_k.

    Architecture::

        Input projection:  D_s → d_h
        N × ResidualBlock(d_h)
        Output LN

    With decoder for L_agent::

        Decoder: d_h → D_s
        (2-layer MLP, lighter than encoder)
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        num_blocks: int = 2,
        expansion: int = 2,
        dropout: float = 0.0,
        use_decoder: bool = True,
        **kwargs,
    ):
        """
        Args:
            input_dim:   Slot dimension D_s.
            output_dim:  Hidden dimension d_h  (= D_s by default for easy aggregation).
            num_blocks:  Number of residual blocks.
            expansion:   Feed-forward expansion ratio inside each block.
            dropout:     Dropout inside blocks.
            use_decoder: Attach lightweight decoder for L_agent loss.
        """
        super().__init__(config={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_blocks": num_blocks,
            "expansion": expansion,
            "dropout": dropout,
            "use_decoder": use_decoder,
        })

        # ── Encoder (slot → hidden) ───────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.blocks = nn.Sequential(
            *[_ResidualBlock(output_dim, expansion, dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(output_dim)

        # ── Decoder (hidden → slot, for L_agent) ─────────────────────────────
        self.decoder: Optional[nn.Module] = None
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, input_dim),
            )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Map slots to hidden representations.

        Args:
            slots: Shape (B, K, D_s)  or  (B, D_s)  or  (D_s,).

        Returns:
            Dict::
                "hidden":        h_k  — shape mirrors input but last dim = d_h
                "reconstructed": ŝ_k  — same shape as input (only if training + decoder exists)
        """
        x = self.input_proj(slots)
        x = self.blocks(x)
        h = self.output_norm(x)

        out: Dict[str, torch.Tensor] = {"hidden": h}

        if self.training and self.decoder is not None:
            out["reconstructed"] = self.decoder(h)

        return out

    # ── L_agent loss helper ───────────────────────────────────────────────────

    def reconstruction_loss(
        self,
        slots: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute L_agent = mean_k ||decoder(h_k) - s_k||²

        Args:
            slots:  Original slots, shape (B, K, D_s) or (N, D_s).
            hidden: Pre-computed h_k; if None, runs forward first.

        Returns:
            Scalar loss tensor.
        """
        if self.decoder is None:
            raise RuntimeError(
                "Decoder not attached. Initialise with use_decoder=True.")

        if hidden is None:
            out = self.forward(slots)
            hidden = out["hidden"]

        reconstructed = self.decoder(hidden)
        return nn.functional.mse_loss(reconstructed, slots, reduction="mean")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def freeze(self):
        """Freeze all parameters (called after task training completes)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self):
        """Unfreeze for continued training."""
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()
