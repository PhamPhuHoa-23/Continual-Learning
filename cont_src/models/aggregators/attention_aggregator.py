"""
Attention-based aggregator with per-agent keys.

Implements block-diagonal attention mechanism from the paper.
Each agent has its own attention key, frozen when the agent freezes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from cont_src.core.base_module import BaseAggregator
from cont_src.core.registry import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("attention")
class AttentionAggregator(BaseAggregator):
    """
    Block-diagonal attention aggregator.

    Each agent i owns an attention key w_i ∈ R^{d_h}.
    Aggregation: H = Σ_k α_k h_k, where α_k = softmax(w_{σ(k)}^T h_k)

    Output dimension d_h is constant regardless of number of agents.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        use_per_agent_keys: bool = True,
        freeze_keys_with_agents: bool = True,
        **kwargs
    ):
        """
        Initialize attention aggregator.

        Args:
            hidden_dim: Hidden dimension (d_h in paper)
            use_per_agent_keys: Use separate key per agent (block-diagonal)
            freeze_keys_with_agents: Freeze agent keys when agents freeze
        """
        super().__init__(config={
            "hidden_dim": hidden_dim,
            "use_per_agent_keys": use_per_agent_keys,
            "freeze_keys_with_agents": freeze_keys_with_agents,
        })

        self.hidden_dim = hidden_dim
        self.use_per_agent_keys = use_per_agent_keys
        self.freeze_keys_with_agents = freeze_keys_with_agents

        # Per-agent keys: agent_id -> key vector
        self.agent_keys = nn.ParameterDict()

        # Track frozen agents
        self.frozen_agents = set()

    def register_agent(self, agent_id: int):
        """
        Register a new agent and create its attention key.

        Args:
            agent_id: Agent ID
        """
        key_name = str(agent_id)
        if key_name not in self.agent_keys:
            # Initialize key
            key = nn.Parameter(torch.randn(
                self.hidden_dim) / (self.hidden_dim ** 0.5))
            self.agent_keys[key_name] = key

    def freeze_agent_key(self, agent_id: int):
        """
        Freeze attention key for an agent.

        Args:
            agent_id: Agent ID to freeze
        """
        key_name = str(agent_id)
        if key_name in self.agent_keys:
            self.agent_keys[key_name].requires_grad = False
            self.frozen_agents.add(agent_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_assignments: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate hidden states using attention mechanism.

        Args:
            hidden_states: Hidden states from agents, shape (B, K, D_h)
            agent_assignments: Agent ID for each slot, shape (B, K)
            return_attention: If True, return attention weights

        Returns:
            Dict with:
                - aggregated: Aggregated representation, shape (B, D_h)
                - attention_weights: Attention weights (if return_attention=True)
        """
        B, K, D_h = hidden_states.shape
        device = hidden_states.device

        if not self.use_per_agent_keys:
            # Simple average pooling
            aggregated = hidden_states.mean(dim=1)  # (B, D_h)
            return {"aggregated": aggregated}

        # Per-agent attention
        attention_logits = torch.zeros(B, K, device=device)

        for b in range(B):
            for k in range(K):
                agent_id = int(agent_assignments[b, k].item())
                key_name = str(agent_id)

                if key_name in self.agent_keys:
                    # w_i^T h_k
                    key = self.agent_keys[key_name]
                    logit = torch.dot(key, hidden_states[b, k])
                    attention_logits[b, k] = logit
                else:
                    # No key for this agent - use zero logit
                    attention_logits[b, k] = 0.0

        # Softmax over slots
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, K)

        # Weighted sum: H = Σ_k α_k h_k
        aggregated = torch.einsum(
            "bk,bkd->bd", attention_weights, hidden_states)  # (B, D_h)

        output = {"aggregated": aggregated}

        if return_attention:
            output["attention_weights"] = attention_weights

        return output


@AGGREGATOR_REGISTRY.register("average")
class AverageAggregator(BaseAggregator):
    """Simple average pooling aggregator."""

    def __init__(self, hidden_dim: int = 256, **kwargs):
        super().__init__(config={"hidden_dim": hidden_dim})

    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Average pool hidden states."""
        aggregated = hidden_states.mean(dim=1)
        return {"aggregated": aggregated}
