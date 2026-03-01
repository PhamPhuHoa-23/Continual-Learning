"""Agent models."""

from cont_src.models.agents.mlp_agent import MLPAgent, MLPAgentWithDecoder, IdentityAgent
from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent

__all__ = ["MLPAgent", "MLPAgentWithDecoder",
           "IdentityAgent", "ResidualMLPAgent"]
