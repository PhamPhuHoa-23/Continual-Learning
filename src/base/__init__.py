"""
Base classes for Resource-Constrained Continual Learning (RCCL) system.

This module provides abstract base classes and interfaces for all major
components in the RCCL framework, ensuring consistent design and easy extension.
"""

from .base_agent import BaseAgent, BaseMetacognitiveAgent
from .base_uncertainty import BaseUncertaintyEstimator
from .base_bidding import BaseBiddingStrategy
from .base_metric import BaseMetric
from .types import AgentOutput, BidResult, UncertaintyOutput, EVCResult

__all__ = [
    'BaseAgent',
    'BaseMetacognitiveAgent',
    'BaseUncertaintyEstimator',
    'BaseBiddingStrategy',
    'BaseMetric',
    'AgentOutput',
    'BidResult',
    'UncertaintyOutput',
    'EVCResult',
]

