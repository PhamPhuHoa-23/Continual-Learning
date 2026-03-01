"""
Base classes for Resource-Constrained Continual Learning (RCCL) system.

This module provides abstract base classes and interfaces for all major
components in the RCCL framework, ensuring consistent design and easy extension.
"""

from .base_agent import BaseAgent, BaseMetacognitiveAgent
from .types import AgentOutput

__all__ = [
    'BaseAgent',
    'BaseMetacognitiveAgent',
    'AgentOutput',
]

