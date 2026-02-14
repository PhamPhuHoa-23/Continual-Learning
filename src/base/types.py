"""
Type definitions and data structures for RCCL system.

This module defines TypedDict and dataclass types used throughout the RCCL
framework to ensure type safety and clear interfaces.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from typing_extensions import TypedDict


@dataclass
class AgentOutput:
    """Output from an agent's forward pass.
    
    Attributes:
        logits: Prediction logits (B, num_classes)
        features: Intermediate features (B, feature_dim)
        uncertainty: Uncertainty estimate (B,)
        metadata: Additional information (slots, attention weights, etc.)
    """
    logits: torch.Tensor
    features: torch.Tensor
    uncertainty: torch.Tensor
    metadata: Dict[str, torch.Tensor]


@dataclass
class UncertaintyOutput:
    """Output from uncertainty estimation.
    
    Attributes:
        epistemic: Epistemic (model) uncertainty (B,)
        aleatoric: Aleatoric (data) uncertainty (B,)
        total: Total uncertainty (B,)
        confidence: Confidence score (B,)
        evidence: Evidence for each class (B, C) - for EDL
    """
    epistemic: torch.Tensor
    aleatoric: torch.Tensor
    total: torch.Tensor
    confidence: torch.Tensor
    evidence: Optional[torch.Tensor] = None


@dataclass
class BidResult:
    """Result of bidding process.
    
    Attributes:
        agent_id: ID of the agent
        bid_value: Bid value (higher = more confident)
        expected_accuracy: Predicted accuracy [0, 1]
        expected_cost: Predicted computational cost (FLOPs)
        evc: Expected Value of Computation
        metadata: Additional bidding information
    """
    agent_id: int
    bid_value: float
    expected_accuracy: float
    expected_cost: float
    evc: float
    metadata: Dict[str, any]


@dataclass
class EVCResult:
    """Expected Value of Computation calculation result.
    
    Attributes:
        utility: Expected utility (reward)
        compute_cost: Computational cost (FLOPs)
        latency_cost: Latency cost (ms)
        opportunity_cost: Opportunity cost
        total_evc: Total EVC = utility - costs
    """
    utility: float
    compute_cost: float
    latency_cost: float
    opportunity_cost: float
    total_evc: float


class MetricResult(TypedDict):
    """Result from a metric computation.
    
    Keys:
        name: Metric name
        value: Metric value
        better: 'higher' or 'lower'
        metadata: Additional metric information
    """
    name: str
    value: float
    better: str  # 'higher' or 'lower'
    metadata: Dict[str, any]


@dataclass
class PrototypeInfo:
    """Information about a prototype.
    
    Attributes:
        prototype_vector: The prototype embedding (D,)
        class_id: Associated class ID
        count: Number of samples contributing to this prototype
        confidence: Confidence score of this prototype
        last_updated: Step when last updated
    """
    prototype_vector: torch.Tensor
    class_id: int
    count: int
    confidence: float
    last_updated: int


@dataclass
class ResourceBudget:
    """Resource budget for computation.
    
    Attributes:
        total_flops: Total FLOPs budget
        remaining_flops: Remaining FLOPs
        total_time_ms: Total time budget (ms)
        remaining_time_ms: Remaining time (ms)
        total_memory_mb: Total memory budget (MB)
        remaining_memory_mb: Remaining memory (MB)
    """
    total_flops: float
    remaining_flops: float
    total_time_ms: float
    remaining_time_ms: float
    total_memory_mb: float
    remaining_memory_mb: float
    
    def can_afford(self, cost_flops: float, cost_time: float = 0, 
                   cost_memory: float = 0) -> bool:
        """Check if we can afford the given costs."""
        return (self.remaining_flops >= cost_flops and
                self.remaining_time_ms >= cost_time and
                self.remaining_memory_mb >= cost_memory)
    
    def spend(self, cost_flops: float, cost_time: float = 0,
              cost_memory: float = 0):
        """Spend resources."""
        self.remaining_flops -= cost_flops
        self.remaining_time_ms -= cost_time
        self.remaining_memory_mb -= cost_memory

