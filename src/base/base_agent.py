"""
Base Agent classes for RCCL system.

Provides abstract base classes for all agent types with clear interfaces
for forward pass, uncertainty estimation, and resource accounting.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .types import AgentOutput, UncertaintyOutput, BidResult, EVCResult, ResourceBudget


class BaseAgent(ABC, nn.Module):
    """Abstract base class for all agents in RCCL system.
    
    All agents must implement:
    - forward(): Process input and return predictions
    - estimate_cost(): Estimate computational cost before execution
    - get_flops(): Get actual FLOPs count
    
    Attributes:
        agent_id: Unique identifier for this agent
        num_classes: Number of output classes
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        device: torch device
    """
    
    def __init__(
        self,
        agent_id: int,
        num_classes: int,
        input_dim: int,
        hidden_dim: int = 256,
        device: str = 'cpu'
    ):
        super().__init__()
        self.agent_id = agent_id
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # Performance tracking
        self.register_buffer('win_count', torch.tensor(0))
        self.register_buffer('total_bids', torch.tensor(0))
        self.register_buffer('total_correct', torch.tensor(0))
        self.register_buffer('total_processed', torch.tensor(0))
        
        # Resource tracking
        self.last_flops = 0
        self.total_flops = 0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> AgentOutput:
        """Forward pass through agent.
        
        Args:
            x: Input tensor (B, D)
            
        Returns:
            AgentOutput containing logits, features, uncertainty, and metadata
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, x: torch.Tensor) -> float:
        """Estimate computational cost (FLOPs) for processing x.
        
        This should be a FAST approximation (meta-computation).
        Should NOT actually run the full forward pass.
        
        Args:
            x: Input tensor (B, D)
            
        Returns:
            Estimated FLOPs
        """
        pass
    
    @abstractmethod
    def get_flops(self) -> float:
        """Get actual FLOPs consumed in last forward pass.
        
        Returns:
            FLOPs count
        """
        pass
    
    def update_stats(self, won_bid: bool, correct: Optional[torch.Tensor] = None):
        """Update agent's performance statistics.
        
        Args:
            won_bid: Whether this agent won the bidding
            correct: Boolean tensor of correct predictions (B,)
        """
        self.total_bids += 1
        
        if won_bid:
            self.win_count += 1
            if correct is not None:
                self.total_correct += correct.sum()
                self.total_processed += correct.numel()
    
    def get_win_rate(self) -> float:
        """Get win rate (wins / total_bids)."""
        if self.total_bids == 0:
            return 0.0
        return (self.win_count / self.total_bids).item()
    
    def get_accuracy(self) -> float:
        """Get accuracy on processed samples."""
        if self.total_processed == 0:
            return 0.0
        return (self.total_correct / self.total_processed).item()
    
    def to(self, device):
        """Move agent to device."""
        super().to(device)
        self.device = device
        return self


class BaseMetacognitiveAgent(BaseAgent):
    """Base class for agents with metacognitive capabilities.
    
    Extends BaseAgent with:
    - Uncertainty estimation
    - Bid computation based on uncertainty
    - EVC (Expected Value of Computation) calculation
    
    Attributes:
        uncertainty_estimator: Module for uncertainty quantification
        bidding_strategy: Strategy for computing bids
    """
    
    def __init__(
        self,
        agent_id: int,
        num_classes: int,
        input_dim: int,
        hidden_dim: int = 256,
        device: str = 'cpu',
        uncertainty_estimator: Optional[nn.Module] = None,
        bidding_strategy: str = 'inverse_uncertainty'
    ):
        super().__init__(agent_id, num_classes, input_dim, hidden_dim, device)
        
        self.uncertainty_estimator = uncertainty_estimator
        self.bidding_strategy = bidding_strategy
    
    @abstractmethod
    def estimate_uncertainty(self, x: torch.Tensor) -> UncertaintyOutput:
        """Estimate uncertainty for input x.
        
        Args:
            x: Input tensor (B, D)
            
        Returns:
            UncertaintyOutput with epistemic, aleatoric, and total uncertainty
        """
        pass
    
    def compute_bid(
        self,
        x: torch.Tensor,
        task_utility: float = 1.0,
        budget: Optional[ResourceBudget] = None
    ) -> BidResult:
        """Compute bid for processing input x.
        
        Args:
            x: Input tensor (B, D)
            task_utility: Utility value of successfully completing task
            budget: Current resource budget
            
        Returns:
            BidResult with bid value, expected accuracy, cost, and EVC
        """
        # Fast uncertainty estimation (pre-execution)
        uncertainty = self.estimate_uncertainty(x)
        
        # Cost estimation (without full forward pass)
        estimated_cost = self.estimate_cost(x)
        
        # Check budget constraint
        if budget is not None and not budget.can_afford(estimated_cost):
            # Cannot afford - submit zero bid
            return BidResult(
                agent_id=self.agent_id,
                bid_value=0.0,
                expected_accuracy=0.0,
                expected_cost=estimated_cost,
                evc=-float('inf'),
                metadata={'reason': 'insufficient_budget'}
            )
        
        # Expected accuracy (inverse of uncertainty)
        expected_accuracy = 1.0 - uncertainty.total.mean().item()
        
        # Compute EVC
        evc_result = self.compute_evc(
            expected_accuracy=expected_accuracy,
            cost=estimated_cost,
            utility=task_utility
        )
        
        # Compute bid value based on strategy
        bid_value = self._compute_bid_value(
            uncertainty=uncertainty,
            evc=evc_result.total_evc,
            expected_accuracy=expected_accuracy
        )
        
        return BidResult(
            agent_id=self.agent_id,
            bid_value=bid_value,
            expected_accuracy=expected_accuracy,
            expected_cost=estimated_cost,
            evc=evc_result.total_evc,
            metadata={
                'uncertainty': uncertainty.total.mean().item(),
                'strategy': self.bidding_strategy
            }
        )
    
    def compute_evc(
        self,
        expected_accuracy: float,
        cost: float,
        utility: float = 1.0
    ) -> EVCResult:
        """Compute Expected Value of Computation (EVC).
        
        EVC = P(Success) * Utility - [Compute_Cost + Latency_Cost + Opportunity_Cost]
        
        Args:
            expected_accuracy: Probability of success [0, 1]
            cost: Computational cost (FLOPs)
            utility: Reward for successful completion
            
        Returns:
            EVCResult with detailed cost breakdown
        """
        # Normalize cost (FLOPs) to [0, 1] range for comparison
        # Assuming 1 GFLOPs = 1 unit cost
        compute_cost_normalized = cost / 1e9
        
        # Latency cost (proportional to FLOPs)
        # Assume 1 GFLOP = 1ms on modern hardware
        latency_cost = cost / 1e9
        
        # Opportunity cost (what else could we do with these resources?)
        # For now, set to 10% of compute cost
        opportunity_cost = 0.1 * compute_cost_normalized
        
        # Total EVC
        expected_utility = expected_accuracy * utility
        total_costs = compute_cost_normalized + latency_cost + opportunity_cost
        total_evc = expected_utility - total_costs
        
        return EVCResult(
            utility=expected_utility,
            compute_cost=compute_cost_normalized,
            latency_cost=latency_cost,
            opportunity_cost=opportunity_cost,
            total_evc=total_evc
        )
    
    def _compute_bid_value(
        self,
        uncertainty: UncertaintyOutput,
        evc: float,
        expected_accuracy: float
    ) -> float:
        """Compute bid value based on bidding strategy.
        
        Args:
            uncertainty: Uncertainty estimates
            evc: Expected Value of Computation
            expected_accuracy: Expected accuracy
            
        Returns:
            Bid value (higher = more confident)
        """
        if self.bidding_strategy == 'inverse_uncertainty':
            # Bid = 1 / (uncertainty + epsilon)
            return 1.0 / (uncertainty.total.mean().item() + 1e-6)
        
        elif self.bidding_strategy == 'confidence':
            # Bid = confidence score
            return uncertainty.confidence.mean().item()
        
        elif self.bidding_strategy == 'evc':
            # Bid = EVC directly
            return max(0.0, evc)  # Ensure non-negative
        
        elif self.bidding_strategy == 'historical':
            # Bid = accuracy * (1 / uncertainty)
            agent_accuracy = self.get_accuracy()
            if agent_accuracy == 0:
                agent_accuracy = 0.5  # Default for new agent
            return agent_accuracy / (uncertainty.total.mean().item() + 1e-6)
        
        elif self.bidding_strategy == 'hybrid':
            # Bid = EVC * historical_accuracy
            agent_accuracy = self.get_accuracy()
            if agent_accuracy == 0:
                agent_accuracy = 0.5
            return max(0.0, evc * agent_accuracy)
        
        else:
            raise ValueError(f"Unknown bidding strategy: {self.bidding_strategy}")


class AgentEnsemble(nn.Module):
    """Ensemble of multiple agents.
    
    Used for Bootstrap-Distill and uncertainty estimation.
    
    Attributes:
        agents: List of BaseAgent instances
        aggregation: How to aggregate predictions ('mean', 'vote', 'weighted')
    """
    
    def __init__(
        self,
        agents: list[BaseAgent],
        aggregation: str = 'mean'
    ):
        super().__init__()
        self.agents = nn.ModuleList(agents)
        self.aggregation = aggregation
    
    def forward(self, x: torch.Tensor) -> AgentOutput:
        """Forward pass through all agents and aggregate.
        
        Args:
            x: Input tensor (B, D)
            
        Returns:
            Aggregated AgentOutput
        """
        outputs = [agent(x) for agent in self.agents]
        
        # Aggregate logits
        logits_stack = torch.stack([out.logits for out in outputs])  # (M, B, C)
        
        if self.aggregation == 'mean':
            aggregated_logits = logits_stack.mean(dim=0)
        elif self.aggregation == 'vote':
            # Majority vote
            predictions = logits_stack.argmax(dim=-1)  # (M, B)
            aggregated_logits = torch.nn.functional.one_hot(
                predictions.mode(dim=0)[0],
                num_classes=logits_stack.size(-1)
            ).float()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Compute ensemble uncertainty (variance)
        ensemble_uncertainty = logits_stack.var(dim=0).mean(dim=-1)  # (B,)
        
        return AgentOutput(
            logits=aggregated_logits,
            features=outputs[0].features,  # Use first agent's features
            uncertainty=ensemble_uncertainty,
            metadata={'num_agents': len(self.agents)}
        )
    
    def estimate_uncertainty(self, x: torch.Tensor) -> UncertaintyOutput:
        """Estimate uncertainty using ensemble disagreement.
        
        Args:
            x: Input tensor (B, D)
            
        Returns:
            UncertaintyOutput with epistemic uncertainty from disagreement
        """
        outputs = [agent(x) for agent in self.agents]
        logits_stack = torch.stack([out.logits for out in outputs])  # (M, B, C)
        
        # Epistemic uncertainty = variance across ensemble
        epistemic = logits_stack.var(dim=0).mean(dim=-1)  # (B,)
        
        # Mean prediction
        mean_logits = logits_stack.mean(dim=0)
        probs = torch.softmax(mean_logits, dim=-1)
        
        # Aleatoric uncertainty = entropy of mean prediction
        aleatoric = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B,)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        # Confidence
        confidence = probs.max(dim=-1)[0]
        
        return UncertaintyOutput(
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence
        )

