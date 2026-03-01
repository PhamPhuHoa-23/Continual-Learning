"""
Base Module for All Components

Provides common interface and utilities for all framework components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModule(nn.Module, ABC):
    """
    Base class for all framework modules.

    Provides:
    - Consistent interface for forward pass
    - Configuration storage
    - Freezing/unfreezing utilities
    - State management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base module.

        Args:
            config: Configuration dict for this module
        """
        super().__init__()
        self.config = config or {}
        self._is_frozen = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass - must be implemented by subclasses.

        Returns:
            Output (can be tensor, dict, tuple, etc.)
        """
        raise NotImplementedError

    def freeze(self):
        """Freeze all parameters in this module."""
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True

    def unfreeze(self):
        """Unfreeze all parameters in this module."""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False

    def is_frozen(self) -> bool:
        """Check if module is frozen."""
        return self._is_frozen

    def get_config(self) -> Dict[str, Any]:
        """Get module configuration."""
        return self.config.copy()

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_device(self) -> torch.device:
        """Get device of module parameters."""
        return next(self.parameters()).device

    def summary(self) -> str:
        """
        Get summary string of module.

        Returns:
            Summary string with parameter counts and frozen status
        """
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        frozen_params = total_params - trainable_params

        lines = [
            f"{self.__class__.__name__}:",
            f"  Total parameters: {total_params:,}",
            f"  Trainable: {trainable_params:,}",
            f"  Frozen: {frozen_params:,}",
            f"  Status: {'FROZEN' if self._is_frozen else 'TRAINABLE'}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        params = self.count_parameters()
        status = "frozen" if self._is_frozen else "trainable"
        return f"{self.__class__.__name__}(params={params:,}, {status})"


class BaseAgent(BaseModule):
    """
    Base class for agent networks.

    Agents map slots to hidden representations: s_k -> h_k
    """

    @abstractmethod
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for agent.

        Args:
            slots: Slot representations, shape (B, K, D_slot) or (B, D_slot)

        Returns:
            Hidden representations, shape (B, K, D_hidden) or (B, D_hidden)
        """
        raise NotImplementedError


class BaseRouter(BaseModule):
    """
    Base class for routing mechanisms.

    Routes slots to agents based on some scoring mechanism.
    """

    @abstractmethod
    def compute_scores(
        self,
        slot: torch.Tensor,
        agent_ids: Optional[list] = None
    ) -> torch.Tensor:
        """
        Compute routing scores for a slot against agents.

        Args:
            slot: Single slot representation, shape (D_slot,)
            agent_ids: List of agent IDs to score against. If None, use all.

        Returns:
            Scores for each agent, shape (N_agents,)
        """
        raise NotImplementedError

    @abstractmethod
    def update_stats(self, agent_id: int, slot: torch.Tensor):
        """
        Update routing statistics for an agent.

        Args:
            agent_id: ID of agent
            slot: Slot that was routed to this agent
        """
        raise NotImplementedError


class BaseAggregator(BaseModule):
    """
    Base class for aggregating agent outputs.

    Aggregates multiple agent hidden representations into single representation.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        agent_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate hidden states from multiple agents.

        Args:
            hidden_states: Hidden states from agents, shape (B, K, D_hidden)
            agent_assignments: Agent ID for each slot, shape (B, K)

        Returns:
            Aggregated representation, shape (B, D_hidden)
        """
        raise NotImplementedError


class BaseClassifier(BaseModule):
    """
    Base class for classification heads.
    """

    @abstractmethod
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Fit classifier on new data (for incremental methods like SLDA).

        Args:
            features: Feature representations, shape (N, D)
            labels: Class labels, shape (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            features: Feature representations, shape (N, D)

        Returns:
            Predicted class labels, shape (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            features: Feature representations, shape (N, D)

        Returns:
            Class probabilities, shape (N, n_classes)
        """
        raise NotImplementedError
