"""
Bandit-based Agent Selection and Weighting.

Uses multi-armed bandit algorithms to balance exploration-exploitation
when selecting and weighting agents for each slot.

Supported algorithms:
    - UCB (Upper Confidence Bound)
    - Thompson Sampling
    - Epsilon-Greedy
    - Weighted combination based on estimated performance

NOTE: Specific methodology to be determined by professor.
This module provides a flexible framework for different bandit strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BanditSelector(ABC):
    """
    Abstract base class for bandit-based agent selection.
    
    Each slot requires selecting k agents from a pool of N agents.
    Bandit algorithms balance:
        - Exploitation: Choose agents with high estimated performance
        - Exploration: Try less-explored agents to gather information
    """
    
    @abstractmethod
    def select_and_weight(
        self,
        slot: torch.Tensor,
        estimated_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k agents and compute their weights.
        
        Args:
            slot: Input slot (batch_size, slot_dim)
            estimated_scores: Estimated performance for each agent (batch_size, num_agents)
                Higher score = better expected performance
            k: Number of agents to select
        
        Returns:
            selected_indices: (batch_size, k) - indices of selected agents
            weights: (batch_size, k) - weights for each selected agent (sum to 1)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        agent_idx: int,
        slot: torch.Tensor,
        reward: float
    ):
        """
        Update bandit statistics after observing reward.
        
        Args:
            agent_idx: Index of agent that was selected
            slot: Slot that was processed
            reward: Observed reward (e.g., negative loss, accuracy)
        """
        pass


class UCBSelector(BanditSelector):
    """
    Upper Confidence Bound (UCB) for agent selection.
    
    UCB balances exploitation (mean reward) and exploration (uncertainty).
    
    Formula:
        UCB(agent_i) = μ_i + c * sqrt(log(t) / n_i)
        where:
            μ_i = mean reward for agent i
            n_i = number of times agent i was selected
            t = total number of selections
            c = exploration constant
    
    Args:
        num_agents: Total number of agents
        exploration_constant: UCB exploration constant (default: 2.0)
    
    Reference:
        Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"
    """
    
    def __init__(self, num_agents: int, exploration_constant: float = 2.0):
        self.num_agents = num_agents
        self.c = exploration_constant
        
        # Statistics (per agent)
        self.counts = np.zeros(num_agents)  # n_i
        self.values = np.zeros(num_agents)  # μ_i
        self.total_count = 0  # t
    
    def select_and_weight(
        self,
        slot: torch.Tensor,
        estimated_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k agents using UCB, then weight by softmax of UCB scores.
        """
        batch_size = slot.size(0)
        device = slot.device
        
        # Compute UCB scores
        ucb_scores = self._compute_ucb()  # (num_agents,)
        ucb_scores_tensor = torch.tensor(ucb_scores, device=device)
        
        # Combine with estimated scores (e.g., average or product)
        # Here we use weighted sum
        combined_scores = estimated_scores + 0.5 * ucb_scores_tensor.unsqueeze(0)
        
        # Select top-k
        top_k_values, top_k_indices = torch.topk(combined_scores, k=k, dim=1)
        
        # Compute weights via softmax
        weights = torch.softmax(top_k_values, dim=1)  # (batch_size, k)
        
        return top_k_indices, weights
    
    def _compute_ucb(self) -> np.ndarray:
        """Compute UCB scores for all agents."""
        ucb_scores = np.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            if self.counts[i] == 0:
                # Infinite bonus for unexplored agents
                ucb_scores[i] = float('inf')
            else:
                # μ_i + c * sqrt(log(t) / n_i)
                exploration_bonus = self.c * np.sqrt(
                    np.log(self.total_count + 1) / self.counts[i]
                )
                ucb_scores[i] = self.values[i] + exploration_bonus
        
        return ucb_scores
    
    def update(self, agent_idx: int, slot: torch.Tensor, reward: float):
        """Update statistics for agent after observing reward."""
        self.counts[agent_idx] += 1
        self.total_count += 1
        
        # Incremental mean update
        n = self.counts[agent_idx]
        self.values[agent_idx] += (reward - self.values[agent_idx]) / n


class ThompsonSamplingSelector(BanditSelector):
    """
    Thompson Sampling for agent selection.
    
    Maintains Beta distributions for each agent's success probability.
    Samples from posterior to balance exploration-exploitation.
    
    Args:
        num_agents: Total number of agents
        alpha_init: Initial alpha (successes + 1) for Beta prior (default: 1.0)
        beta_init: Initial beta (failures + 1) for Beta prior (default: 1.0)
    
    Reference:
        Thompson (1933) "On the Likelihood that One Unknown Probability 
        Exceeds Another in View of the Evidence of Two Samples"
    """
    
    def __init__(
        self,
        num_agents: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0
    ):
        self.num_agents = num_agents
        self.alphas = np.full(num_agents, alpha_init)  # Successes
        self.betas = np.full(num_agents, beta_init)    # Failures
    
    def select_and_weight(
        self,
        slot: torch.Tensor,
        estimated_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k agents by sampling from Beta posteriors.
        """
        batch_size = slot.size(0)
        device = slot.device
        
        # Sample from Beta posteriors
        sampled_probs = np.random.beta(self.alphas, self.betas)  # (num_agents,)
        sampled_probs_tensor = torch.tensor(sampled_probs, device=device, dtype=torch.float32)
        
        # Combine with estimated scores
        combined_scores = estimated_scores + sampled_probs_tensor.unsqueeze(0)
        
        # Select top-k
        top_k_values, top_k_indices = torch.topk(combined_scores, k=k, dim=1)
        
        # Compute weights (softmax of sampled probabilities)
        weights = torch.softmax(top_k_values, dim=1)
        
        return top_k_indices, weights
    
    def update(self, agent_idx: int, slot: torch.Tensor, reward: float):
        """
        Update Beta distribution for agent.
        
        Reward should be in [0, 1] or converted to binary success/failure.
        """
        # Convert reward to success probability
        # Assume reward in [0, 1] where 1 = success, 0 = failure
        success = reward  # Can be fractional for soft updates
        failure = 1 - reward
        
        self.alphas[agent_idx] += success
        self.betas[agent_idx] += failure


class EpsilonGreedySelector(BanditSelector):
    """
    Epsilon-Greedy selection strategy.
    
    With probability ε: explore (random selection)
    With probability 1-ε: exploit (select top-k by estimated score)
    
    Args:
        num_agents: Total number of agents
        epsilon: Exploration probability (default: 0.1)
        epsilon_decay: Decay rate for epsilon (default: 0.995)
        min_epsilon: Minimum epsilon value (default: 0.01)
    """
    
    def __init__(
        self,
        num_agents: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Statistics
        self.counts = np.zeros(num_agents)
        self.values = np.zeros(num_agents)
    
    def select_and_weight(
        self,
        slot: torch.Tensor,
        estimated_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k agents with epsilon-greedy strategy.
        """
        batch_size = slot.size(0)
        device = slot.device
        
        # Explore or exploit?
        if np.random.rand() < self.epsilon:
            # Explore: random selection
            indices = torch.randperm(self.num_agents, device=device)[:k]
            indices = indices.unsqueeze(0).expand(batch_size, -1)
            weights = torch.ones(batch_size, k, device=device) / k
        else:
            # Exploit: top-k by estimated score
            top_k_values, indices = torch.topk(estimated_scores, k=k, dim=1)
            weights = torch.softmax(top_k_values, dim=1)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return indices, weights
    
    def update(self, agent_idx: int, slot: torch.Tensor, reward: float):
        """Update statistics for agent."""
        self.counts[agent_idx] += 1
        n = self.counts[agent_idx]
        self.values[agent_idx] += (reward - self.values[agent_idx]) / n


class WeightedTopKSelector(BanditSelector):
    """
    Simple weighted top-k selection (no exploration).
    
    Selects top-k agents by estimated score and weights them via softmax.
    This is the baseline without bandit algorithms.
    
    Args:
        num_agents: Total number of agents
        temperature: Temperature for softmax weighting (default: 1.0)
    """
    
    def __init__(self, num_agents: int, temperature: float = 1.0):
        self.num_agents = num_agents
        self.temperature = temperature
    
    def select_and_weight(
        self,
        slot: torch.Tensor,
        estimated_scores: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k agents by score and weight via softmax.
        """
        # Select top-k
        top_k_values, top_k_indices = torch.topk(estimated_scores, k=k, dim=1)
        
        # Compute weights via softmax
        weights = torch.softmax(top_k_values / self.temperature, dim=1)
        
        return top_k_indices, weights
    
    def update(self, agent_idx: int, slot: torch.Tensor, reward: float):
        """No-op for simple top-k selector."""
        pass


class UCBWeightedMoE:
    """
    UCB-based Weighted Mixture of Experts.

    Adapted from omoe-codebase UCB_Bandit for the slot-based multi-agent
    continual-learning system.  Instead of selecting a single subset of
    experts (as in combo_bandit.py), this class computes UCB-derived
    weights for ALL pre-filtered agents to form a weighted committee.

    Full inference pipeline (per slot):
        1. VAE/MLP estimators fast-filter 50 agents → top-K candidates
        2. UCBWeightedMoE computes UCB score for each of the K agents
        3. Softmax(UCB / temperature) → committee weights  (sums to 1)
        4. All K agents run on the slot → weighted sum of outputs

    UCB formula per agent i:
        UCB(i) = μ_i + c × √(ln t / n_i)
    where
        μ_i  = empirical mean reward for agent i
        n_i  = number of rounds in which agent i participated
        t    = total number of selection rounds
        c    = exploration constant (default √2)

    During a burn-in period (first ``burn_in`` rounds) all agents
    receive uniform weights to ensure sufficient exploration.

    Reward distribution:  After the CRP aggregator makes a prediction
    the binary reward (correct/incorrect) is distributed to every
    committee member *proportionally to its weight*.

    Reference:
        Auer et al. (2002) "Finite-time Analysis of the Multiarmed
        Bandit Problem"
        omoe-codebase: combo_bandit.py (UCB_Bandit), theta_wmv.py

    Args:
        num_agents: Total number of agents in the pool.
        exploration_constant: UCB exploration bonus coefficient.
        temperature: Softmax temperature for weight computation.
        burn_in: Number of initial rounds with uniform weights.
    """

    def __init__(
        self,
        num_agents: int,
        exploration_constant: float = 1.414,   # √2
        temperature: float = 1.0,
        burn_in: int = 100,
    ):
        self.num_agents = num_agents
        self.c = exploration_constant
        self.temperature = temperature
        self.burn_in = burn_in

        # Per-agent statistics
        self.counts = np.zeros(num_agents)          # n_i
        self.values = np.zeros(num_agents)          # μ_i  (mean reward)
        self.total_count = 0                        # t

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    def get_ucb_scores(self, agent_ids: List[int]) -> np.ndarray:
        """
        Compute raw UCB scores for a subset of agents.

        Args:
            agent_ids: List of agent indices (already filtered by estimators).

        Returns:
            ucb_scores: (len(agent_ids),) — raw UCB values.
        """
        scores = np.zeros(len(agent_ids))
        for i, aid in enumerate(agent_ids):
            if self.counts[aid] == 0:
                scores[i] = float('inf')
            else:
                scores[i] = self.values[aid] + self.c * np.sqrt(
                    np.log(self.total_count + 1) / self.counts[aid]
                )
        return scores

    def get_weights(
        self,
        filtered_agent_ids: List[int],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Compute committee weights for ALL filtered agents.

        During burn-in (total_count < burn_in) returns uniform weights
        to encourage exploration.  Afterwards uses softmax over UCB
        scores.

        Args:
            filtered_agent_ids: Agent IDs that passed the VAE/MLP filter.

        Returns:
            agent_ids:  Same list (for downstream convenience).
            weights:    (K,) array summing to 1.
        """
        K = len(filtered_agent_ids)
        if K == 0:
            return filtered_agent_ids, np.array([])

        # Burn-in → uniform
        if self.total_count < self.burn_in:
            return filtered_agent_ids, np.ones(K) / K

        ucb_scores = self.get_ucb_scores(filtered_agent_ids)

        # Handle unexplored agents (inf UCB)
        has_inf = np.isinf(ucb_scores)
        if has_inf.any():
            weights = np.zeros(K)
            weights[has_inf] = 1.0 / has_inf.sum()
        else:
            s = ucb_scores / self.temperature
            s -= s.max()                       # numerical stability
            exp_s = np.exp(s)
            weights = exp_s / exp_s.sum()

        return filtered_agent_ids, weights

    # ------------------------------------------------------------------
    #  Reward updates
    # ------------------------------------------------------------------

    def update(self, agent_id: int, reward: float):
        """Incremental mean-reward update for a single agent.
        NOTE: does NOT increment total_count — caller must do that once per sample.
        """
        self.counts[agent_id] += 1
        n = self.counts[agent_id]
        self.values[agent_id] += (reward - self.values[agent_id]) / n

    def update_batch(
        self,
        agent_ids: List[int],
        weights: np.ndarray,
        reward: float,
    ):
        """
        Distribute reward to all committee members proportional to weight.
        Increments total_count ONCE per call (one call = one slot of one sample).

        Args:
            agent_ids: Committee agent IDs.
            weights:   Their committee weights (sum to 1).
            reward:    Global reward (1.0 = correct, 0.0 = wrong).
        """
        for aid, w in zip(agent_ids, weights):
            self.update(aid, reward * w)
        # One round = one slot assignment decision
        self.total_count += 1

    # ------------------------------------------------------------------
    #  Convenience / analysis
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return per-agent UCB statistics (only for agents that were used)."""
        stats: dict = {'total_count': int(self.total_count), 'per_agent': {}}
        for i in range(self.num_agents):
            if self.counts[i] > 0:
                stats['per_agent'][i] = {
                    'count': int(self.counts[i]),
                    'value': float(self.values[i]),
                    'ucb': float(
                        self.values[i]
                        + self.c * np.sqrt(np.log(self.total_count + 1) / self.counts[i])
                    ),
                }
        return stats

    def save(self, path: str):
        """Persist bandit state to disk."""
        np.savez(
            path,
            counts=self.counts,
            values=self.values,
            total_count=np.array([self.total_count]),
        )

    def load(self, path: str):
        """Restore bandit state from disk."""
        data = np.load(path)
        self.counts = data['counts']
        self.values = data['values']
        self.total_count = int(data['total_count'][0])


def create_bandit_selector(
    strategy: str,
    num_agents: int,
    **kwargs
) -> "BanditSelector | UCBWeightedMoE":
    """
    Factory function to create bandit selectors.

    Args:
        strategy: One of ['ucb', 'thompson', 'epsilon_greedy',
                  'weighted_topk', 'ucb_weighted_moe']
        num_agents: Total number of agents
        **kwargs: Additional arguments for specific selectors

    Returns:
        BanditSelector (or UCBWeightedMoE) instance.

    Example:
        >>> selector = create_bandit_selector('ucb', num_agents=50, exploration_constant=2.0)
        >>> moe = create_bandit_selector('ucb_weighted_moe', num_agents=50)
    """
    if strategy == 'ucb':
        return UCBSelector(num_agents, **kwargs)
    elif strategy == 'thompson':
        return ThompsonSamplingSelector(num_agents, **kwargs)
    elif strategy == 'epsilon_greedy':
        return EpsilonGreedySelector(num_agents, **kwargs)
    elif strategy == 'weighted_topk':
        return WeightedTopKSelector(num_agents, **kwargs)
    elif strategy == 'ucb_weighted_moe':
        return UCBWeightedMoE(num_agents, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: ucb, thompson, epsilon_greedy, weighted_topk, "
            f"ucb_weighted_moe"
        )


