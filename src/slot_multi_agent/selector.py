"""
Top-K Agent Selector.

Given a slot and performance estimators, select the top-k agents
that are most likely to perform well on that slot.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union
from .estimators import VAEEstimator, MLPEstimator, HybridEstimator


class TopKAgentSelector:
    """
    Select top-k agents for a given slot based on performance estimates.
    
    Simple selection strategy: Estimate performance for all agents,
    sort by score, return top-k.
    """
    
    def __init__(
        self,
        estimators: List[Union[VAEEstimator, MLPEstimator, HybridEstimator]],
        k: int = 3,
        temperature: float = 1.0
    ):
        """
        Args:
            estimators: List of performance estimators (one per agent or shared)
            k: Number of agents to select
            temperature: Temperature for softmax (if using probabilistic selection)
        """
        self.estimators = estimators
        self.k = k
        self.temperature = temperature
        self.num_agents = len(estimators)
    
    def select_top_k(
        self,
        slot: torch.Tensor,
        return_scores: bool = True
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Select top-k agents for a slot.
        
        Args:
            slot: (slot_dim,) - slot representation
            return_scores: If True, also return performance scores
            
        Returns:
            If return_scores=False:
                selected_ids: List of k agent IDs
            If return_scores=True:
                (selected_ids, scores): Tuple of agent IDs and their scores
        """
        # Get scores from all estimators
        scores = []
        
        for agent_id, estimator in enumerate(self.estimators):
            if isinstance(estimator, MLPEstimator):
                # MLP estimator needs agent_id
                score = estimator.estimate_performance(slot, agent_id)
            else:
                # VAE or Hybrid estimator
                score = estimator.estimate_performance(slot)
            
            scores.append(score.item() if torch.is_tensor(score) else score)
        
        # Sort by score (descending)
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        # Select top-k
        selected_ids = sorted_indices[:self.k]
        selected_scores = [scores[i] for i in selected_ids]
        
        if return_scores:
            return selected_ids, selected_scores
        else:
            return selected_ids
    
    def select_batch(
        self,
        slots: torch.Tensor,
        return_scores: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select top-k agents for a batch of slots.
        
        Args:
            slots: (batch_size, num_slots, slot_dim)
            return_scores: If True, also return scores
            
        Returns:
            If return_scores=False:
                selected_ids: (batch_size, num_slots, k) - agent IDs
            If return_scores=True:
                (selected_ids, scores): Agent IDs and scores
        """
        batch_size, num_slots, slot_dim = slots.shape
        
        all_selected_ids = []
        all_scores = []
        
        for b in range(batch_size):
            batch_selected = []
            batch_scores = []
            
            for s in range(num_slots):
                slot = slots[b, s]
                
                if return_scores:
                    ids, scores = self.select_top_k(slot, return_scores=True)
                else:
                    ids = self.select_top_k(slot, return_scores=False)
                    scores = None
                
                batch_selected.append(ids)
                if scores is not None:
                    batch_scores.append(scores)
            
            all_selected_ids.append(batch_selected)
            if len(batch_scores) > 0:
                all_scores.append(batch_scores)
        
        # Convert to tensor
        selected_ids_tensor = torch.tensor(all_selected_ids, dtype=torch.long)
        
        if return_scores:
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            return selected_ids_tensor, scores_tensor
        else:
            return selected_ids_tensor
    
    def select_probabilistic(
        self,
        slot: torch.Tensor,
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Probabilistic selection: Sample k agents based on scores.
        
        Useful for exploration during training.
        
        Args:
            slot: (slot_dim,)
            return_probs: If True, return sampling probabilities
            
        Returns:
            selected_ids: List of k agent IDs (sampled)
            probs: (optional) Sampling probabilities
        """
        # Get scores
        scores = []
        for agent_id, estimator in enumerate(self.estimators):
            if isinstance(estimator, MLPEstimator):
                score = estimator.estimate_performance(slot, agent_id)
            else:
                score = estimator.estimate_performance(slot)
            scores.append(score.item() if torch.is_tensor(score) else score)
        
        # Convert to probabilities with temperature
        scores_tensor = torch.tensor(scores) / self.temperature
        probs = torch.softmax(scores_tensor, dim=0)
        
        # Sample k agents (without replacement)
        selected_ids = torch.multinomial(
            probs, num_samples=self.k, replacement=False
        ).tolist()
        
        if return_probs:
            return selected_ids, probs
        else:
            return selected_ids
    
    def update_k(self, new_k: int):
        """Update the number of agents to select."""
        self.k = min(new_k, self.num_agents)
    
    def get_all_scores(self, slot: torch.Tensor) -> torch.Tensor:
        """
        Get performance scores from all agents (for analysis).
        
        Args:
            slot: (slot_dim,)
            
        Returns:
            scores: (num_agents,) - performance estimates
        """
        scores = []
        
        for agent_id, estimator in enumerate(self.estimators):
            if isinstance(estimator, MLPEstimator):
                score = estimator.estimate_performance(slot, agent_id)
            else:
                score = estimator.estimate_performance(slot)
            scores.append(score.item() if torch.is_tensor(score) else score)
        
        return torch.tensor(scores)


class AdaptiveKSelector(TopKAgentSelector):
    """
    Adaptive selector that adjusts k based on uncertainty.
    
    High uncertainty → use more agents (higher k)
    Low uncertainty → use fewer agents (lower k)
    """
    
    def __init__(
        self,
        estimators: List,
        k_min: int = 2,
        k_max: int = 5,
        temperature: float = 1.0
    ):
        super().__init__(estimators, k=k_min, temperature=temperature)
        self.k_min = k_min
        self.k_max = k_max
    
    def select_adaptive(
        self,
        slot: torch.Tensor,
        return_scores: bool = True
    ) -> Union[List[int], Tuple[List[int], List[float], int]]:
        """
        Select agents with adaptive k.
        
        Args:
            slot: (slot_dim,)
            return_scores: If True, return scores and selected k
            
        Returns:
            selected_ids, scores, k_used
        """
        # Get all scores
        all_scores = self.get_all_scores(slot)
        
        # Compute uncertainty (entropy of score distribution)
        probs = torch.softmax(all_scores, dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(float(len(all_scores))))
        normalized_entropy = entropy / max_entropy
        
        # Adaptive k: high entropy → more agents
        k_adaptive = int(
            self.k_min + (self.k_max - self.k_min) * normalized_entropy
        )
        k_adaptive = max(self.k_min, min(k_adaptive, self.k_max))
        
        # Select top-k_adaptive
        self.k = k_adaptive
        selected_ids, selected_scores = self.select_top_k(slot, return_scores=True)
        
        if return_scores:
            return selected_ids, selected_scores, k_adaptive
        else:
            return selected_ids

