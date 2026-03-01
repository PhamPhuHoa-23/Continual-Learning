"""
Aggregator: Decision Tree for Continual Learning.

Aggregates hidden labels from agents and makes final class predictions.
Uses incremental decision trees (Hoeffding Trees) to support online learning
and continual learning without catastrophic forgetting.

Key Features:
    - Handles continuous features (softmax probabilities from agents)
    - Incremental learning (learns one example at a time)
    - Supports new classes without retraining
    - No task ID required at test time
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from river.ensemble import AdaptiveRandomForestClassifier


class IncrementalTreeAggregator:
    """
    Incremental decision tree for aggregating agent outputs.
    
    Uses Hoeffding Tree (VFDT) from the River library, which supports:
        - Online learning (learn_one)
        - Continuous features
        - New classes without retraining
        - Adaptive splitting criteria
    
    Args:
        grace_period: Number of instances before split attempt (default: 200)
        split_confidence: Confidence level for splitting (default: 1e-5)
        leaf_prediction: Prediction strategy at leaves (default: 'nba')
            - 'mc': Majority class
            - 'nb': Naive Bayes
            - 'nba': Naive Bayes Adaptive (best for continual learning)
        adaptive: Use adaptive tree (handles concept drift) (default: True)
    
    Reference:
        Domingos & Hulten (2000) "Mining High-Speed Data Streams"
        https://riverml.xyz/
    """
    
    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-5,
        leaf_prediction: str = 'nba',
        adaptive: bool = True
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.leaf_prediction = leaf_prediction
        self.adaptive = adaptive
        
        # Create tree
        if adaptive:
            self.tree = HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                delta=split_confidence,
                leaf_prediction=leaf_prediction
            )
        else:
            self.tree = HoeffdingTreeClassifier(
                grace_period=grace_period,
                delta=split_confidence,
                leaf_prediction=leaf_prediction
            )
        
        # Statistics
        self.num_examples_seen = 0
        self.num_classes_seen = 0
        self.class_counts = {}
    
    def learn_one(
        self,
        hidden_labels: np.ndarray,
        label: int
    ):
        """
        Learn from one example (online learning).
        
        Args:
            hidden_labels: (num_slots × k × num_prototypes,) or flattened array
                Concatenated softmax probabilities from all agents
            label: Ground truth class label
        """
        # Convert to feature dict (River format)
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        
        # Learn
        self.tree.learn_one(x, label)
        
        # Update statistics
        self.num_examples_seen += 1
        if label not in self.class_counts:
            self.num_classes_seen += 1
            self.class_counts[label] = 0
        self.class_counts[label] += 1
    
    def predict_one(
        self,
        hidden_labels: np.ndarray
    ) -> Optional[int]:
        """
        Predict class for one example.
        
        Args:
            hidden_labels: (num_slots × k × num_prototypes,)
        
        Returns:
            predicted_label: Predicted class (int) or None if tree is empty
        """
        # Convert to feature dict
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        
        # Predict
        pred = self.tree.predict_one(x)
        
        return pred
    
    def predict_proba_one(
        self,
        hidden_labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Predict class probabilities for one example.
        
        Args:
            hidden_labels: (num_slots × k × num_prototypes,)
        
        Returns:
            class_probabilities: Dict mapping class → probability
        """
        # Convert to feature dict
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        
        # Predict probabilities
        proba = self.tree.predict_proba_one(x)
        
        return proba if proba is not None else {}
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        return {
            'num_examples_seen': self.num_examples_seen,
            'num_classes_seen': self.num_classes_seen,
            'class_counts': self.class_counts,
            'tree_size': self.tree.n_nodes if hasattr(self.tree, 'n_nodes') else None
        }


class EnsembleTreeAggregator:
    """
    Ensemble of incremental decision trees for better performance.
    
    Uses Adaptive Random Forest from River, which combines:
        - Multiple Hoeffding Trees
        - Online bagging
        - ADWIN drift detection
    
    Args:
        n_models: Number of trees in ensemble (default: 10)
        max_features: Max features per tree (default: 'sqrt')
        grace_period: Instances before split (default: 200)
        split_confidence: Confidence for splitting (default: 1e-5)
    """
    
    def __init__(
        self,
        n_models: int = 10,
        max_features: str = 'sqrt',
        grace_period: int = 200,
        split_confidence: float = 1e-5
    ):
        self.n_models = n_models
        
        self.forest = AdaptiveRandomForestClassifier(
            n_models=n_models,
            max_features=max_features,
            grace_period=grace_period,
            delta=split_confidence
        )
        
        # Statistics
        self.num_examples_seen = 0
        self.num_classes_seen = 0
        self.class_counts = {}
    
    def learn_one(
        self,
        hidden_labels: np.ndarray,
        label: int
    ):
        """Learn from one example."""
        # Convert to feature dict
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        
        # Learn
        self.forest.learn_one(x, label)
        
        # Update statistics
        self.num_examples_seen += 1
        if label not in self.class_counts:
            self.num_classes_seen += 1
            self.class_counts[label] = 0
        self.class_counts[label] += 1
    
    def predict_one(
        self,
        hidden_labels: np.ndarray
    ) -> Optional[int]:
        """Predict class for one example."""
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        pred = self.forest.predict_one(x)
        return pred
    
    def predict_proba_one(
        self,
        hidden_labels: np.ndarray
    ) -> Dict[int, float]:
        """Predict class probabilities."""
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        proba = self.forest.predict_proba_one(x)
        return proba if proba is not None else {}
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics."""
        return {
            'num_examples_seen': self.num_examples_seen,
            'num_classes_seen': self.num_classes_seen,
            'class_counts': self.class_counts,
            'n_models': self.n_models
        }


class BatchTreeAggregator:
    """
    Wrapper to handle batched predictions (for efficiency).
    
    Wraps an IncrementalTreeAggregator and provides batch interface.
    Note: Learning is still incremental (one-by-one).
    
    Args:
        aggregator: IncrementalTreeAggregator or EnsembleTreeAggregator
    """
    
    def __init__(self, aggregator):
        self.aggregator = aggregator
    
    def learn_batch(
        self,
        hidden_labels_batch: torch.Tensor,
        labels_batch: torch.Tensor
    ):
        """
        Learn from a batch (internally loops one-by-one).
        
        Args:
            hidden_labels_batch: (batch_size, feature_dim)
            labels_batch: (batch_size,)
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        labels_np = labels_batch.cpu().numpy()
        
        for hidden_labels, label in zip(hidden_labels_np, labels_np):
            self.aggregator.learn_one(hidden_labels, int(label))
    
    def predict_batch(
        self,
        hidden_labels_batch: torch.Tensor
    ) -> List[Optional[int]]:
        """
        Predict for a batch.
        
        Args:
            hidden_labels_batch: (batch_size, feature_dim)
        
        Returns:
            predictions: List of predicted labels
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        
        predictions = [
            self.aggregator.predict_one(hidden_labels)
            for hidden_labels in hidden_labels_np
        ]
        
        return predictions
    
    def predict_proba_batch(
        self,
        hidden_labels_batch: torch.Tensor
    ) -> List[Dict[int, float]]:
        """
        Predict probabilities for a batch.
        
        Args:
            hidden_labels_batch: (batch_size, feature_dim)
        
        Returns:
            probabilities: List of probability dicts
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        
        probas = [
            self.aggregator.predict_proba_one(hidden_labels)
            for hidden_labels in hidden_labels_np
        ]
        
        return probas
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return self.aggregator.get_stats()


def create_aggregator(
    aggregator_type: str = 'hoeffding',
    **kwargs
):
    """
    Factory function to create aggregators.
    
    Args:
        aggregator_type: One of ['hoeffding', 'hoeffding_adaptive', 'ensemble']
        **kwargs: Additional arguments for specific aggregators
    
    Returns:
        Aggregator instance
    
    Example:
        >>> agg = create_aggregator('hoeffding_adaptive', grace_period=200)
        >>> agg.learn_one(hidden_labels, label)
        >>> pred = agg.predict_one(hidden_labels)
    """
    if aggregator_type == 'hoeffding':
        return IncrementalTreeAggregator(adaptive=False, **kwargs)
    elif aggregator_type == 'hoeffding_adaptive':
        return IncrementalTreeAggregator(adaptive=True, **kwargs)
    elif aggregator_type == 'ensemble':
        return EnsembleTreeAggregator(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregator_type: {aggregator_type}. "
            f"Choose from: hoeffding, hoeffding_adaptive, ensemble"
        )
