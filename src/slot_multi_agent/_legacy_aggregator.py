"""
Legacy Aggregators: Hoeffding Tree-based aggregators.

Preserved for backward compatibility. These require the 'river' package.

Usage:
    from ._legacy_aggregator import IncrementalTreeAggregator, EnsembleTreeAggregator
"""

import numpy as np
from typing import Dict, Optional
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from river.ensemble import AdaptiveRandomForestClassifier


class IncrementalTreeAggregator:
    """
    Incremental decision tree for aggregating agent outputs.

    Uses Hoeffding Tree (VFDT) from the River library.

    Args:
        grace_period: Number of instances before split attempt.
        split_confidence: Confidence level for splitting.
        leaf_prediction: Prediction strategy at leaves ('mc', 'nb', 'nba').
        adaptive: Use adaptive tree (handles concept drift).
    """

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-5,
        leaf_prediction: str = "nba",
        adaptive: bool = True,
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.leaf_prediction = leaf_prediction
        self.adaptive = adaptive

        if adaptive:
            self.tree = HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                delta=split_confidence,
                leaf_prediction=leaf_prediction,
            )
        else:
            self.tree = HoeffdingTreeClassifier(
                grace_period=grace_period,
                delta=split_confidence,
                leaf_prediction=leaf_prediction,
            )

        self.num_examples_seen = 0
        self.num_classes_seen = 0
        self.class_counts = {}

    def learn_one(self, hidden_labels: np.ndarray, label: int):
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        self.tree.learn_one(x, label)
        self.num_examples_seen += 1
        if label not in self.class_counts:
            self.num_classes_seen += 1
            self.class_counts[label] = 0
        self.class_counts[label] += 1

    def predict_one(self, hidden_labels: np.ndarray) -> Optional[int]:
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        return self.tree.predict_one(x)

    def predict_proba_one(self, hidden_labels: np.ndarray) -> Dict[int, float]:
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        proba = self.tree.predict_proba_one(x)
        return proba if proba is not None else {}

    def get_stats(self) -> Dict:
        return {
            "num_examples_seen": self.num_examples_seen,
            "num_classes_seen": self.num_classes_seen,
            "class_counts": self.class_counts,
            "tree_size": self.tree.n_nodes if hasattr(self.tree, "n_nodes") else None,
        }


class EnsembleTreeAggregator:
    """
    Ensemble of incremental decision trees (Adaptive Random Forest).

    Args:
        n_models: Number of trees in ensemble.
        max_features: Max features per tree.
        grace_period: Instances before split.
        split_confidence: Confidence for splitting.
    """

    def __init__(
        self,
        n_models: int = 10,
        max_features: str = "sqrt",
        grace_period: int = 200,
        split_confidence: float = 1e-5,
    ):
        self.n_models = n_models
        self.forest = AdaptiveRandomForestClassifier(
            n_models=n_models,
            max_features=max_features,
            grace_period=grace_period,
            delta=split_confidence,
        )
        self.num_examples_seen = 0
        self.num_classes_seen = 0
        self.class_counts = {}

    def learn_one(self, hidden_labels: np.ndarray, label: int):
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        self.forest.learn_one(x, label)
        self.num_examples_seen += 1
        if label not in self.class_counts:
            self.num_classes_seen += 1
            self.class_counts[label] = 0
        self.class_counts[label] += 1

    def predict_one(self, hidden_labels: np.ndarray) -> Optional[int]:
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        return self.forest.predict_one(x)

    def predict_proba_one(self, hidden_labels: np.ndarray) -> Dict[int, float]:
        x = {f"f_{i}": float(v) for i, v in enumerate(hidden_labels)}
        proba = self.forest.predict_proba_one(x)
        return proba if proba is not None else {}

    def get_stats(self) -> Dict:
        return {
            "num_examples_seen": self.num_examples_seen,
            "num_classes_seen": self.num_classes_seen,
            "class_counts": self.class_counts,
            "n_models": self.n_models,
        }
