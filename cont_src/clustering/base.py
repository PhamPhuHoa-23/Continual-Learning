"""
Base Clustering Classes

Defines the interface for all clustering algorithms.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ClusteringResult:
    """
    Result from clustering algorithm.

    Attributes:
        labels: Cluster labels for each sample (N,)
        centers: Cluster centers (K, D) or None
        n_clusters: Number of clusters found
        scores: Optional clustering quality scores
        metadata: Additional algorithm-specific information
    """
    labels: np.ndarray
    centers: Optional[np.ndarray] = None
    n_clusters: int = 0
    scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.n_clusters == 0:
            self.n_clusters = len(np.unique(self.labels[self.labels >= 0]))


class BaseClustering(nn.Module):
    """
    Base class for clustering algorithms.

    All clustering algorithms should inherit from this class and implement
    the fit(), predict(), and fit_predict() methods.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Number of clusters (None for auto-detect algorithms)
            random_state: Random seed for reproducibility
            **kwargs: Algorithm-specific parameters
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.is_fitted = False
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> 'BaseClustering':
        """
        Fit the clustering algorithm.

        Args:
            embeddings: Input embeddings (N, D)
            **kwargs: Additional fit parameters

        Returns:
            self
        """
        raise NotImplementedError

    def predict(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        """
        Predict cluster labels for new embeddings.

        Args:
            embeddings: Input embeddings (N, D)
            **kwargs: Additional predict parameters

        Returns:
            labels: Cluster labels (N,)
        """
        raise NotImplementedError

    def fit_predict(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> ClusteringResult:
        """
        Fit and predict in one step.

        Args:
            embeddings: Input embeddings (N, D)
            **kwargs: Additional parameters

        Returns:
            ClusteringResult with labels and centers
        """
        self.fit(embeddings, **kwargs)
        labels = self.predict(embeddings, **kwargs)

        return ClusteringResult(
            labels=labels,
            centers=self.cluster_centers_,
            n_clusters=self.n_clusters if self.n_clusters else len(
                np.unique(labels[labels >= 0])),
            scores=self.get_scores(embeddings, labels),
            metadata=self.get_metadata()
        )

    def get_scores(
        self,
        embeddings: torch.Tensor,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute clustering quality scores.

        Args:
            embeddings: Input embeddings (N, D)
            labels: Cluster labels (N,)

        Returns:
            Dictionary of scores (silhouette, davies_bouldin, etc.)
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Filter out noise points (-1 labels)
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            return {}

        embeddings_valid = embeddings[valid_mask]
        labels_valid = labels[valid_mask]

        # Compute scores
        scores = {}

        try:
            if len(np.unique(labels_valid)) > 1:
                scores['silhouette'] = float(
                    silhouette_score(embeddings_valid, labels_valid))
                scores['davies_bouldin'] = float(
                    davies_bouldin_score(embeddings_valid, labels_valid))
                scores['calinski_harabasz'] = float(
                    calinski_harabasz_score(embeddings_valid, labels_valid))
        except Exception as e:
            print(f"Warning: Could not compute clustering scores: {e}")

        return scores

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get algorithm-specific metadata.

        Returns:
            Dictionary of metadata
        """
        return {
            'algorithm': self.__class__.__name__,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted,
        }

    def to_device(self, device: torch.device):
        """Move clustering model to device (if needed)."""
        return self.to(device)
