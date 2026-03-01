"""
Hierarchical Clustering Algorithms

Implements agglomerative clustering with different linkage methods.
"""

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from typing import Optional

from .base import BaseClustering
from cont_src.core.registry import CLUSTERING_REGISTRY


@CLUSTERING_REGISTRY.register('agglomerative')
class AgglomerativeClustering(BaseClustering):
    """
    Agglomerative (hierarchical) clustering.

    Builds a hierarchy of clusters by iteratively merging.
    Can specify number of clusters OR distance threshold for auto-detection.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = 8,
        distance_threshold: Optional[float] = None,
        linkage: str = 'ward',
        metric: str = 'euclidean',
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Number of clusters (None if using distance_threshold)
            distance_threshold: Distance threshold for auto cluster detection
                               (n_clusters must be None)
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            metric: Distance metric (used with linkage != 'ward')
            random_state: Random seed (not used but kept for consistency)
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)

        # Can't specify both n_clusters and distance_threshold
        if n_clusters is not None and distance_threshold is not None:
            raise ValueError(
                "Cannot specify both n_clusters and distance_threshold")

        if n_clusters is None and distance_threshold is None:
            raise ValueError(
                "Must specify either n_clusters or distance_threshold")

        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric

        # Ward linkage requires euclidean metric
        if linkage == 'ward' and metric != 'euclidean':
            raise ValueError("Ward linkage requires euclidean metric")

        self.model = SklearnAgglomerative(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage,
            metric=metric,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'AgglomerativeClustering':
        """Fit agglomerative clustering."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.labels_ = self.model.labels_
        self.is_fitted = True

        # Update n_clusters if using distance_threshold
        if self.distance_threshold is not None:
            self.n_clusters = self.model.n_clusters_

        # Compute cluster centers
        unique_labels = np.unique(self.labels_)
        centers = []
        for label in unique_labels:
            mask = self.labels_ == label
            centers.append(embeddings[mask].mean(axis=0))
        self.cluster_centers_ = np.array(centers)

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: Agglomerative clustering doesn't have native predict.
        We assign new points to nearest cluster center.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if self.cluster_centers_ is None:
            raise RuntimeError("No cluster centers found")

        # Assign to nearest cluster
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, self.cluster_centers_,
                          metric=self.metric)
        return distances.argmin(axis=1)

    def get_metadata(self) -> dict:
        """Get agglomerative clustering metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'linkage': self.linkage,
            'metric': self.metric,
            'distance_threshold': self.distance_threshold,
        })
        return metadata
