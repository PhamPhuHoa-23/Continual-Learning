"""
Density-Based Clustering Algorithms

Implements DBSCAN and HDBSCAN - automatically detect number of clusters.
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from typing import Optional

from .base import BaseClustering
from cont_src.core.registry import CLUSTERING_REGISTRY


@CLUSTERING_REGISTRY.register('dbscan')
class DBSCANClustering(BaseClustering):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Automatically detects number of clusters based on density.
    Can identify outliers (labeled as -1).
    Does NOT require knowing number of clusters in advance.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            eps: Maximum distance between two samples to be neighbors
            min_samples: Minimum samples in a neighborhood for core point
            metric: Distance metric ('euclidean', 'cosine', etc.)
            random_state: Random seed (not used by DBSCAN but kept for consistency)
        """
        super().__init__(n_clusters=None, random_state=random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'DBSCANClustering':
        """Fit DBSCAN."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.labels_ = self.model.fit_predict(embeddings)
        self.is_fitted = True

        # Compute cluster centers (excluding noise points)
        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        if len(unique_labels) > 0:
            centers = []
            for label in unique_labels:
                mask = self.labels_ == label
                centers.append(embeddings[mask].mean(axis=0))
            self.cluster_centers_ = np.array(centers)
            self.n_clusters = len(unique_labels)
        else:
            self.cluster_centers_ = None
            self.n_clusters = 0

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: DBSCAN doesn't have a native predict method.
        We assign new points to nearest cluster center or mark as noise.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if self.cluster_centers_ is None or len(self.cluster_centers_) == 0:
            return np.full(len(embeddings), -1)  # All noise

        # Assign to nearest cluster or noise
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, self.cluster_centers_,
                          metric=self.metric)

        # Points within eps of a center get assigned, others are noise
        min_distances = distances.min(axis=1)
        labels = distances.argmin(axis=1)
        labels[min_distances > self.eps] = -1  # Mark as noise

        return labels

    def get_metadata(self) -> dict:
        """Get DBSCAN-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'n_noise': int((self.labels_ == -1).sum()) if self.labels_ is not None else 0,
        })
        return metadata


@CLUSTERING_REGISTRY.register('hdbscan')
class HDBSCANClustering(BaseClustering):
    """
    HDBSCAN (Hierarchical DBSCAN).

    More robust than DBSCAN - automatically selects eps parameter.
    Can better handle varying density clusters.
    Requires hdbscan package: pip install hdbscan
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood (default: min_cluster_size)
            metric: Distance metric
            cluster_selection_method: 'eom' (excess of mass) or 'leaf'
            random_state: Random seed
        """
        super().__init__(n_clusters=None, random_state=random_state)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        try:
            import hdbscan
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "hdbscan not installed. Install with: pip install hdbscan"
            )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'HDBSCANClustering':
        """Fit HDBSCAN."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.labels_ = self.model.labels_
        self.is_fitted = True

        # Compute cluster centers (excluding noise points)
        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        if len(unique_labels) > 0:
            centers = []
            for label in unique_labels:
                mask = self.labels_ == label
                # Weight by exemplar scores if available
                if hasattr(self.model, 'exemplars_'):
                    centers.append(embeddings[mask].mean(axis=0))
                else:
                    centers.append(embeddings[mask].mean(axis=0))
            self.cluster_centers_ = np.array(centers)
            self.n_clusters = len(unique_labels)
        else:
            self.cluster_centers_ = None
            self.n_clusters = 0

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Uses approximate_predict from hdbscan if available.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Try approximate_predict if available
        if hasattr(self.model, 'approximate_predict'):
            labels, _ = self.model.approximate_predict(embeddings)
            return labels
        else:
            # Fallback: assign to nearest cluster
            if self.cluster_centers_ is None or len(self.cluster_centers_) == 0:
                return np.full(len(embeddings), -1)

            from scipy.spatial.distance import cdist
            distances = cdist(
                embeddings, self.cluster_centers_, metric=self.metric)
            return distances.argmin(axis=1)

    def get_metadata(self) -> dict:
        """Get HDBSCAN-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'n_noise': int((self.labels_ == -1).sum()) if self.labels_ is not None else 0,
        })
        return metadata
