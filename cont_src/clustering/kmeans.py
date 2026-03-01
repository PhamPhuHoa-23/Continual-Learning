"""
K-Means Clustering Algorithms

Implements K-means and variants with fixed number of clusters.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Optional

from .base import BaseClustering
from cont_src.core.registry import CLUSTERING_REGISTRY


@CLUSTERING_REGISTRY.register('kmeans')
class KMeansClustering(BaseClustering):
    """
    Standard K-means clustering.

    Requires knowing the number of clusters in advance.
    Good for when you have prior knowledge about cluster count.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            n_init: Number of initializations
            random_state: Random seed
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.max_iter = max_iter
        self.n_init = n_init

        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'KMeansClustering':
        """Fit K-means."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.is_fitted = True

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.model.predict(embeddings)


@CLUSTERING_REGISTRY.register('minibatch_kmeans')
class MiniBatchKMeansClustering(BaseClustering):
    """
    Mini-batch K-means clustering.

    More efficient than standard K-means for large datasets.
    Trades some accuracy for speed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        batch_size: int = 1024,
        n_init: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            batch_size: Size of mini-batches
            n_init: Number of initializations
            random_state: Random seed
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_init = n_init

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'MiniBatchKMeansClustering':
        """Fit mini-batch K-means."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.is_fitted = True

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.model.predict(embeddings)
