"""
Gaussian Mixture Model Clustering

Implements probabilistic clustering using Gaussian Mixture Models.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from typing import Optional

from .base import BaseClustering
from cont_src.core.registry import CLUSTERING_REGISTRY


@CLUSTERING_REGISTRY.register('gmm')
class GaussianMixtureClustering(BaseClustering):
    """
    Gaussian Mixture Model (GMM) clustering.

    Probabilistic clustering - each point has soft assignment to clusters.
    Can model ellipsoidal clusters of different sizes.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        covariance_type: str = 'full',
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Number of mixture components
            covariance_type: Type of covariance matrix
                           ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum EM iterations
            n_init: Number of initializations
            random_state: Random seed
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'GaussianMixtureClustering':
        """Fit GMM."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.labels_ = self.model.predict(embeddings)
        self.cluster_centers_ = self.model.means_
        self.is_fitted = True

        return self

    def predict(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        """Predict cluster labels (hard assignment)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Predict soft cluster assignments.

        Returns:
            probs: Probability matrix (N, K) where probs[i, k] = P(cluster k | point i)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.model.predict_proba(embeddings)

    def get_metadata(self) -> dict:
        """Get GMM-specific metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'covariance_type': self.covariance_type,
            'converged': bool(self.model.converged_),
            'n_iter': int(self.model.n_iter_),
            'bic': float(self.model.bic(self.model.means_)),
            'aic': float(self.model.aic(self.model.means_)),
        })
        return metadata


@CLUSTERING_REGISTRY.register('bayesian_gmm')
class BayesianGaussianMixtureClustering(BaseClustering):
    """
    Bayesian Gaussian Mixture Model.

    Automatically determines effective number of clusters through
    Bayesian inference. Can start with large n_clusters and prune.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        covariance_type: str = 'full',
        weight_concentration_prior_type: str = 'dirichlet_process',
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            n_clusters: Maximum number of components
            covariance_type: Type of covariance matrix
            weight_concentration_prior_type: Prior type ('dirichlet_process' or 'dirichlet_distribution')
            max_iter: Maximum EM iterations
            n_init: Number of initializations
            random_state: Random seed
        """
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.max_iter = max_iter
        self.n_init = n_init

        self.model = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            weight_concentration_prior_type=weight_concentration_prior_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )

    def fit(self, embeddings: torch.Tensor, **kwargs) -> 'BayesianGaussianMixtureClustering':
        """Fit Bayesian GMM."""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.model.fit(embeddings)
        self.labels_ = self.model.predict(embeddings)
        self.cluster_centers_ = self.model.means_

        # Determine effective number of clusters (weights > threshold)
        weights = self.model.weights_
        self.n_clusters = int((weights > 0.01).sum())
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

    def predict_proba(self, embeddings: torch.Tensor) -> np.ndarray:
        """Predict soft cluster assignments."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return self.model.predict_proba(embeddings)

    def get_metadata(self) -> dict:
        """Get Bayesian GMM metadata."""
        metadata = super().get_metadata()
        metadata.update({
            'covariance_type': self.covariance_type,
            'effective_n_clusters': self.n_clusters,
            'weights': self.model.weights_.tolist(),
            'converged': bool(self.model.converged_),
            'n_iter': int(self.model.n_iter_),
        })
        return metadata
