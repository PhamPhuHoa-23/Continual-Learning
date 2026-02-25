"""
Clustering Module

Provides various clustering algorithms for slot/embedding clustering:
- K-means (fixed clusters)
- DBSCAN (density-based, auto clusters)
- HDBSCAN (hierarchical density-based)
- Agglomerative (hierarchical)
- Gaussian Mixture (probabilistic)
"""

from cont_src.core.registry import CLUSTERING_REGISTRY

# Import base classes
from .base import BaseClustering, ClusteringResult
from .embedding_wrapper import (
    BaseEmbeddingWrapper,
    IdentityWrapper,
    LinearProjectionWrapper,
    NonLinearWrapper,
    SubmanifoldWrapper,
)

# Import clustering algorithms
from .kmeans import KMeansClustering, MiniBatchKMeansClustering
from .density_based import DBSCANClustering, HDBSCANClustering
from .hierarchical import AgglomerativeClustering
from .gaussian_mixture import GaussianMixtureClustering

__all__ = [
    # Registry
    'CLUSTERING_REGISTRY',

    # Base classes
    'BaseClustering',
    'ClusteringResult',

    # Embedding wrappers
    'BaseEmbeddingWrapper',
    'IdentityWrapper',
    'LinearProjectionWrapper',
    'NonLinearWrapper',
    'SubmanifoldWrapper',

    # Algorithms
    'KMeansClustering',
    'MiniBatchKMeansClustering',
    'DBSCANClustering',
    'HDBSCANClustering',
    'AgglomerativeClustering',
    'GaussianMixtureClustering',
]
