"""
VAE (Variational Autoencoder) for uncertainty estimation.

VAE provides a simple but effective way to estimate uncertainty:
- High reconstruction error -> high uncertainty (sample is unusual/difficult)
- Low reconstruction error -> low uncertainty (sample is familiar)

This is one of the simplest ideas for resource-constrained continual learning.
"""

from .vae import VAE, ConvVAE
from .uncertainty import VAEUncertaintyEstimator

__all__ = [
    'VAE',
    'ConvVAE',
    'VAEUncertaintyEstimator',
]

