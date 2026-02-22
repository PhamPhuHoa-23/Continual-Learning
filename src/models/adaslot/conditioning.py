"""
Conditioning for slot initialization.

Mirrors: ocl.conditioning.RandomConditioning
Checkpoint key prefix: models.conditioning.slots_mu, models.conditioning.slots_logsigma
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


class RandomConditioning(nn.Module):
    """
    Random conditioning with learnable mean and stddev.

    Generates initial slot representations by sampling from a learned
    Gaussian distribution: slots = mu + sigma * N(0,1).

    Args:
        object_dim: Slot embedding dimension.
        n_slots: Number of slots.
        learn_mean: Whether to learn the mean (default True).
        learn_std: Whether to learn the stddev (default True).
    """

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        if mean_init is None:
            mean_init = nn.init.xavier_uniform_
        if logsigma_init is None:
            logsigma_init = nn.init.xavier_uniform_

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate initial slot embeddings.

        Args:
            batch_size: Batch size.

        Returns:
            slots: (batch_size, n_slots, object_dim)
        """
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)
