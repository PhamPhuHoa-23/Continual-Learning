"""
VAE-based router for slot-to-agent assignment.

Uses Mahalanobis distance in VAE latent space for routing decisions.
VAE network is frozen, but latent statistics (μ, Σ) are updated incrementally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from cont_src.core.base_module import BaseRouter
from cont_src.core.registry import ROUTER_REGISTRY


class VAE(nn.Module):
    """Simple VAE for slot encoding."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = [128, 64]
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Latent parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent parameters."""
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {
            "reconstruction": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


class WelfordStats:
    """
    Welford's online algorithm for mean and covariance.

    Allows incremental updates without storing all data.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))  # Sum of squared differences

    def update(self, x: np.ndarray):
        """Update with new sample."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

    def get_mean(self) -> np.ndarray:
        """Get current mean."""
        return self.mean

    def get_covariance(self, shrinkage: float = 1e-4) -> np.ndarray:
        """Get current covariance with shrinkage."""
        if self.n < 2:
            return np.eye(self.dim) * shrinkage
        cov = self.M2 / (self.n - 1)
        # Add shrinkage for numerical stability
        cov += np.eye(self.dim) * shrinkage
        return cov


@ROUTER_REGISTRY.register("vae")
class VAERouter(BaseRouter):
    """
    VAE-based router using Mahalanobis distance.

    Key design:
    - VAE network is frozen after training
    - Only latent statistics (μ_i, Σ_i) are updated per agent
    - Routing score = negative Mahalanobis distance
    """

    def __init__(
        self,
        input_dim: int = 64,
        latent_dim: int = 32,
        hidden_dims: list = None,
        threshold_match: float = -10.0,
        threshold_novel: float = -50.0,
        shrinkage: float = 1e-4,
        **kwargs
    ):
        """
        Initialize VAE router.

        Args:
            input_dim: Slot dimension
            latent_dim: VAE latent dimension
            hidden_dims: Hidden layer dimensions
            threshold_match: Threshold for matching to existing agent
            threshold_novel: Threshold below which slot is novel
            shrinkage: Covariance shrinkage for numerical stability
        """
        super().__init__(config={
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims or [128, 64],
            "threshold_match": threshold_match,
            "threshold_novel": threshold_novel,
            "shrinkage": shrinkage,
        })

        self.latent_dim = latent_dim
        self.threshold_match = threshold_match
        self.threshold_novel = threshold_novel
        self.shrinkage = shrinkage

        # VAE network (will be frozen)
        self.vae = VAE(input_dim, latent_dim, hidden_dims or [128, 64])

        # Per-agent latent statistics (updated incrementally)
        self.agent_stats = {}  # agent_id -> WelfordStats

    def train_vae(
        self,
        slots: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cuda"
    ):
        """
        Train VAE on slots.

        Args:
            slots: Slot data, shape (N, D)
            epochs: Training epochs
            lr: Learning rate
            device: Training device
        """
        self.vae.to(device)
        self.vae.train()

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(slots)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for (batch_slots,) in loader:
                batch_slots = batch_slots.to(device)

                # Forward
                output = self.vae(batch_slots)

                # Reconstruction loss
                recon_loss = F.mse_loss(
                    output["reconstruction"], batch_slots, reduction="mean"
                )

                # KL divergence
                kl_loss = -0.5 * torch.sum(
                    1 + output["logvar"] -
                    output["mu"].pow(2) - output["logvar"].exp()
                )
                kl_loss = kl_loss / batch_slots.size(0)

                # Total loss
                loss = recon_loss + 0.1 * kl_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def init_agent_stats(self, agent_id: int, slots: torch.Tensor):
        """
        Initialize statistics for a new agent.

        Args:
            agent_id: Agent ID
            slots: Initial slots for this agent, shape (N, D)
        """
        # Encode slots to latent space
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(slots)
            latents = mu.cpu().numpy()  # Use mean, not sampled z

        # Initialize Welford stats
        stats = WelfordStats(self.latent_dim)
        for latent in latents:
            stats.update(latent)

        self.agent_stats[agent_id] = stats

    def compute_scores(
        self,
        slot: torch.Tensor,
        agent_ids: Optional[list] = None
    ) -> torch.Tensor:
        """
        Compute routing scores using Mahalanobis distance.

        Args:
            slot: Single slot, shape (D,)
            agent_ids: Agent IDs to score. If None, use all agents.

        Returns:
            Scores for each agent, shape (N_agents,)
            Higher score = better match
        """
        if agent_ids is None:
            agent_ids = list(self.agent_stats.keys())

        if len(agent_ids) == 0:
            return torch.tensor([])

        # Encode slot to latent space
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(slot.unsqueeze(0))
            z = mu.squeeze(0).cpu().numpy()  # (latent_dim,)

        # Compute Mahalanobis distance to each agent
        scores = []
        for agent_id in agent_ids:
            stats = self.agent_stats[agent_id]

            mean = stats.get_mean()
            cov = stats.get_covariance(shrinkage=self.shrinkage)

            # Mahalanobis distance: (z - μ)^T Σ^(-1) (z - μ)
            diff = z - mean
            try:
                inv_cov = np.linalg.inv(cov)
                mahal_dist = diff @ inv_cov @ diff
            except np.linalg.LinAlgError:
                # Fallback to large distance if singular
                mahal_dist = 1e6

            # Negative distance as score (higher = better match)
            score = -mahal_dist
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32)

    def update_stats(self, agent_id: int, slot: torch.Tensor):
        """
        Update agent statistics with new slot.

        Args:
            agent_id: Agent ID
            slot: Slot routed to this agent, shape (D,)
        """
        # Encode to latent
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(slot.unsqueeze(0))
            z = mu.squeeze(0).cpu().numpy()

        # Update stats
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = WelfordStats(self.latent_dim)

        self.agent_stats[agent_id].update(z)

    def forward(self, slot: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Route slot to best agent.

        Args:
            slot: Single slot, shape (D,)

        Returns:
            Dict with:
                - agent_id: Best matching agent ID (or -1 for novel)
                - score: Routing score
                - all_scores: Scores for all agents
        """
        scores = self.compute_scores(slot)

        if len(scores) == 0:
            # No agents yet - novel slot
            return {
                "agent_id": torch.tensor(-1),
                "score": torch.tensor(float("-inf")),
                "all_scores": scores,
            }

        best_score, best_idx = scores.max(dim=0)

        # Check thresholds
        if best_score >= self.threshold_match:
            agent_id = best_idx
        elif best_score < self.threshold_novel:
            agent_id = torch.tensor(-1)  # Novel
        else:
            agent_id = torch.tensor(-2)  # Uncertain (buffer)

        return {
            "agent_id": agent_id,
            "score": best_score,
            "all_scores": scores,
        }
