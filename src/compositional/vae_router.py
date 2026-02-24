"""
VAE Router: per-agent VAE networks + Welford latent statistics for routing.

Routing score (paper eq. 3):
    score(s, i) = -(mu_q(s) - mu_i)^T  Sigma_i^{-1}  (mu_q(s) - mu_i)

VAE network is frozen after initial training per agent.
Only (mu_i, Sigma_i) are updated incrementally via Welford's algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from tqdm import tqdm

from src.models.vae.vae import VAE


class VAERouter(nn.Module):
    """
    Pool of per-agent VAEs with Welford latent statistics.

    The VAE *network* is frozen after training.
    The latent *statistics* (mu_i, Sigma_i) are updated online via Welford
    whenever new slots are routed to agent i.

    Args:
        slot_dim:   Input slot dimension (= VAE input_dim)
        latent_dim: VAE latent dimension
    """

    def __init__(self, slot_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim

        # VAE networks (frozen after per-agent training)
        self.vaes: nn.ModuleDict = nn.ModuleDict()

        # Welford stats per agent: count, mu, M2 (running sum-of-sq-diffs), Sigma_inv
        # Stored as plain dicts of tensors (NOT Parameters) so they can be updated freely.
        self.stats: dict = {}

    # ── agent lifecycle ───────────────────────────────────────────────────────

    def spawn_vae(self, agent_id: str, device: torch.device) -> None:
        """Create a new (untrained) VAE for agent_id."""
        vae = VAE(input_dim=self.slot_dim, latent_dim=self.latent_dim,
                  hidden_dims=[256, 128])
        vae.to(device)
        self.vaes[agent_id] = vae

        self.stats[agent_id] = {
            "count":     0,
            "mu":        torch.zeros(self.latent_dim, device=device),
            "M2":        torch.zeros(self.latent_dim, self.latent_dim, device=device),
            "Sigma_inv": torch.eye(self.latent_dim, device=device),
        }

    def train_vae(
        self,
        agent_id: str,
        slot_batches: List[torch.Tensor],   # list of (B, slot_dim) tensors
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> None:
        """
        Train the VAE for agent_id on the provided slot batches, then freeze it.

        Args:
            agent_id:     Which VAE to train
            slot_batches: List of slot tensors (each (B, slot_dim))
            epochs:       Training epochs
            lr:           Learning rate
        """
        vae = self.vaes[agent_id]
        device = next(vae.parameters()).device
        vae.train()
        optimizer = optim.Adam(vae.parameters(), lr=lr)

        epoch_bar = tqdm(range(epochs), desc=f"  VAE {agent_id}", leave=False, unit="ep")
        for epoch in epoch_bar:
            total_loss = 0.0
            n_batches = 0
            for batch in slot_batches:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = vae(batch)
                loss_dict = vae.compute_loss(
                    batch, out["recon"], out["mu"], out["logvar"]
                )
                loss_dict["loss"].backward()
                optimizer.step()
                total_loss += loss_dict["loss"].item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            epoch_bar.set_postfix(loss=f"{avg_loss:.2f}")
            
            # Log every 20% of training
            if epochs >= 100 and (epoch + 1) % (epochs // 5) == 0:
                epoch_bar.write(f"    [{agent_id}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.2f}")

        # Freeze network — only stats will be updated going forward
        for p in vae.parameters():
            p.requires_grad_(False)
        vae.eval()

    # ── statistics update (Welford) ───────────────────────────────────────────

    @torch.no_grad()
    def update_stats(self, agent_id: str, mu_q: torch.Tensor) -> None:
        """
        Welford online update of (mu_i, Sigma_i) from a batch of posterior means.

        Args:
            agent_id: Which agent's stats to update
            mu_q:     (B, latent_dim) — VAE posterior means for routed slots
        """
        st = self.stats[agent_id]
        for i in range(mu_q.size(0)):
            x = mu_q[i]
            st["count"] += 1
            n = st["count"]
            delta = x - st["mu"]
            st["mu"] += delta / n
            delta2 = x - st["mu"]
            st["M2"] += torch.outer(delta, delta2)

        if st["count"] > 1:
            Sigma = st["M2"] / (st["count"] - 1)
            Sigma += torch.eye(self.latent_dim, device=Sigma.device) * 1e-5
            st["Sigma_inv"] = torch.linalg.inv(Sigma)

    # ── scoring & routing ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_score(self, s: torch.Tensor, agent_id: str) -> torch.Tensor:
        """
        Negative Mahalanobis distance score for slot(s) w.r.t. agent_id.

        Args:
            s: (B, slot_dim)
        Returns:
            scores: (B,)  — higher is better match
        """
        vae = self.vaes[agent_id]
        st = self.stats[agent_id]
        mu_q, _ = vae.encode(s)                        # (B, latent_dim)
        diff = mu_q - st["mu"].unsqueeze(0)            # (B, L)
        left = diff @ st["Sigma_inv"]                  # (B, L)
        return -(left * diff).sum(dim=1)               # (B,)

    @torch.no_grad()
    def route(
        self,
        s: torch.Tensor,
        theta_match: float,
        theta_novel: float,
    ) -> list:
        """
        Route a batch of slots.

        Returns:
            List[str] of length B — each element is an agent_id or 'unassigned'.
        """
        B = s.size(0)
        if not self.vaes:
            return ["unassigned"] * B

        agent_ids = list(self.vaes.keys())
        scores = torch.stack(
            [self.compute_score(s, aid) for aid in agent_ids], dim=1
        )                                               # (B, num_agents)

        max_scores, best_idx = scores.max(dim=1)       # (B,), (B,)

        result = []
        for b in range(B):
            if max_scores[b] >= theta_match:
                result.append(agent_ids[best_idx[b].item()])
            else:
                result.append("unassigned")
        return result
