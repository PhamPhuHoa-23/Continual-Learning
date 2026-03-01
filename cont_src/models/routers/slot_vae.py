"""
SlotVAE: VAE for slot-to-agent routing with three scoring modes.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Each agent i owns one SlotVAE trained on its cluster of slots.
After training, the VAE is frozen. Only (μ_i, Σ_i) statistics may update
incrementally via Welford's algorithm when new slots are routed.

Purpose: answer "how likely is slot s_k to belong to agent i's sub-concept?"

═══════════════════════════════════════════════════════════════════════════════
THREE SCORING MODES
═══════════════════════════════════════════════════════════════════════════════

Mode A — Generative (prior sampling)  [DEFAULT, recommended]
─────────────────────────────────────
Idea (user's original design):
    1. Sample K latent codes from prior:  z_j ~ N(0, I)
    2. Decode each:  ŝ_j = decoder(z_j)   ← "example slots of sub-concept i"
    3. Measure similarity of query slot to all examples:
            score = mean_j sim(s_k, ŝ_j) / (std_j + ε)

Why mean/std?
    - mean high  → s_k resembles many examples → belongs to sub-concept i
    - std  low   → all examples agree          → confident match
    - mean high + std high → borderline, uncertain
    - mean low             → wrong sub-concept → don't route here

Inference cost: K decoder forwards per agent (no encoder needed).
Use: routing without encoder at inference; graceful with multimodal VAE.

─────────────────────────────────────────────────────────────────────────────
Mode B — Mahalanobis in z-space (encoder-based)
─────────────────────────────────────────────────
Idea:
    1. Encode query: z_q = μ_encoder(s_k)    ← posterior mean (no sampling)
    2. Mahalanobis distance to agent's latent statistics (μ_i, Σ_i):
            score = -(z_q - μ_i)^T Σ_i^{-1} (z_q - μ_i)

Why use z-space instead of slot-space?
    - KL loss during training forces z ~ N(0,I) → Gaussian assumption VALID
    - Even if slot distribution is non-Gaussian, z-space IS Gaussian
    - Mahalanobis is statistically correct only under Gaussian assumption

Inference cost: 1 encoder forward per slot (no decoder needed).
Welford tracks (μ_i, Σ_i) in z-space.

─────────────────────────────────────────────────────────────────────────────
Mode C — Mahalanobis in slot-space (no network)
─────────────────────────────────────────────────
Idea:
    Score(s_k, i) = -(s_k - μ_i)^T Σ_i^{-1} (s_k - μ_i)
    where (μ_i, Σ_i) tracked directly in slot-space via Welford.

Trade-off:
    - Zero neural-network cost at inference
    - Risk: if slot cluster is non-Gaussian, Mahalanobis is inaccurate
    - Practical OK for compact clusters (e.g. KMeans-initialized)

Inference cost: 0 forward passes.
Welford tracks (μ_i, Σ_i) directly in slot-space.

═══════════════════════════════════════════════════════════════════════════════
TRAINING
═══════════════════════════════════════════════════════════════════════════════

VAE loss  = L_recon  +  β · L_KL
          = MSE(ŝ, s)  +  β · KL(q(z|s) || N(0,I))

After train_vae() call:
    - Network weights frozen
    - Agent statistics initialized from training slots
    - Only Welford stats updated afterwards (via update_stats())

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Internal VAE network ─────────────────────────────────────────────────────

class _VAENetwork(nn.Module):
    """
    Tiny VAE network.  Encoder: slot_dim → latent_dim.  Decoder: latent_dim → slot_dim.

    Kept intentionally small (default 64→32) so that it stays lighter than the
    agent MLP it serves.  The decoder is only needed during training; at
    inference only the encoder (Mode B) or decoder (Mode A) is used.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        self.enc_backbone = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    # ── Core ops ─────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x → (μ_q, log σ²_q)   shape: (N, latent_dim) each."""
        h = self.enc_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick:  z = μ + ε·σ,  ε ~ N(0,I)."""
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → reconstructed slot.   shape: (N, input_dim)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}


# ── Welford online statistics ────────────────────────────────────────────────

class _WelfordStats:
    """
    Online mean + covariance via Welford's algorithm.
    Equivalent to batch computation but processes one sample at a time.
    Useful for streaming / exemplar-free updates.
    """

    def __init__(self, dim: int, shrinkage: float = 1e-4):
        self.dim = dim
        self.shrinkage = shrinkage
        self.n: int = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        # sum of outer products
        self._M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x: np.ndarray):
        """Add one sample (shape: (dim,))."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += np.outer(delta, delta2)

    def update_batch(self, X: np.ndarray):
        """Add multiple samples (shape: (N, dim))."""
        for x in X:
            self.update(x)

    @property
    def covariance(self) -> np.ndarray:
        if self.n < 2:
            return np.eye(self.dim) * self.shrinkage
        return self._M2 / (self.n - 1) + np.eye(self.dim) * self.shrinkage

    @property
    def inv_covariance(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            return np.eye(self.dim) / self.shrinkage

    def mahalanobis(self, x: np.ndarray) -> float:
        """(x - μ)^T Σ^{-1} (x - μ)  ≥ 0."""
        diff = x - self.mean
        return float(diff @ self.inv_covariance @ diff)


# ── Public SlotVAE class ──────────────────────────────────────────────────────

ScoringMode = Literal["generative", "mahal_z", "mahal_slot"]


class SlotVAE:
    """
    VAE-based density estimator for one agent's slot cluster.

    Usage (typical lifecycle)::

        vae = SlotVAE(slot_dim=64, latent_dim=16)

        # Phase 1: train on cluster slots (once, then frozen)
        vae.train_vae(cluster_slots, epochs=100, beta=1.0)

        # Phase 2: at inference, score an incoming slot
        score = vae.score(s_k, mode="generative", n_samples=50)

        # Optional: update statistics when slot is assigned
        vae.update_stats(s_k)

    Not a nn.Module; holds one internally.  Serialize with ``state_dict()``.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        shrinkage: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Args:
            slot_dim:    Input slot dimension D_s.
            latent_dim:  VAE latent dimension D_z  (recommended: D_s // 4).
            hidden_dims: MLP widths encoder/decoder (default: [32]).
            shrinkage:   Covariance diagonal regularisation.
            device:      Torch device for network ops.
        """
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        self.device = device

        hidden_dims = hidden_dims or [slot_dim // 2]

        self._net = _VAENetwork(slot_dim, latent_dim, hidden_dims).to(device)
        self._trained = False

        # Mode B (mahal_z): Welford in *latent* space
        self._stats_z = _WelfordStats(latent_dim,  shrinkage)
        # Mode C (mahal_slot): Welford in *slot* space
        self._stats_slot = _WelfordStats(slot_dim,    shrinkage)

    # ── Training ─────────────────────────────────────────────────────────────

    def train_vae(
        self,
        slots: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        beta: float = 1.0,
        verbose: bool = True,
    ):
        """
        Train ELBO on cluster slots, then freeze network.

        L = MSE(recon, slot)  +  β · KL(q(z|s) || N(0,I))

        After this call:
            - _net weights are frozen (requires_grad=False)
            - Welford stats (z-space and slot-space) are initialised

        Args:
            slots:      Training slots, shape (N, slot_dim).
            epochs:     Number of epochs.
            batch_size: Mini-batch size.
            lr:         Adam learning rate.
            beta:       KL weight (β-VAE; β=1 = standard VAE).
            verbose:    Print loss every 20 epochs.
        """
        slots = slots.to(self.device).float()
        loader = DataLoader(TensorDataset(
            slots), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)

        self._net.train()
        epoch_bar = tqdm(range(1, epochs + 1), desc="  VAE",
                         leave=False, unit="ep")
        for epoch in epoch_bar:
            epoch_loss = 0.0
            for (batch,) in loader:
                out = self._net(batch)
                l_recon = F.mse_loss(out["recon"], batch, reduction="mean")
                l_kl = -0.5 * \
                    (1 + out["logvar"] - out["mu"].pow(2) -
                     out["logvar"].exp()).sum(1).mean()
                loss = l_recon + beta * l_kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            epoch_bar.set_postfix(loss=f"{avg:.4f}")

        # ── Freeze network ────────────────────────────────────────────────────
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._trained = True

        # ── Initialise Welford statistics ─────────────────────────────────────
        with torch.no_grad():
            mu, _ = self._net.encode(slots)
            latents = mu.cpu().numpy()                # z-space
        slots_np = slots.cpu().numpy()                # slot-space

        self._stats_z.update_batch(latents)
        self._stats_slot.update_batch(slots_np)

    # ── Scoring ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(
        self,
        slots: torch.Tensor,
        mode: ScoringMode = "generative",
        n_samples: int = 50,
    ) -> torch.Tensor:
        """
        Compute routing score for one or more slots.

        Higher score  →  slot more likely belongs to this agent's sub-concept.

        Args:
            slots:     Shape (D_s,) or (N, D_s).
            mode:      Scoring strategy — see module docstring.
                       ``"generative"``   prior sampling (Mode A)
                       ``"mahal_z"``      Mahalanobis in z-space (Mode B)
                       ``"mahal_slot"``   Mahalanobis in slot-space (Mode C)
            n_samples: Number of prior samples for Mode A.

        Returns:
            Scalar tensor per slot, shape () or (N,).
        """
        single = slots.dim() == 1
        if single:
            slots = slots.unsqueeze(0)   # (1, D_s)
        slots = slots.to(self.device).float()

        if mode == "generative":
            scores = self._score_generative(slots, n_samples)
        elif mode == "mahal_z":
            scores = self._score_mahal_z(slots)
        elif mode == "mahal_slot":
            scores = self._score_mahal_slot(slots)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose: generative | mahal_z | mahal_slot")

        return scores.squeeze(0) if single else scores

    # ── Mode A: generative (prior sampling) ──────────────────────────────────

    def _score_generative(self, slots: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Mode A — generative prior sampling.

        Algorithm:
            z_j ~ N(0, I)  for j=1..K
            ŝ_j = decoder(z_j)
            sim_j = cosine_similarity(s_k, ŝ_j)
            score = mean(sim) / (std(sim) + ε)

        Intuition:
            mean/std is a signal-to-noise ratio over random "examples" of the
            sub-concept.  High mean = slot looks like this concept.
            Low std = VAE generates consistent examples → confident.

        Note: no encoder forward at inference.  Only decoder used.
        """
        N = slots.shape[0]

        # Sample K latent codes from prior N(0,I)
        z = torch.randn(n_samples, self.latent_dim,
                        device=self.device)    # (K, D_z)
        # (K, D_s)
        examples = self._net.decode(z)

        # Cosine similarity: each slot vs all K examples → (N, K)
        slots_norm = F.normalize(slots,    dim=-1)   # (N, D_s)
        examples_norm = F.normalize(examples, dim=-1)   # (K, D_s)
        sims = slots_norm @ examples_norm.T              # (N, K)

        # Mean / std  → SNR score per slot  (N,)
        mean = sims.mean(dim=1)
        std = sims.std(dim=1)
        return mean / (std + 1e-6)

    # ── Mode B: Mahalanobis in z-space ───────────────────────────────────────

    def _score_mahal_z(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Mode B — Mahalanobis distance in VAE latent space.

        Algorithm:
            z_q = μ_encoder(s_k)                 ← posterior mean (no sampling)
            d   = (z_q - μ_i)^T Σ_i^{-1} (z_q - μ_i)
            score = -d

        Gaussian assumption holds in z-space because KL loss enforces it.
        Encoder forward is needed at inference.
        """
        mu, _ = self._net.encode(slots)       # (N, D_z)
        z_np = mu.cpu().numpy()              # (N, D_z)
        scores = np.array([-self._stats_z.mahalanobis(z) for z in z_np])
        return torch.tensor(scores, dtype=torch.float32)

    # ── Mode C: Mahalanobis in slot-space ────────────────────────────────────

    def _score_mahal_slot(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Mode C — Mahalanobis distance directly in slot space.

        No neural network involved at inference.
        Risk: if slot cluster is non-Gaussian, distance is approximate.
        Practical for compact KMeans-initialized clusters.
        """
        s_np = slots.cpu().numpy()          # (N, D_s)
        scores = np.array([-self._stats_slot.mahalanobis(s) for s in s_np])
        return torch.tensor(scores, dtype=torch.float32)

    # ── Incremental statistics update ─────────────────────────────────────────

    @torch.no_grad()
    def update_stats(self, slots: torch.Tensor):
        """
        Welford-update both z-space and slot-space statistics with new slots.

        Called when slot(s) are routed to this agent (Tasks 2+).
        Network stays frozen; only running mean/cov are updated.

        Args:
            slots: Shape (D_s,) or (N, D_s).
        """
        if slots.dim() == 1:
            slots = slots.unsqueeze(0)
        slots = slots.to(self.device).float()

        mu, _ = self._net.encode(slots)
        self._stats_z.update_batch(mu.cpu().numpy())
        self._stats_slot.update_batch(slots.cpu().numpy())

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialise full state (network + statistics)."""
        return {
            "net": {k: v.cpu() for k, v in self._net.state_dict().items()},
            "trained": self._trained,
            "stats_z": {
                "n": self._stats_z.n,
                "mean": self._stats_z.mean.copy(),
                "M2": self._stats_z._M2.copy(),
            },
            "stats_slot": {
                "n": self._stats_slot.n,
                "mean": self._stats_slot.mean.copy(),
                "M2": self._stats_slot._M2.copy(),
            },
        }

    def load_state_dict(self, state: dict):
        """Restore from serialised state."""
        self._net.load_state_dict({k: v.to(self.device)
                                  for k, v in state["net"].items()})
        self._trained = state["trained"]
        if self._trained:
            for p in self._net.parameters():
                p.requires_grad_(False)
            self._net.eval()

        for attr, key in [("_stats_z", "stats_z"), ("_stats_slot", "stats_slot")]:
            stats = getattr(self, attr)
            s = state[key]
            stats.n = s["n"]
            stats.mean = s["mean"].copy()
            stats._M2 = s["M2"].copy()

    def __repr__(self) -> str:
        return (
            f"SlotVAE(slot_dim={self.slot_dim}, latent_dim={self.latent_dim}, "
            f"trained={self._trained}, n_z={self._stats_z.n})"
        )
