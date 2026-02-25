"""
SLDA Trainer  –  Phase C
=========================
Incremental closed-form fitting of a Stream Linear Discriminant Analysis
(SLDA) classifier over the *frozen* pipeline's output features H.

No gradient steps — this is a one-pass analytical update:
    μ_c  ←  running class mean
    Σ    ←  running shared covariance  (Welford + ridge shrinkage)

At inference, class is predicted by:
    ŷ = argmax_c  [ H^T Σ^{-1} μ_c  -  ½ μ_c^T Σ^{-1} μ_c + log π_c ]

Usage
-----
    from cont_src.training.configs      import SLDAConfig
    from cont_src.training.slda_trainer import SLDATrainer, StreamLDA

    slda    = StreamLDA(n_classes=100, feature_dim=64)
    cfg     = SLDAConfig(feature_dim=64, n_classes=100)
    trainer = SLDATrainer(cfg, slot_model, agents, aggregator, slda)
    trainer.fit(train_loader)
    preds   = trainer.predict(test_loader)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from cont_src.training.configs import SLDAConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StreamLDA: online mean + covariance
# ---------------------------------------------------------------------------

class StreamLDA:
    """
    Incremental SLDA accumulator.

    Maintains:
        n_c      : class counts           dict[int → int]
        mu_c     : class means            dict[int → Tensor(D,)]
        S        : shared scatter matrix  Tensor(D, D)   (Welford)
        n_total  : total samples seen
    """

    def __init__(self, n_classes: int, feature_dim: int, shrinkage: float = 1e-4):
        self.n_classes   = n_classes
        self.feature_dim = feature_dim
        self.shrinkage   = shrinkage

        self._n_c:     Dict[int, int]            = {}
        self._mu_c:    Dict[int, torch.Tensor]   = {}
        self._S:       torch.Tensor              = torch.zeros(feature_dim, feature_dim)
        self._n_total: int                       = 0

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update statistics with a new batch.

        Parameters
        ----------
        features : (B, D) – L2-normalised recommended but not required
        labels   : (B,)   – integer class indices
        """
        features = features.detach().cpu().float()
        labels   = labels.detach().cpu()

        for feat, lbl in zip(features, labels):
            c = int(lbl.item())

            # --- update class mean (online) ---
            if c not in self._n_c:
                self._n_c[c]  = 0
                self._mu_c[c] = torch.zeros(self.feature_dim)

            n_old       = self._n_c[c]
            n_new       = n_old + 1
            delta       = feat - self._mu_c[c]
            self._mu_c[c] = self._mu_c[c] + delta / n_new
            self._n_c[c]  = n_new

            # --- update scatter matrix (Welford) ---
            delta2   = feat - self._mu_c[c]
            self._S  = self._S + torch.outer(delta, delta2)
            self._n_total += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @property
    def cov(self) -> torch.Tensor:
        """Regularised shared covariance matrix."""
        if self._n_total <= 1:
            return torch.eye(self.feature_dim) * (self.shrinkage + 1.0)
        cov = self._S / (self._n_total - 1)
        cov = cov + self.shrinkage * torch.eye(self.feature_dim)
        return cov

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels using LDA decision rule.

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        preds : (B,) LongTensor
        """
        if not self._mu_c:
            raise RuntimeError("StreamLDA has not been updated with any data yet.")

        features = features.detach().cpu().float()

        cov_inv = torch.linalg.inv(self.cov)             # (D, D)

        # Gather class means into (C, D)
        classes  = sorted(self._mu_c.keys())
        mu_stack = torch.stack([self._mu_c[c] for c in classes], dim=0)  # (C, D)

        # Scores:  H @ Σ^{-1} @ μ_c^T  -  ½ μ_c @ Σ^{-1} @ μ_c^T
        # (B, D) @ (D, D) → (B, D) @ (D, C) → (B, C)
        h_proj   = features @ cov_inv                           # (B, D)
        scores   = h_proj @ mu_stack.t()                        # (B, C)
        quadform = 0.5 * (mu_stack @ cov_inv * mu_stack).sum(dim=1)   # (C,)
        scores   = scores - quadform.unsqueeze(0)

        idx   = scores.argmax(dim=1)                            # (B,)
        preds = torch.tensor([classes[i] for i in idx.tolist()])
        return preds

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "n_c":     self._n_c,
            "mu_c":    {k: v.clone() for k, v in self._mu_c.items()},
            "S":       self._S.clone(),
            "n_total": self._n_total,
        }

    def load_state_dict(self, state: dict) -> None:
        self._n_c     = state["n_c"]
        self._mu_c    = {k: v.clone() for k, v in state["mu_c"].items()}
        self._S       = state["S"].clone()
        self._n_total = state["n_total"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, n_classes: int, feature_dim: int, shrinkage: float = 1e-4) -> "StreamLDA":
        obj = cls(n_classes, feature_dim, shrinkage)
        obj.load_state_dict(torch.load(path, weights_only=False))
        return obj


# ---------------------------------------------------------------------------
# SLDA Trainer
# ---------------------------------------------------------------------------

class SLDATrainer:
    """
    Drives the SLDA fitting phase.  Not a subclass of BaseTrainer because
    there are no gradient steps — no optimizer is needed.

    Parameters
    ----------
    config : SLDAConfig
    slot_model : nn.Module
        Frozen backbone.
    agents : List[nn.Module]
        Frozen agents.
    aggregator : nn.Module
        Frozen aggregator that maps (B, K, D_h) → (B, D_h).
    slda : StreamLDA
        The SLDA accumulator to update.
    vaes : list, optional
        VAE routers (used for hard assignment in hidden computation).
    """

    def __init__(
        self,
        config: SLDAConfig,
        slot_model: nn.Module,
        agents: List[nn.Module],
        aggregator: nn.Module,
        slda: StreamLDA,
        vaes: Optional[List[Any]] = None,
    ):
        self.config      = config
        self.device      = self._resolve_device(config.device)
        self.slot_model  = slot_model.to(self.device).eval()
        self.agents      = [a.to(self.device).eval() for a in agents]
        self.aggregator  = aggregator.to(self.device).eval()
        self.slda        = slda
        self.vaes        = vaes or []

        # Freeze all
        for mod in [self.slot_model, *self.agents, self.aggregator]:
            for p in mod.parameters():
                p.requires_grad_(False)

        # Register agents with aggregator (required by AttentionAggregator)
        if hasattr(self.aggregator, "register_agent"):
            for i in range(len(self.agents)):
                self.aggregator.register_agent(i)

        logger.info(
            f"[SLDATrainer] feature_dim={config.feature_dim}  "
            f"n_classes={config.n_classes}  shrinkage={config.shrinkage}"
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        """One-pass incremental SLDA fitting."""
        logger.info("[SLDATrainer] Starting SLDA fitting pass.")
        total = 0
        max_b = self.config.max_batches

        pbar = tqdm(enumerate(dataloader), total=max_b if max_b > 0 else len(dataloader),
                     desc="SLDA fit", unit="batch", dynamic_ncols=True)
        for batch_idx, batch in pbar:
            if max_b > 0 and batch_idx >= max_b:
                break

            images, labels = self._unpack(batch)
            if labels is None:
                continue

            images = images.to(self.device)

            H = self._extract_H(images)     # (B, D_h)
            self.slda.update(H, labels)
            total += images.shape[0]
            pbar.set_postfix(samples=total)

        logger.info(f"[SLDATrainer] SLDA fitted on {total} samples.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run prediction over a dataloader.

        Returns
        -------
        preds  : LongTensor  (N,)
        labels : LongTensor  (N,)
        """
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="SLDA predict", unit="batch",
                          dynamic_ncols=True, leave=False):
            images, labels = self._unpack(batch)
            images = images.to(self.device)

            H = self._extract_H(images)
            preds = self.slda.predict(H)
            all_preds.append(preds)
            if labels is not None:
                all_labels.append(labels)

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels) if all_labels else torch.zeros_like(preds)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Predict + compute top-1 accuracy."""
        preds, labels = self.predict(dataloader)
        labels = labels.cpu()
        acc = (preds == labels).float().mean().item()
        logger.info(f"[SLDATrainer] Top-1 accuracy: {acc:.4f}")
        return {"accuracy": acc}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_slda(self, path: str) -> None:
        self.slda.save(path)
        logger.info(f"[SLDATrainer] SLDA saved → {path}")

    # ------------------------------------------------------------------
    # Internal: feature extraction
    # ------------------------------------------------------------------

    def _extract_H(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: images → slots → (hard) route & agent → aggregate → H.
        """
        # 1. Slots
        out   = self.slot_model(images)
        slots = out["slots"]                         # (B, K, D)

        # 2. Route (hard argmax)
        # vae.score() expects (N, D_s) not (B, K, D) — flatten then reshape back
        B, K, D = slots.shape
        if self.vaes:
            slots_flat = slots.reshape(B * K, D)
            scores = torch.stack(
                [vae.score(slots_flat) for vae in self.vaes], dim=-1
            )  # (B*K, n_agents)
            scores = scores.reshape(B, K, len(self.vaes))  # (B, K, n_agents)
            assignments = scores.argmax(dim=-1)             # (B, K)
        else:
            assignments = torch.zeros(
                slots.shape[:2], dtype=torch.long, device=self.device
            )

        # 3. Per-agent hidden (mix hard)
        n_agents = len(self.agents)
        D_h = None

        # Build a tensor to hold the final hidden for each slot
        # We'll do a hard assignment: h_k = agent_{assignments[b,k]}(slots_{b,k})
        hidden_slots = []
        for m, agent in enumerate(self.agents):
            out_m = agent(slots)
            h_m   = out_m["hidden"]           # (B, K, D_h)
            if D_h is None:
                D_h = h_m.shape[-1]
            hidden_slots.append(h_m)

        if not hidden_slots:
            raise RuntimeError("No agents available.")

        # For hard routing, select the right agent per slot
        H_stack = torch.stack(hidden_slots, dim=2)   # (B, K, n_agents, D_h)
        assign_exp = assignments.unsqueeze(-1).unsqueeze(-1).expand(
            B, K, 1, D_h
        )                                             # (B, K, 1, D_h)
        chosen = H_stack.gather(2, assign_exp).squeeze(2)  # (B, K, D_h)

        # 4. Aggregate
        H = self._agg(chosen, assignments)              # (B, D_h)
        return H

    def _agg(self, hidden_slots: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, K, D_h) → (B, D_h)."""
        try:
            result = self.aggregator(hidden_slots, assignments)
            return result["aggregated"] if isinstance(result, dict) else result
        except Exception:
            return hidden_slots.mean(dim=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack(batch):
        if isinstance(batch, (list, tuple)):
            return batch[0], (batch[1] if len(batch) > 1 else None)
        if isinstance(batch, dict):
            return batch["image"], batch.get("label")
        return batch, None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
