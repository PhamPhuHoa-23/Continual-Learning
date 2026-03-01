"""
Streaming Linear Discriminant Analysis (SLDA) Classifier
=========================================================

Incremental, closed-form classifier that updates class means and a shared
covariance matrix one sample at a time — no gradient steps required.

Decision rule at inference:
    ŷ = argmax_c [ H^T Σ⁻¹ μ_c  −  ½ μ_c^T Σ⁻¹ μ_c  +  log π_c ]

Usage
-----
    from cont_src.models.classifiers.slda import SLDAClassifier

    clf = SLDAClassifier(feature_dim=64, n_classes=100)
    clf.fit(features, labels)          # update on a batch
    preds = clf.predict(features)      # (B,) LongTensor
    proba = clf.predict_proba(features) # (B, C) FloatTensor
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from cont_src.core.base_module import BaseClassifier


class SLDAClassifier(BaseClassifier):
    """
    Streaming Linear Discriminant Analysis classifier.

    Maintains incremental per-class means and a shared covariance matrix
    using Welford's online algorithm.  No backward pass is needed.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of input feature vectors.
    n_classes : int
        Maximum number of classes (informational, not hard-enforced).
    shrinkage : float
        Ridge regularisation added to the covariance diagonal.
    normalize_input : bool
        If True, L2-normalises features before updating / predicting.
        Recommended when upstream training used SupCon loss.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 1000,
        shrinkage: float = 1e-4,
        normalize_input: bool = True,
    ):
        super().__init__(config={
            "feature_dim": feature_dim,
            "n_classes": n_classes,
            "shrinkage": shrinkage,
            "normalize_input": normalize_input,
        })

        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.shrinkage = shrinkage
        self.normalize_input = normalize_input

        # Statistics (CPU tensors — they never need autograd)
        self._n_c:     Dict[int, int] = {}
        self._mu_c:    Dict[int, torch.Tensor] = {}
        self._S:       torch.Tensor = torch.zeros(feature_dim, feature_dim)
        self._n_total: int = 0

    # ------------------------------------------------------------------
    # BaseClassifier abstract methods
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Alias for predict — makes clf(x) work naturally."""
        return self.predict(features)

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Incrementally update SLDA statistics with a batch.

        Parameters
        ----------
        features : (B, D)
        labels   : (B,) – integer class indices
        """
        if self.normalize_input:
            features = F.normalize(features, p=2, dim=-1)

        features = features.detach().cpu().float()
        labels   = labels.detach().cpu()

        for feat, lbl in zip(features, labels):
            c = int(lbl.item())

            if c not in self._n_c:
                self._n_c[c]  = 0
                self._mu_c[c] = torch.zeros(self.feature_dim)

            n_old = self._n_c[c]
            n_new = n_old + 1
            delta       = feat - self._mu_c[c]
            self._mu_c[c] = self._mu_c[c] + delta / n_new
            self._n_c[c]  = n_new

            # Welford scatter update
            delta2  = feat - self._mu_c[c]
            self._S = self._S + torch.outer(delta, delta2)
            self._n_total += 1

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        preds : (B,) LongTensor
        """
        scores = self._decision_scores(features)          # (B, C)
        idx    = scores.argmax(dim=1)                     # (B,)
        classes = sorted(self._mu_c.keys())
        return torch.tensor([classes[i] for i in idx.tolist()])

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities via softmax over LDA scores.

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        proba : (B, C) FloatTensor
        """
        scores = self._decision_scores(features)          # (B, C)
        return torch.softmax(scores, dim=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def covariance(self) -> torch.Tensor:
        """Regularised shared covariance matrix  (D, D)."""
        if self._n_total <= 1:
            return torch.eye(self.feature_dim) * (self.shrinkage + 1.0)
        cov = self._S / (self._n_total - 1)
        return cov + self.shrinkage * torch.eye(self.feature_dim)

    @property
    def classes(self) -> list:
        """Sorted list of observed class indices."""
        return sorted(self._mu_c.keys())

    @property
    def n_seen(self) -> int:
        """Total number of samples seen so far."""
        return self._n_total

    def is_fitted(self) -> bool:
        return len(self._mu_c) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decision_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute LDA decision scores for each class.

        Returns (B, C) FloatTensor.
        """
        if not self._mu_c:
            raise RuntimeError("SLDAClassifier has not been fitted yet.")

        if self.normalize_input:
            features = F.normalize(features, p=2, dim=-1)

        features = features.detach().cpu().float()

        cov_inv  = torch.linalg.inv(self.covariance)     # (D, D)

        classes   = self.classes
        mu_stack  = torch.stack(
            [self._mu_c[c] for c in classes], dim=0
        )                                                  # (C, D)

        # H @ Σ⁻¹ @ μ_c^T  −  ½ μ_c @ Σ⁻¹ @ μ_c^T
        h_proj   = features @ cov_inv                     # (B, D)
        scores   = h_proj @ mu_stack.t()                  # (B, C)
        quadform = 0.5 * (mu_stack @ cov_inv * mu_stack).sum(dim=1)  # (C,)
        return scores - quadform.unsqueeze(0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:  # type: ignore[override]
        return {
            "n_c":     self._n_c,
            "mu_c":    {k: v.clone() for k, v in self._mu_c.items()},
            "S":       self._S.clone(),
            "n_total": self._n_total,
            "config":  self.config,
        }

    def load_state_dict(self, state: dict, strict: bool = True):  # type: ignore[override]
        self._n_c     = state["n_c"]
        self._mu_c    = {k: v.clone() for k, v in state["mu_c"].items()}
        self._S       = state["S"].clone()
        self._n_total = state["n_total"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        feature_dim: int,
        n_classes: int = 1000,
        shrinkage: float = 1e-4,
    ) -> "SLDAClassifier":
        obj = cls(feature_dim=feature_dim, n_classes=n_classes, shrinkage=shrinkage)
        state = torch.load(path, weights_only=False)
        obj.load_state_dict(state)
        return obj

    def __repr__(self) -> str:
        return (
            f"SLDAClassifier("
            f"feature_dim={self.feature_dim}, "
            f"n_classes_seen={len(self._mu_c)}, "
            f"n_samples={self._n_total})"
        )
