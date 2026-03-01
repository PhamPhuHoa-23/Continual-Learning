"""
Linear Classifier
=================

Standard trainable linear classification head.

Usage
-----
    from cont_src.models.classifiers.linear import LinearClassifier

    clf = LinearClassifier(feature_dim=64, n_classes=100)
    optim = torch.optim.Adam(clf.parameters(), lr=1e-3)

    # Training
    logits = clf(features)
    loss = F.cross_entropy(logits, labels)

    # Inference
    preds = clf.predict(features)
    proba = clf.predict_proba(features)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.core.base_module import BaseClassifier


class LinearClassifier(BaseClassifier):
    """
    Single linear layer classification head.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of input feature vectors.
    n_classes : int
        Number of output classes.
    bias : bool
        Whether to include a bias term.
    normalize_input : bool
        If True, L2-normalises features before the linear layer.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        bias: bool = True,
        normalize_input: bool = False,
    ):
        super().__init__(config={
            "feature_dim": feature_dim,
            "n_classes": n_classes,
            "bias": bias,
            "normalize_input": normalize_input,
        })

        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.normalize_input = normalize_input

        self.fc = nn.Linear(feature_dim, n_classes, bias=bias)

    # ------------------------------------------------------------------
    # BaseClassifier abstract methods
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute raw logits.

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        logits : (B, n_classes)
        """
        if self.normalize_input:
            features = F.normalize(features, p=2, dim=-1)
        return self.fc(features)

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Not used for gradient-based training — call optimizer.step() directly.
        Provided to satisfy the BaseClassifier interface.
        """
        raise NotImplementedError(
            "LinearClassifier is trained via gradient descent. "
            "Use the standard PyTorch training loop (loss.backward, optimizer.step)."
        )

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels (argmax over logits).

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        preds : (B,) LongTensor
        """
        with torch.no_grad():
            logits = self(features)
        return logits.argmax(dim=1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities via softmax.

        Parameters
        ----------
        features : (B, D)

        Returns
        -------
        proba : (B, n_classes) FloatTensor
        """
        with torch.no_grad():
            logits = self(features)
        return torch.softmax(logits, dim=1)

    def __repr__(self) -> str:
        return (
            f"LinearClassifier("
            f"feature_dim={self.feature_dim}, "
            f"n_classes={self.n_classes})"
        )
