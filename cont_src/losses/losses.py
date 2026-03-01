"""
Loss Functions Module

Implements various loss functions for continual learning:
- Primitive loss (concept-level KL divergence)
- Supervised contrastive loss
- Agent reconstruction loss
- Local geometry loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from cont_src.core.registry import LOSS_REGISTRY


class BaseLoss(nn.Module):
    """Base class for loss functions."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute loss. Must return scalar tensor."""
        raise NotImplementedError


@LOSS_REGISTRY.register("primitive")
class PrimitiveLoss(BaseLoss):
    """
    Primitive/Concept Loss (L_p) from CompSLOT paper.

    Matrix-level KL divergence between label similarity and hidden similarity:
    d^y_ij = 1[y_i = y_j] / Σ_k 1[y_i = y_k]
    d^H_ij = exp(τ·sim(H_i, H_j)) / Σ_k exp(τ·sim(H_i, H_k))
    L_p = Σ_ij d^y_ij log(d^y_ij / d^H_ij)

    Reference: Training pipeline Eq. (3)
    """

    def __init__(self, temperature: float = 10.0, weight: float = 1.0):
        """
        Args:
            temperature: Temperature parameter τ for similarity
            weight: Loss weight
        """
        super().__init__(weight)
        self.temperature = temperature

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute primitive loss.

        Args:
            hidden: Aggregated hidden representations, shape (B, D_h)
            labels: Class labels, shape (B,)

        Returns:
            Scalar loss
        """
        B = hidden.size(0)
        device = hidden.device

        # Normalize hidden representations
        hidden_norm = F.normalize(hidden, p=2, dim=1)

        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.mm(hidden_norm, hidden_norm.t())

        # Label similarity matrix d^y: (B, B)
        # d^y_ij = 1[y_i = y_j] / |{k : y_k = y_i}|
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        label_matrix = label_matrix.float()

        # Normalize by class size
        class_sizes = label_matrix.sum(dim=1, keepdim=True)  # (B, 1)
        d_y = label_matrix / (class_sizes + 1e-8)  # (B, B)

        # Hidden similarity matrix d^H: (B, B)
        # d^H_ij = exp(τ·sim) / Σ_k exp(τ·sim)
        logits = self.temperature * sim_matrix
        d_H = F.softmax(logits, dim=1)  # (B, B)

        # KL divergence: Σ_ij d^y_ij log(d^y_ij / d^H_ij)
        # Avoid log(0) by adding small epsilon
        kl_div = d_y * torch.log((d_y + 1e-8) / (d_H + 1e-8))

        loss = kl_div.sum() / B

        return self.weight * loss


@LOSS_REGISTRY.register("supcon")
class SupervisedContrastiveLoss(BaseLoss):
    """
    Supervised Contrastive Loss (L_SupCon).

    Pair-level metric learning loss:
    L = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log[exp(sim(H_i, H_p)/τ) / Σ_{a≠i} exp(sim(H_i, H_a)/τ)]

    where P(i) is the set of same-class samples.

    Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(self, temperature: float = 0.07, weight: float = 1.0):
        """
        Args:
            temperature: Temperature parameter τ
            weight: Loss weight
        """
        super().__init__(weight)
        self.temperature = temperature

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            hidden: Hidden representations, shape (B, D_h)
            labels: Class labels, shape (B,)

        Returns:
            Scalar loss
        """
        B = hidden.size(0)
        device = hidden.device

        # Normalize
        hidden_norm = F.normalize(hidden, p=2, dim=1)

        # Similarity matrix
        sim_matrix = torch.mm(hidden_norm, hidden_norm.t()
                              ) / self.temperature  # (B, B)

        # Mask for same-class pairs (excluding self)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        mask_positive = labels_eq.float()
        # Exclude diagonal (self)
        mask_positive = mask_positive * (1 - torch.eye(B, device=device))

        # Mask for negatives (different class)
        mask_negative = (1 - labels_eq.float()) * \
            (1 - torch.eye(B, device=device))

        # Number of positives per sample
        num_positives = mask_positive.sum(dim=1)  # (B,)

        # Compute loss for each sample
        losses = []
        for i in range(B):
            if num_positives[i] == 0:
                # No positive pairs for this sample
                continue

            # Numerator: exp(sim(i, positive))
            positive_logits = sim_matrix[i] * mask_positive[i]  # (B,)

            # Denominator: Σ_{a≠i} exp(sim(i, a))
            # Create mask for all except self
            mask_all_except_self = 1 - torch.eye(B, device=device)[i]
            denominator = torch.logsumexp(
                sim_matrix[i] + torch.log(mask_all_except_self + 1e-8), dim=0)

            # Log probability for each positive
            log_probs = []
            for j in range(B):
                if mask_positive[i, j] > 0:
                    log_prob = sim_matrix[i, j] - denominator
                    log_probs.append(log_prob)

            if len(log_probs) > 0:
                # Average over positives
                loss_i = -torch.stack(log_probs).mean()
                losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(losses).mean()

        return self.weight * loss


@LOSS_REGISTRY.register("agent_reconstruction")
class AgentReconstructionLoss(BaseLoss):
    """
    Agent Reconstruction Loss (L_agent).

    Prevents agent collapse by requiring agents to reconstruct slots:
    L_agent = (1/K) Σ_k ||decoder_σ(k)(h_k) - s_k||^2

    Reference: Training pipeline Eq. (4)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(
        self,
        slots: torch.Tensor,
        reconstructed_slots: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute agent reconstruction loss.

        Args:
            slots: Original slots, shape (B, K, D_slot)
            reconstructed_slots: Reconstructed slots, shape (B, K, D_slot)

        Returns:
            Scalar loss
        """
        loss = F.mse_loss(reconstructed_slots, slots, reduction="mean")
        return self.weight * loss


@LOSS_REGISTRY.register("reconstruction")
class ReconstructionLoss(BaseLoss):
    """
    Slot Attention Reconstruction Loss (L_recon).

    Standard reconstruction loss for Slot Attention:
    L_recon = ||E - Ẽ||^2

    Reference: Training pipeline Eq. (1)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            reconstruction: Reconstructed features
            target: Target features

        Returns:
            Scalar loss
        """
        loss = F.mse_loss(reconstruction, target, reduction="mean")
        return self.weight * loss


@LOSS_REGISTRY.register("local_geometry")
class LocalGeometryLoss(BaseLoss):
    """
    Local Geometry Loss (L_local) from Gao et al. 2023.

    Enforces local neighborhood structure:
    L_local = Σ_{i,j} e^{t-1}_{ij} · ||H_i^t - H_j^t||^2

    where e^{t-1}_{ij} = w^{t-1}_{ij} - b^{t-1}_{ij}
    w = within-class neighbors, b = between-class neighbors

    Requires buffer of exemplars to compute neighbor affinities.

    Reference: Training pipeline Eq. (5), Gao et al. "Exploring Data Geometry for CL" (CVPR 2023)
    """

    def __init__(self, k_neighbors: int = 5, weight: float = 0.5):
        """
        Args:
            k_neighbors: Number of neighbors to consider
            weight: Loss weight
        """
        super().__init__(weight)
        self.k_neighbors = k_neighbors

    def forward(
        self,
        hidden_current: torch.Tensor,
        hidden_previous: torch.Tensor,
        labels_current: torch.Tensor,
        labels_previous: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local geometry loss.

        Args:
            hidden_current: Current hidden representations, shape (B_curr, D_h)
            hidden_previous: Previous hidden representations (from buffer), shape (B_prev, D_h)
            labels_current: Current labels, shape (B_curr,)
            labels_previous: Previous labels, shape (B_prev,)

        Returns:
            Scalar loss
        """
        B_curr = hidden_current.size(0)
        B_prev = hidden_previous.size(0)

        if B_prev == 0:
            # No previous data - skip loss
            return torch.tensor(0.0, device=hidden_current.device)

        # Compute pairwise distances in previous step
        with torch.no_grad():
            # Normalize
            hidden_prev_norm = F.normalize(hidden_previous, p=2, dim=1)

            # Distance matrix (use negative cosine similarity)
            # (B_prev, B_prev)
            dist_matrix = 1 - torch.mm(hidden_prev_norm, hidden_prev_norm.t())

            # Find k nearest neighbors for each sample
            _, neighbor_indices = torch.topk(
                dist_matrix, k=self.k_neighbors + 1, largest=False, dim=1)
            # Exclude self (first neighbor is always self with distance 0)
            neighbor_indices = neighbor_indices[:, 1:]  # (B_prev, k)

            # Compute affinities e_ij = w_ij - b_ij
            affinity_matrix = torch.zeros(
                B_prev, B_prev, device=hidden_current.device)

            for i in range(B_prev):
                for j_idx in range(self.k_neighbors):
                    j = neighbor_indices[i, j_idx].item()

                    # w_ij = 1 if same class and neighbors, else 0
                    # b_ij = 1 if different class and neighbors, else 0
                    if labels_previous[i] == labels_previous[j]:
                        w_ij = 1.0
                        b_ij = 0.0
                    else:
                        w_ij = 0.0
                        b_ij = 1.0

                    e_ij = w_ij - b_ij
                    affinity_matrix[i, j] = e_ij

        # Compute loss on current representations
        # We enforce that pairs with positive affinity stay close,
        # and pairs with negative affinity stay far

        # For simplicity, use current batch only
        # (Full implementation would use buffer examples)
        hidden_curr_norm = F.normalize(hidden_current, p=2, dim=1)
        dist_curr = torch.cdist(
            hidden_curr_norm, hidden_curr_norm, p=2) ** 2  # (B_curr, B_curr)

        # Use a subset of affinity matrix if B_curr < B_prev
        if B_curr <= B_prev:
            affinity_subset = affinity_matrix[:B_curr, :B_curr]
            loss = (affinity_subset * dist_curr).sum() / (B_curr * B_curr)
        else:
            loss = torch.tensor(0.0, device=hidden_current.device)

        return self.weight * loss


@LOSS_REGISTRY.register("sparse_penalty")
class SparsePenalty(BaseLoss):
    """
    Sparse Penalty Loss from AdaSlot paper.

    Penalizes the mean slot keep probability to encourage dropping unnecessary slots.
    Uses combination of linear and quadratic penalties:

    L_sparse = λ_1 * mean(p_keep) + λ_2 * (mean(p_keep) - bias)^2

    where:
    - p_keep: Slot keep probabilities from Gumbel selector
    - λ_1: Linear weight (encourages dropping slots)
    - λ_2: Quadratic weight (encourages staying near bias)
    - bias: Target keep probability (default: 0.5)

    Reference: AdaSlot implementation (ocl/losses.py)
    """

    def __init__(
        self,
        linear_weight: float = 1.0,
        quadratic_weight: float = 0.0,
        quadratic_bias: float = 0.5,
        weight: float = 1.0
    ):
        """
        Args:
            linear_weight: λ_1 for linear penalty
            quadratic_weight: λ_2 for quadratic penalty
            quadratic_bias: Target mean keep probability
            weight: Overall loss weight
        """
        super().__init__(weight)
        self.linear_weight = linear_weight
        self.quadratic_weight = quadratic_weight
        self.quadratic_bias = quadratic_bias

    def forward(
        self,
        slot_keep_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sparsity penalty.

        Args:
            slot_keep_probs: Slot keep probabilities, shape (B, K) or (B, K, 2)
                If shape is (B, K, 2), takes [..., 1] (keep probability)

        Returns:
            Scalar loss
        """
        # Handle both soft probs (B, K, 2) and hard decisions (B, K)
        if slot_keep_probs.dim() == 3:
            # Extract keep probability (second channel)
            slot_keep_probs = slot_keep_probs[..., 1]

        # Compute mean keep probability across batch and slots
        sparse_degree = torch.mean(slot_keep_probs)

        # Linear + quadratic penalty
        linear_term = self.linear_weight * sparse_degree
        quadratic_term = self.quadratic_weight * \
            (sparse_degree - self.quadratic_bias) ** 2

        return self.weight * (linear_term + quadratic_term)


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions.

    Automatically weights and combines losses based on config.
    """

    def __init__(self, loss_config: Dict):
        """
        Args:
            loss_config: Dictionary with loss configuration
        """
        super().__init__()
        self.losses = nn.ModuleDict()

        # Register losses based on config
        if loss_config.get("use_primitive", False):
            self.losses["primitive"] = PrimitiveLoss(
                temperature=loss_config.get("primitive_temperature", 10.0),
                weight=loss_config.get("weight_primitive", 1.0)
            )

        if loss_config.get("use_supcon", False):
            self.losses["supcon"] = SupervisedContrastiveLoss(
                temperature=loss_config.get("supcon_temperature", 0.07),
                weight=loss_config.get("weight_supcon", 1.0)
            )

        if loss_config.get("use_agent_recon", False):
            self.losses["agent_recon"] = AgentReconstructionLoss(
                weight=loss_config.get("weight_agent_recon", 1.0)
            )

        if loss_config.get("use_reconstruction", False):
            self.losses["reconstruction"] = ReconstructionLoss(
                weight=loss_config.get("weight_reconstruction", 1.0)
            )

        if loss_config.get("use_local_geometry", False):
            self.losses["local_geometry"] = LocalGeometryLoss(
                k_neighbors=loss_config.get("local_k_neighbors", 5),
                weight=loss_config.get("weight_local_geometry", 0.5)
            )

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            **inputs: Input tensors for losses (hidden, labels, slots, etc.)

        Returns:
            Dict with individual losses and total loss
        """
        output = {}
        total_loss = 0.0

        # Compute each loss
        if "primitive" in self.losses:
            loss = self.losses["primitive"](inputs["hidden"], inputs["labels"])
            output["primitive"] = loss
            total_loss = total_loss + loss

        if "supcon" in self.losses:
            loss = self.losses["supcon"](inputs["hidden"], inputs["labels"])
            output["supcon"] = loss
            total_loss = total_loss + loss

        if "agent_recon" in self.losses:
            loss = self.losses["agent_recon"](
                inputs["slots"], inputs["reconstructed_slots"]
            )
            output["agent_recon"] = loss
            total_loss = total_loss + loss

        if "reconstruction" in self.losses:
            loss = self.losses["reconstruction"](
                inputs["reconstruction"], inputs["target"]
            )
            output["reconstruction"] = loss
            total_loss = total_loss + loss

        if "local_geometry" in self.losses:
            loss = self.losses["local_geometry"](
                inputs["hidden_current"],
                inputs.get("hidden_previous", torch.tensor([])),
                inputs["labels_current"],
                inputs.get("labels_previous", torch.tensor([]))
            )
            output["local_geometry"] = loss
            total_loss = total_loss + loss

        output["total"] = total_loss

        return output
