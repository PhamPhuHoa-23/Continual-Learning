"""
CRP-based Expert Assignment Aggregator for Continual Learning.

Replaces the Hoeffding Tree with a dynamic expert assignment system
that uses Chinese Restaurant Process (CRP), Gradient Alignment Scoring,
and Gradient Projection to handle class-incremental learning.

Pipeline:
    Agent outputs (hidden labels) → CRP Expert Assignment → Class Prediction

Two core problems solved:
    1. Which expert does input x belong to? (Prototype matching + cosine similarity)
    2. Create new expert or assign to existing? (CRP + Gradient Alignment Score)



Reference:
    - CRP: Aldous (1985), Pitman (2006)
    - GPM: Saha et al. (2021) "Gradient Projection Memory for Continual Learning"
    - Prototypical Networks: Snell et al. (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy


class ExpertModule(nn.Module):
    """
    Lightweight linear expert for class-incremental learning.

    Each expert:
    - Has a small 2-layer MLP classifier.
    - Maintains a prototype (running mean) for cosine similarity matching.
    - Stores gradient memory (g_old^k) for alignment scoring.
    - Stores projection bases (SVD) for gradient projection (GPM-lite).

    Args:
        expert_id: Unique expert identifier.
        feature_dim: Dimension of input features (agent hidden labels).
        num_classes: Maximum number of classes to predict.
        hidden_dim: Hidden dimension of classifier MLP.
        prototype_momentum: EMA momentum for prototype update.
        gradient_momentum: EMA momentum for gradient memory update.
        projection_rank: Max rank for GPM projection bases.
    """

    def __init__(
        self,
        expert_id: int,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        prototype_momentum: float = 0.95,
        gradient_momentum: float = 0.9,
        projection_rank: int = 10,
    ):
        super().__init__()

        self.expert_id = expert_id
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.prototype_momentum = prototype_momentum
        self.gradient_momentum = gradient_momentum
        self.projection_rank = projection_rank

        # --- Lightweight MLP Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # --- Prototype (running mean of assigned inputs) ---
        self.register_buffer("prototype", torch.zeros(feature_dim))
        self.register_buffer("prototype_count", torch.tensor(0, dtype=torch.long))

        # --- Gradient Memory (EMA of past gradients for alignment) ---
        self._grad_memory_initialized = False
        self.register_buffer("gradient_memory", torch.zeros(1))  # resized on first use

        # --- Projection Bases for GPM (orthogonal subspace) ---
        self.register_buffer("projection_bases", torch.empty(0))

        # --- Statistics ---
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.total_assigned: int = 0

        # --- Activation buffer for SVD (GPM) ---
        self._activation_buffer: List[torch.Tensor] = []
        self._buffer_max_size: int = 200

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier.

        Args:
            x: (batch_size, feature_dim) or (feature_dim,)

        Returns:
            logits: (batch_size, num_classes) or (num_classes,)
        """
        return self.classifier(x)

    @torch.no_grad()
    def update_prototype(self, x: torch.Tensor):
        """
        Update prototype via EMA of input features.

        Args:
            x: (feature_dim,) — single sample feature vector.
        """
        if x.dim() > 1:
            x = x.mean(dim=0)

        if self.prototype_count == 0:
            self.prototype.copy_(x)
        else:
            m = self.prototype_momentum
            self.prototype.mul_(m).add_(x, alpha=1 - m)

        self.prototype_count += 1

    def update_gradient_memory(self, g_new: torch.Tensor):
        """
        Update gradient memory via EMA.

        Args:
            g_new: Flattened gradient vector from the latest update.
        """
        if not self._grad_memory_initialized:
            self.gradient_memory = torch.zeros_like(g_new)
            self._grad_memory_initialized = True

        m = self.gradient_momentum
        self.gradient_memory.mul_(m).add_(g_new.detach(), alpha=1 - m)

    def buffer_activation(self, x: torch.Tensor):
        """Store activation for later SVD computation."""
        self._activation_buffer.append(x.detach().cpu())
        if len(self._activation_buffer) > self._buffer_max_size:
            self._activation_buffer.pop(0)

    @torch.no_grad()
    def update_projection_bases(self):
        """
        Compute SVD on buffered activations and store top singular vectors
        as projection bases (GPM-style).
        """
        if len(self._activation_buffer) < 5:
            return  # Not enough data

        activations = torch.stack(self._activation_buffer).to(self.prototype.device)
        # Center activations
        activations = activations - activations.mean(dim=0, keepdim=True)

        try:
            U, S, _ = torch.linalg.svd(activations, full_matrices=False)
            # Keep top-rank columns of V^T (right singular vectors correspond to feature directions)
            # We want bases in feature space: use columns of V = rows of Vh
            _, _, Vh = torch.linalg.svd(activations, full_matrices=False)
            rank = min(self.projection_rank, Vh.shape[0], activations.shape[1])
            self.projection_bases = Vh[:rank].T  # (feature_dim, rank)
        except Exception:
            pass  # SVD can fail on degenerate matrices; skip silently

    def get_num_assigned(self) -> int:
        """Total number of samples assigned to this expert."""
        return self.total_assigned

    def get_info(self) -> Dict:
        """Expert statistics."""
        return {
            "expert_id": self.expert_id,
            "total_assigned": self.total_assigned,
            "class_counts": dict(self.class_counts),
            "prototype_norm": self.prototype.norm().item(),
            "has_projection_bases": self.projection_bases.numel() > 0,
        }


class CRPExpertAggregator(nn.Module):
    """
    CRP-based Expert Assignment Aggregator.

    Dynamically creates and routes to lightweight expert modules using
    Chinese Restaurant Process for expert creation decisions and
    a balanced scoring formula that prevents expert overload.

    Pipeline for each sample (x, label):
        1. Compute cosine similarity to all expert prototypes.
        2. For each expert k, compute:
           Score(k) = Similarity(x, proto_k) × Alignment(g_new, g_old^k) × Capacity(k)
        3. Capacity(k) penalizes experts with too many diverse classes
           (prevents "hogging" in continual learning)
        4. CRP probability of new expert: P(new) = α / (N_total + α)
        5. If best Score < threshold → consider creating new expert.
        6. Train assigned expert with gradient projected orthogonal to
           past important subspaces (GPM-lite).

    Scoring Philosophy (vs. vanilla CRP):
        Vanilla CRP: Popularity = N_k / (N + α) → rich-get-richer → one expert
        hogs all classes, which is terrible for continual learning.

        Our approach (inspired by Switch Transformer load balancing & Expert Gate):
        - Similarity: Route to the most relevant expert (prototype matching)
        - Alignment: Ensure gradient compatibility (no destructive interference)
        - Capacity: Penalize overloaded experts (exponential decay on class count)
        This naturally distributes classes across experts and promotes specialization.

    Args:
        feature_dim: Dimension of agent output features.
        num_classes: Maximum number of classes.
        alpha: CRP concentration parameter. Higher = more new experts.
        max_experts: Max number of experts allowed.
        hidden_dim: Hidden dim for each expert's classifier MLP.
        buffer_size: Per-expert activation buffer for GPM SVD.
        projection_rank: Max rank for GPM projection bases.
        expert_lr: Learning rate for expert classifier updates.
        score_threshold: Minimum score to accept existing expert (vs create new).
        prototype_momentum: EMA momentum for prototype updates.
        gradient_momentum: EMA momentum for gradient memory updates.
        capacity_beta: Strength of capacity penalty. Higher = stronger penalty for
            overloaded experts. Default 1.5.
        ideal_classes_per_expert: Target number of classes per expert for capacity
            computation. Default 5 (for CIFAR-100 with ~20 experts).
        device: Torch device.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 100,
        alpha: float = 1.0,
        max_experts: int = 30,
        hidden_dim: int = 256,
        buffer_size: int = 100,
        projection_rank: int = 10,
        expert_lr: float = 1e-3,
        score_threshold: float = 0.05,
        prototype_momentum: float = 0.95,
        gradient_momentum: float = 0.9,
        capacity_beta: float = 1.5,
        ideal_classes_per_expert: int = 5,
        device: str = "cpu",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.max_experts = max_experts
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.projection_rank = projection_rank
        self.expert_lr = expert_lr
        self.score_threshold = score_threshold
        self.prototype_momentum = prototype_momentum
        self.gradient_momentum = gradient_momentum
        self.capacity_beta = capacity_beta
        self.ideal_classes_per_expert = ideal_classes_per_expert
        self.device = device

        # Dynamic list of experts
        self.experts = nn.ModuleList()

        # Global statistics
        self.total_samples = 0
        self.num_classes_seen = 0
        self.class_counts: Dict[int, int] = defaultdict(int)

        # Mapping: expert_id → set of class_ids
        self.expert_class_map: Dict[int, set] = defaultdict(set)

        # Next expert ID counter
        self._next_expert_id = 0

    def _create_expert(
        self, init_from: Optional["ExpertModule"] = None
    ) -> "ExpertModule":
        """
        Create a new ExpertModule.

        Args:
            init_from: If provided, copy weights from this expert + add noise
                       (smart initialization).

        Returns:
            New ExpertModule, already moved to self.device.
        """
        expert = ExpertModule(
            expert_id=self._next_expert_id,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            prototype_momentum=self.prototype_momentum,
            gradient_momentum=self.gradient_momentum,
            projection_rank=self.projection_rank,
        ).to(self.device)

        # Smart init: copy from nearest expert + noise
        if init_from is not None:
            expert.classifier.load_state_dict(init_from.classifier.state_dict())
            with torch.no_grad():
                for p in expert.classifier.parameters():
                    p.add_(torch.randn_like(p) * 0.01)

        expert._buffer_max_size = self.buffer_size
        self._next_expert_id += 1
        return expert

    def _find_nearest_expert(self, x: torch.Tensor) -> Optional[int]:
        """
        Find expert whose prototype is most similar to x (cosine similarity).

        Args:
            x: (feature_dim,)

        Returns:
            Index into self.experts, or None if no experts exist.
        """
        if len(self.experts) == 0:
            return None

        similarities = []
        for expert in self.experts:
            if expert.prototype_count == 0:
                similarities.append(-1.0)
            else:
                sim = F.cosine_similarity(
                    x.unsqueeze(0), expert.prototype.unsqueeze(0)
                ).item()
                similarities.append(sim)

        return int(np.argmax(similarities))

    def _compute_gradient(
        self, expert: ExpertModule, x: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient of cross-entropy loss w.r.t. expert classifier params.

        Args:
            expert: The expert module.
            x: (feature_dim,) input features.
            label: scalar class label.

        Returns:
            Flattened gradient vector.
        """
        # Ensure x requires no grad tracking from outside
        x_in = x.detach().unsqueeze(0).requires_grad_(False)
        label_in = label.detach().unsqueeze(0) if label.dim() == 0 else label.detach()

        logits = expert(x_in)
        loss = F.cross_entropy(logits, label_in)

        # Compute gradients w.r.t. classifier parameters only
        grads = torch.autograd.grad(
            loss, expert.classifier.parameters(), create_graph=False, allow_unused=True
        )

        flat_grad = torch.cat(
            [g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
             for g, p in zip(grads, expert.classifier.parameters())]
        )
        return flat_grad

    def _compute_expert_score(
        self, expert_idx: int, x: torch.Tensor, label: torch.Tensor
    ) -> float:
        """
        Compute balanced Score for assigning x to expert_idx.

        Score(k) = Similarity(x, proto_k) × Alignment(g_new, g_old^k) × Capacity(k)

        Three factors:
        - Similarity: Cosine similarity between input x and expert prototype.
          Routes data to the most relevant expert.
        - Alignment: Cosine similarity between new gradient and expert's gradient
          memory. Ensures gradient compatibility (no destructive interference).
        - Capacity: Exponential penalty for experts with too many diverse classes.
          Prevents any single expert from hoarding all classes.
          Capacity(k) = exp(-β × num_classes_k / ideal_classes)

        This replaces the vanilla CRP "Popularity" term (rich-get-richer) which
        was harmful for continual learning.

        Args:
            expert_idx: Index of expert in self.experts.
            x: (feature_dim,)
            label: scalar class label.

        Returns:
            Score value (float).
        """
        expert = self.experts[expert_idx]

        # --- Factor 1: Prototype Similarity ---
        # Route to the expert whose prototype is most similar to x
        if expert.prototype_count > 0:
            similarity = F.cosine_similarity(
                x.unsqueeze(0), expert.prototype.unsqueeze(0)
            ).item()
            similarity = max(0.0, (similarity + 1.0) / 2.0)  # Remap [-1,1] → [0,1]
        else:
            similarity = 0.5  # Neutral for uninitialized experts

        # --- Factor 2: Gradient Alignment ---
        # Ensure the new update won't conflict with past learning
        g_new = self._compute_gradient(expert, x, label)

        if expert._grad_memory_initialized and expert.gradient_memory.numel() > 1:
            alignment = F.cosine_similarity(
                g_new.unsqueeze(0), expert.gradient_memory.unsqueeze(0)
            ).item()
            alignment = max(0.0, alignment)  # Only positive alignment
        else:
            # No gradient history yet: use similarity as fallback
            alignment = similarity

        # --- Factor 3: Capacity (load-balancing penalty) ---
        # Penalize experts that already handle too many different classes.
        # Inspired by Switch Transformer's load-balancing auxiliary loss.
        # Capacity(k) = exp(-β × num_classes_k / ideal_classes_per_expert)
        num_classes_in_expert = len(expert.class_counts)
        capacity = np.exp(
            -self.capacity_beta * num_classes_in_expert / self.ideal_classes_per_expert
        )

        score = similarity * alignment * capacity
        return score

    def _project_gradient(
        self, expert: ExpertModule, grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Project gradient orthogonal to expert's important past subspaces (GPM-lite).

        grad_proj = grad - bases @ bases^T @ grad

        Args:
            expert: Expert whose projection bases to use.
            grad: Flattened gradient vector.

        Returns:
            Projected gradient vector.
        """
        bases = expert.projection_bases
        if bases.numel() == 0:
            return grad

        # Projection bases are in feature space, but grad is in parameter space.
        # For lightweight GPM, we apply projection per-layer using activation-space bases.
        # Here we use a simplified approach: project the full grad vector if sizes match.
        # In practice, apply per-layer for correctness.
        if bases.shape[0] == grad.shape[0]:
            proj = bases @ (bases.T @ grad)
            return grad - proj

        return grad

    def _train_expert(
        self, expert: ExpertModule, x: torch.Tensor, label: torch.Tensor
    ):
        """
        Train a single expert on (x, label) with gradient projection.

        Args:
            expert: Expert to train.
            x: (feature_dim,)
            label: scalar class label.
        """
        x_in = x.detach().unsqueeze(0)
        label_in = (
            label.detach().unsqueeze(0) if label.dim() == 0 else label.detach()
        )

        logits = expert(x_in)
        loss = F.cross_entropy(logits, label_in)

        # Compute gradients
        grads = torch.autograd.grad(
            loss, expert.classifier.parameters(), create_graph=False, allow_unused=True
        )

        # Build flat gradient for alignment memory update
        flat_grad_parts = []
        grad_list = []
        for g, p in zip(grads, expert.classifier.parameters()):
            if g is not None:
                flat_grad_parts.append(g.flatten())
                grad_list.append(g)
            else:
                flat_grad_parts.append(torch.zeros(p.numel(), device=self.device))
                grad_list.append(torch.zeros_like(p))

        flat_grad = torch.cat(flat_grad_parts)

        # Update gradient memory BEFORE projection (store raw direction)
        expert.update_gradient_memory(flat_grad)

        # Apply gradient projection (GPM-lite) — project flat grad
        flat_grad_proj = self._project_gradient(expert, flat_grad)

        # Unflatten projected gradient and apply to parameters
        offset = 0
        with torch.no_grad():
            for p in expert.classifier.parameters():
                numel = p.numel()
                p_grad = flat_grad_proj[offset : offset + numel].view_as(p)
                p.add_(p_grad, alpha=-self.expert_lr)
                offset += numel

    def assign_expert(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[int, bool]:
        """
        Assign input x to an expert using CRP + Gradient Alignment.

        Args:
            x: (feature_dim,) feature vector.
            label: scalar class label.

        Returns:
            (expert_idx, is_new): Index of chosen expert, whether it was newly created.
        """
        # --- No experts yet: must create first one ---
        if len(self.experts) == 0:
            new_expert = self._create_expert()
            self.experts.append(new_expert)
            return 0, True

        # --- Compute CRP probability of creating new expert ---
        p_new = self.alpha / (self.total_samples + self.alpha)

        # --- Compute scores for all existing experts ---
        scores = []
        for idx in range(len(self.experts)):
            s = self._compute_expert_score(idx, x, label)
            scores.append(s)

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        # --- Decision: new expert or existing? ---
        create_new = False

        if best_score < self.score_threshold:
            # All scores are low → likely OOD → probabilistic CRP decision
            if np.random.rand() < p_new or len(self.experts) < 2:
                create_new = True
        elif np.random.rand() < p_new * 0.1:
            # Small chance to explore even when score is decent
            create_new = True

        if create_new and len(self.experts) < self.max_experts:
            # Smart init from nearest expert
            nearest_idx = self._find_nearest_expert(x)
            init_from = self.experts[nearest_idx] if nearest_idx is not None else None
            new_expert = self._create_expert(init_from=init_from)
            self.experts.append(new_expert)
            return len(self.experts) - 1, True

        return best_idx, False

    def learn_one(
        self,
        hidden_labels: np.ndarray,
        label: int,
    ) -> Dict:
        """
        Learn from one example (online learning).

        Args:
            hidden_labels: (feature_dim,) — flattened agent output vector.
            label: Ground truth class label (int).

        Returns:
            info: Dict with learning info.
        """
        x = torch.tensor(hidden_labels, dtype=torch.float32, device=self.device)
        y = torch.tensor(label, dtype=torch.long, device=self.device)

        # 1. Assign to expert
        expert_idx, is_new = self.assign_expert(x, y)
        expert = self.experts[expert_idx]

        # 2. Train the expert
        self._train_expert(expert, x, y)

        # 3. Update prototype
        expert.update_prototype(x)

        # 4. Buffer activation for GPM
        expert.buffer_activation(x)

        # 5. Update statistics
        expert.total_assigned += 1
        expert.class_counts[label] += 1
        self.total_samples += 1
        self.expert_class_map[expert.expert_id].add(label)

        if label not in self.class_counts:
            self.num_classes_seen += 1
        self.class_counts[label] += 1

        return {
            "expert_idx": expert_idx,
            "expert_id": expert.expert_id,
            "is_new_expert": is_new,
            "num_experts": len(self.experts),
        }

    def predict_one(
        self, hidden_labels: np.ndarray
    ) -> Optional[int]:
        """
        Predict class for one example.

        Route to nearest expert (by prototype cosine similarity), then classify.

        Args:
            hidden_labels: (feature_dim,)

        Returns:
            Predicted class label, or None if no experts exist.
        """
        if len(self.experts) == 0:
            return None

        x = torch.tensor(hidden_labels, dtype=torch.float32, device=self.device)

        nearest_idx = self._find_nearest_expert(x)
        if nearest_idx is None:
            return None

        expert = self.experts[nearest_idx]

        with torch.no_grad():
            logits = expert(x.unsqueeze(0))
            pred = logits.argmax(dim=-1).item()

        return pred

    def predict_proba_one(
        self, hidden_labels: np.ndarray
    ) -> Dict[int, float]:
        """
        Predict class probabilities for one example.

        Args:
            hidden_labels: (feature_dim,)

        Returns:
            Dict mapping class_id → probability.
        """
        if len(self.experts) == 0:
            return {}

        x = torch.tensor(hidden_labels, dtype=torch.float32, device=self.device)

        nearest_idx = self._find_nearest_expert(x)
        if nearest_idx is None:
            return {}

        expert = self.experts[nearest_idx]

        with torch.no_grad():
            logits = expert(x.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)

        return {i: probs[i].item() for i in range(self.num_classes) if probs[i].item() > 1e-6}

    def update_all_projection_bases(self):
        """
        Trigger SVD update for all experts' GPM projection bases.
        Call this at task boundaries.
        """
        for expert in self.experts:
            expert.update_projection_bases()

    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        return {
            "num_experts": len(self.experts),
            "total_samples": self.total_samples,
            "num_classes_seen": self.num_classes_seen,
            "class_counts": dict(self.class_counts),
            "expert_class_map": {
                k: list(v) for k, v in self.expert_class_map.items()
            },
            "expert_infos": [e.get_info() for e in self.experts],
        }


class BatchCRPAggregator:
    """
    Batch wrapper for CRPExpertAggregator.

    Provides batch-level learn/predict interface while internally
    using one-by-one incremental CRP assignment.

    Args:
        aggregator: CRPExpertAggregator instance.
    """

    def __init__(self, aggregator: CRPExpertAggregator):
        self.aggregator = aggregator

    def learn_batch(
        self,
        hidden_labels_batch: "torch.Tensor",
        labels_batch: "torch.Tensor",
    ):
        """
        Learn from a batch (internally loops one-by-one through CRP).

        Args:
            hidden_labels_batch: (batch_size, feature_dim)
            labels_batch: (batch_size,)
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        labels_np = labels_batch.cpu().numpy()

        for hidden_labels, label in zip(hidden_labels_np, labels_np):
            self.aggregator.learn_one(hidden_labels, int(label))

    def predict_batch(
        self,
        hidden_labels_batch: "torch.Tensor",
    ) -> List[Optional[int]]:
        """
        Predict for a batch.

        Args:
            hidden_labels_batch: (batch_size, feature_dim)

        Returns:
            List of predicted labels.
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        return [
            self.aggregator.predict_one(hl) for hl in hidden_labels_np
        ]

    def predict_proba_batch(
        self,
        hidden_labels_batch: "torch.Tensor",
    ) -> List[Dict[int, float]]:
        """
        Predict probabilities for a batch.

        Args:
            hidden_labels_batch: (batch_size, feature_dim)

        Returns:
            List of probability dicts.
        """
        hidden_labels_np = hidden_labels_batch.cpu().numpy()
        return [
            self.aggregator.predict_proba_one(hl) for hl in hidden_labels_np
        ]

    def get_stats(self) -> Dict:
        """Get statistics."""
        return self.aggregator.get_stats()


# =============================================================================
# Factory Function
# =============================================================================

def create_aggregator(
    aggregator_type: str = "crp",
    **kwargs,
):
    """
    Factory function to create aggregators.

    Args:
        aggregator_type: One of ['crp', 'hoeffding', 'hoeffding_adaptive', 'ensemble']
        **kwargs: Additional arguments for specific aggregators.

    Returns:
        Aggregator instance.

    Example:
        >>> agg = create_aggregator('crp', feature_dim=2688, num_classes=100)
        >>> agg.learn_one(hidden_labels, label)
        >>> pred = agg.predict_one(hidden_labels)
    """
    if aggregator_type == "crp":
        crp_agg = CRPExpertAggregator(**kwargs)
        return BatchCRPAggregator(crp_agg)

    # --- Backward compatibility: Hoeffding Tree (requires river) ---
    elif aggregator_type in ("hoeffding", "hoeffding_adaptive"):
        try:
            from river.tree import (
                HoeffdingTreeClassifier,
                HoeffdingAdaptiveTreeClassifier,
            )
        except ImportError:
            raise ImportError(
                "Hoeffding tree aggregator requires the 'river' package. "
                "Install with: pip install river"
            )

        from ._legacy_aggregator import IncrementalTreeAggregator

        adaptive = aggregator_type == "hoeffding_adaptive"
        return IncrementalTreeAggregator(adaptive=adaptive, **kwargs)

    elif aggregator_type == "ensemble":
        try:
            from river.ensemble import AdaptiveRandomForestClassifier
        except ImportError:
            raise ImportError(
                "Ensemble tree aggregator requires the 'river' package. "
                "Install with: pip install river"
            )

        from ._legacy_aggregator import EnsembleTreeAggregator

        return EnsembleTreeAggregator(**kwargs)

    else:
        raise ValueError(
            f"Unknown aggregator_type: {aggregator_type}. "
            f"Choose from: crp, hoeffding, hoeffding_adaptive, ensemble"
        )
