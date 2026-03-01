"""
MoE Cross-Attention Expert Aggregator for Continual Learning.

Exclusive Expert Ownership Design:
    Each expert owns a NON-OVERLAPPING set of classes and has its own
    internal classifier.  No two experts share any class.

Expert Identity:
    Each expert holds L learnable query embeddings E ∈ R^(L × d).
    It "asks" agent outputs via cross-attention:
        z = CrossAttn(Q=E, K=agent_outputs, V=agent_outputs)
    Score = ||z|| (magnitude = expert confidence for routing)

CRP Routing (exclusive, hard assignment):
    For each input H:
        - Run all active experts → routing scores (magnitude)
        - Pick expert with highest routing score
        - That expert classifies internally using its local classifier
    New expert: when a new class appears that no expert owns
                AND CRP Bernoulli(alpha / (N_total + alpha)) triggers

Training (dual loss):
    1. Classification loss: correct expert classifies input → CE loss
    2. Contrastive routing loss: if input was routed to wrong expert,
       push correct expert's score UP and wrong expert's score DOWN:
           L_route = max(0, margin + score_wrong - score_correct)

Continual Learning guarantees:
    - Each expert's local classifier only handles its owned classes
    - Old experts are frozen when training new tasks
    - New experts are created for new class groups
    - Contrastive routing ensures correct expert selection over time

Inspiration:
    - CompSLOT: Liao et al. (ICLR 2026)
    - CRP: Aldous (1985), Pitman (2006)
    - Hard MoE routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math


class LearnableExpert(nn.Module):
    """
    Expert with exclusive class ownership + internal classifier.

    Each expert:
        1. Has learnable query embeddings for cross-attention (routing identity)
        2. Owns a specific set of classes (non-overlapping with other experts)
        3. Has its own local classifier to predict within its class set

    Args:
        expert_id:       Unique expert identifier.
        n_queries:       Number of learnable query embeddings (L).
        embed_dim:       Query / key / value hidden dimension (d).
        agent_dim:       Input agent-output dimension per slot.
        max_local_classes: Maximum classes this expert can own.
    """

    def __init__(
        self,
        expert_id: int,
        n_queries: int = 4,
        embed_dim: int = 256,
        agent_dim: int = 256,
        max_local_classes: int = 20,
    ):
        super().__init__()

        self.expert_id = expert_id
        self.n_queries = n_queries
        self.embed_dim = embed_dim
        self.agent_dim = agent_dim
        self.max_local_classes = max_local_classes

        # ── Routing: learnable query embeddings ──
        self.queries = nn.Parameter(torch.randn(n_queries, embed_dim) * 0.02)

        # Project agent slot outputs to K, V
        self.key_proj = nn.Linear(agent_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(agent_dim, embed_dim, bias=False)

        # ── Internal classifier: predict within owned classes ──
        self.local_classifier = nn.Linear(embed_dim, max_local_classes)

        # ── Class ownership ──
        self.owned_classes: List[int] = []          # global class indices
        self._global_to_local: Dict[int, int] = {}  # global → local index
        self._local_to_global: Dict[int, int] = {}  # local → global index

        # ── Statistics ──
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.total_assigned: int = 0

    def add_class(self, global_class: int):
        """Register a new class for this expert."""
        if global_class in self._global_to_local:
            return  # already owned
        local_idx = len(self.owned_classes)
        assert local_idx < self.max_local_classes, (
            f"Expert {self.expert_id} exceeded max_local_classes "
            f"({self.max_local_classes})"
        )
        self.owned_classes.append(global_class)
        self._global_to_local[global_class] = local_idx
        self._local_to_global[local_idx] = global_class

    def owns_class(self, global_class: int) -> bool:
        """Check if this expert owns a given class."""
        return global_class in self._global_to_local

    def global_to_local(self, global_class: int) -> int:
        """Map global class index to local index within this expert."""
        return self._global_to_local[global_class]

    def local_to_global(self, local_class: int) -> int:
        """Map local class index back to global class index."""
        return self._local_to_global[local_class]

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: expert queries attend to agent slot outputs.

        Args:
            H: (B, num_slots, agent_dim)

        Returns:
            z:       (B, embed_dim)  aggregated representation
            entropy: (B,)            attention entropy — routing confidence
        """
        B, k, _ = H.shape

        K = self.key_proj(H)   # (B, k, d)
        V = self.val_proj(H)   # (B, k, d)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, d)

        # Scaled dot-product attention
        scale = math.sqrt(self.embed_dim)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, L, k)
        attn = F.softmax(scores, dim=-1)                    # (B, L, k)

        # Mean over L queries → single representation
        z = torch.bmm(attn, V).mean(dim=1)                  # (B, d)

        # Attention entropy — OOD diagnostic
        attn_avg = attn.mean(dim=1)                         # (B, k)
        entropy = -(attn_avg * (attn_avg + 1e-8).log()).sum(dim=-1)  # (B,)

        return z, entropy

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classify using internal classifier (only over owned classes).

        Args:
            z: (B, embed_dim)

        Returns:
            logits: (B, num_owned_classes) — masked to only owned indices
        """
        all_logits = self.local_classifier(z)  # (B, max_local_classes)
        n_owned = len(self.owned_classes)
        # Only return logits for owned classes
        return all_logits[:, :n_owned]

    def get_info(self) -> Dict:
        """Expert statistics."""
        return {
            "expert_id": self.expert_id,
            "owned_classes": list(self.owned_classes),
            "num_classes": len(self.owned_classes),
            "total_assigned": self.total_assigned,
            "class_counts": dict(self.class_counts),
            "n_queries": self.n_queries,
        }


class CRPExpertAggregator(nn.Module):
    """
    CRP Expert Aggregator with Exclusive Class Ownership.

    Each expert owns a non-overlapping set of classes and has its own
    internal classifier.  Routing is hard (pick best expert).
    Training uses a dual loss:
        - Classification loss (within correct expert)
        - Contrastive routing loss (when misrouted)

    Pipeline for each sample:
        1. Reshape features → H: (1, num_slots, agent_dim).
        2. Run all N experts on H → routing scores (||z||).
        3. Find correct_expert (the one owning the label).
           If none → CRP creates new expert, assigns class to it.
        4. Classification: correct_expert classifies → CE loss.
        5. Routing: if routed_expert ≠ correct_expert →
           contrastive loss = max(0, margin + score_wrong - score_correct).
        6. Backprop total loss → update expert queries + classifiers.

    Args:
        feature_dim:         Total input dim = num_slots × agent_dim.
        num_slots:           Number of slots (S).
        agent_dim:           Agent output dimension per slot.
        num_classes:         Maximum total number of classes.
        alpha:               CRP concentration. Higher → more new experts.
        max_experts:         Hard cap on expert pool size.
        n_queries:           Learnable query embeddings per expert (L).
        embed_dim:           Cross-attention hidden dimension (d).
        expert_lr:           Learning rate for all learnable parameters.
        routing_margin:      Margin for contrastive routing loss.
        routing_weight:      Weight for routing loss relative to classification.
        classes_per_expert:  Max classes each expert can handle.
        device:              Torch device.
    """

    def __init__(
        self,
        feature_dim: int,
        num_slots: int,
        agent_dim: int,
        num_classes: int = 200,
        alpha: float = 1.0,
        max_experts: int = 50,
        n_queries: int = 4,
        embed_dim: int = 256,
        expert_lr: float = 1e-3,
        routing_margin: float = 1.0,
        routing_weight: float = 0.5,
        classes_per_expert: int = 10,
        device: str = "cpu",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.agent_dim = agent_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.max_experts = max_experts
        self.n_queries = n_queries
        self.embed_dim = embed_dim
        self.expert_lr = expert_lr
        self.routing_margin = routing_margin
        self.routing_weight = routing_weight
        self.classes_per_expert = classes_per_expert
        self.device = device

        # Expert pool — grows dynamically via CRP
        self.experts = nn.ModuleList()

        # Global class → expert mapping (for fast lookup)
        self._class_to_expert: Dict[int, int] = {}  # global_class → expert_idx

        # Global statistics
        self.total_samples: int = 0
        self.num_classes_seen: int = 0
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.seen_classes: Set[int] = set()

        self._next_expert_id: int = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._frozen_experts: Set[int] = set()  # indices of frozen experts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_optimizer(self):
        """Build Adam optimizer over all current trainable parameters."""
        trainable_params = [
            p for p in self.parameters() if p.requires_grad
        ]
        if trainable_params:
            self._optimizer = torch.optim.Adam(
                trainable_params, lr=self.expert_lr
            )
        else:
            self._optimizer = None

    def _create_expert(
        self, init_from: Optional[LearnableExpert] = None
    ) -> LearnableExpert:
        """Create a new LearnableExpert, optionally warm-started."""
        expert = LearnableExpert(
            expert_id=self._next_expert_id,
            n_queries=self.n_queries,
            embed_dim=self.embed_dim,
            agent_dim=self.agent_dim,
            max_local_classes=self.classes_per_expert,
        ).to(self.device)

        if init_from is not None:
            with torch.no_grad():
                expert.queries.data.copy_(init_from.queries.data)
                expert.key_proj.weight.data.copy_(init_from.key_proj.weight.data)
                expert.val_proj.weight.data.copy_(init_from.val_proj.weight.data)
                # Small noise to break symmetry
                for p in [expert.queries, expert.key_proj.weight,
                          expert.val_proj.weight]:
                    p.add_(torch.randn_like(p) * 0.01)

        self._next_expert_id += 1
        return expert

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat feature → (B, num_slots, agent_dim)."""
        B = x.shape[0] if x.dim() > 1 else 1
        return x.view(B, self.num_slots, self.agent_dim)

    def _compute_routing_scores(
        self, H: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute routing scores for all experts.

        Args:
            H: (B, num_slots, agent_dim)

        Returns:
            all_z:    list of (B, embed_dim) — one per expert
            scores:   (B, N) — routing scores (||z||)
        """
        N = len(self.experts)
        all_z = []
        score_list = []

        for expert in self.experts:
            z, _ = expert(H)            # (B, d)
            s = z.norm(dim=-1)           # (B,)
            all_z.append(z)
            score_list.append(s)

        scores = torch.stack(score_list, dim=1)  # (B, N)
        return all_z, scores

    def _find_expert_for_class(self, label: int) -> Optional[int]:
        """Find which expert index owns a given class."""
        return self._class_to_expert.get(label, None)

    def _assign_class_to_expert(self, label: int, expert_idx: int):
        """Assign a class to an expert (exclusive ownership)."""
        self._class_to_expert[label] = expert_idx
        self.experts[expert_idx].add_class(label)

    def _ensure_expert_for_class(self, label: int) -> int:
        """
        Ensure there is an expert that owns this class.
        Assigns classes in INTERLEAVED order: new classes go to the
        unfrozen expert with the fewest owned classes (round-robin).
        If all unfrozen experts are full, create a new one.
        """
        existing_idx = self._find_expert_for_class(label)
        if existing_idx is not None:
            return existing_idx

        # Find unfrozen expert with fewest classes (interleaved assignment)
        best_idx = None
        best_count = float('inf')
        for idx in range(len(self.experts)):
            if idx in self._frozen_experts:
                continue
            expert = self.experts[idx]
            n = len(expert.owned_classes)
            if n < self.classes_per_expert and n < best_count:
                best_count = n
                best_idx = idx

        if best_idx is not None:
            self._assign_class_to_expert(label, best_idx)
            return best_idx

        # No room → create new expert
        if len(self.experts) < self.max_experts:
            init_src = (self.experts[-1] if len(self.experts) > 0
                        else None)
            new_expert = self._create_expert(init_from=init_src)
            self.experts.append(new_expert)
            new_idx = len(self.experts) - 1
            self._assign_class_to_expert(label, new_idx)
            self._init_optimizer()
            return new_idx

        # Max experts reached — force-assign to the last unfrozen expert
        for idx in reversed(range(len(self.experts))):
            if idx not in self._frozen_experts:
                self._assign_class_to_expert(label, idx)
                return idx

        raise RuntimeError(
            "All experts are frozen and at max capacity. "
            "Cannot assign new class."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeze_old_classes(self, old_classes: Set[int]):
        """
        Freeze old experts' CLASSIFIERS only (not routing params).
        Call at the start of each new task to protect previous knowledge.

        Only local_classifier is frozen (preserves classification knowledge).
        Routing params (queries, key_proj, val_proj) stay trainable so
        old experts can still compete for routing against new experts.

        Args:
            old_classes: Set of class labels from previous tasks.
        """
        for idx, expert in enumerate(self.experts):
            expert_classes = set(expert.owned_classes)
            if expert_classes and expert_classes.issubset(old_classes):
                self._frozen_experts.add(idx)
                # Only freeze the classifier — keep routing params trainable
                for p in expert.local_classifier.parameters():
                    p.requires_grad_(False)

        # Rebuild optimizer with only trainable params
        self._init_optimizer()

    def learn_one(self, hidden_labels: np.ndarray, label: int) -> Dict:
        """
        Online learning: one example with exclusive expert routing.

        Args:
            hidden_labels: (feature_dim,) — flattened agent outputs.
            label:         Ground-truth class label (int).

        Returns:
            info Dict with routing and classification details.
        """
        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                          # (1, F)

        # 1. Ensure correct expert exists for this label
        correct_expert_idx = self._ensure_expert_for_class(label)
        correct_expert = self.experts[correct_expert_idx]

        H = self._reshape(x)                                    # (1, S, D)

        # 2. Classification loss (from correct expert)
        z_correct, _ = correct_expert(H)   # (1, d)
        local_logits = correct_expert.classify(z_correct)  # (1, n_owned)
        local_label = correct_expert.global_to_local(label)
        local_target = torch.tensor(
            [local_label], dtype=torch.long, device=self.device
        )
        cls_loss = F.cross_entropy(local_logits, local_target)

        # 3. Contrastive routing loss (if multiple experts exist)
        routing_loss = torch.tensor(0.0, device=self.device)
        routed_expert_idx = correct_expert_idx  # default

        if len(self.experts) > 1:
            all_z, routing_scores = self._compute_routing_scores(H)
            # (1, N) → squeeze to (N,)
            scores = routing_scores.squeeze(0)
            routed_expert_idx = scores.argmax().item()

            score_correct = scores[correct_expert_idx]

            if routed_expert_idx != correct_expert_idx:
                score_wrong = scores[routed_expert_idx]
                # Margin loss: push correct up, wrong down
                routing_loss = F.relu(
                    self.routing_margin + score_wrong - score_correct
                )

        # 4. Total loss
        total_loss = cls_loss + self.routing_weight * routing_loss

        # 5. Backprop
        if self._optimizer is None:
            self._init_optimizer()

        if self._optimizer is not None:
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

        # 6. Update statistics
        correct_expert.total_assigned += 1
        correct_expert.class_counts[label] += 1

        if label not in self.seen_classes:
            self.seen_classes.add(label)
            self.num_classes_seen += 1
        self.class_counts[label] += 1
        self.total_samples += 1

        is_misrouted = (routed_expert_idx != correct_expert_idx)

        return {
            "expert_idx": correct_expert_idx,
            "routed_expert_idx": routed_expert_idx,
            "is_misrouted": is_misrouted,
            "num_experts": len(self.experts),
            "cls_loss": cls_loss.item(),
            "routing_loss": routing_loss.item() if isinstance(
                routing_loss, torch.Tensor) else routing_loss,
            "total_loss": total_loss.item(),
        }

    def predict_one(self, hidden_labels: np.ndarray) -> Optional[int]:
        """
        Predict class for one example.

        Routes to the expert with highest routing score,
        that expert classifies internally.
        """
        if len(self.experts) == 0:
            return None

        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            H = self._reshape(x)
            _, routing_scores = self._compute_routing_scores(H)
            # Pick expert with highest routing score
            best_expert_idx = routing_scores.squeeze(0).argmax().item()
            best_expert = self.experts[best_expert_idx]

            z, _ = best_expert(H)
            local_logits = best_expert.classify(z)  # (1, n_owned)
            local_pred = local_logits.argmax(dim=-1).item()

            # Map back to global class
            if local_pred < len(best_expert.owned_classes):
                return best_expert.local_to_global(local_pred)
            else:
                return None

    def predict_proba_one(
        self, hidden_labels: np.ndarray
    ) -> Dict[int, float]:
        """Predict class probabilities for one example."""
        if len(self.experts) == 0:
            return {}

        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            H = self._reshape(x)
            _, routing_scores = self._compute_routing_scores(H)
            best_expert_idx = routing_scores.squeeze(0).argmax().item()
            best_expert = self.experts[best_expert_idx]

            z, _ = best_expert(H)
            local_logits = best_expert.classify(z)
            probs = F.softmax(local_logits, dim=-1).squeeze(0)

        result = {}
        for local_idx in range(len(best_expert.owned_classes)):
            global_class = best_expert.local_to_global(local_idx)
            result[global_class] = probs[local_idx].item()
        return result

    def update_all_projection_bases(self):
        """No-op (API compatibility)."""
        pass

    def get_stats(self) -> Dict:
        """Aggregator statistics."""
        return {
            "num_experts": len(self.experts),
            "total_samples": self.total_samples,
            "num_classes_seen": self.num_classes_seen,
            "seen_classes": sorted(self.seen_classes),
            "class_counts": dict(self.class_counts),
            "frozen_experts": sorted(self._frozen_experts),
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
        self._agg = aggregator   # alias so train.py can reach freeze_old_classes

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
        >>> agg = create_aggregator('crp', feature_dim=2816, num_classes=200)
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
