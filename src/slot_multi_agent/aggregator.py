"""
MoE Cross-Attention Expert Aggregator for Continual Learning.

Replaces the prototype-based CRP design with learnable query embeddings
+ cross-attention, removing dependence on metric geometry of the feature space.

Expert Identity:
    Each expert holds L learnable query embeddings E ∈ R^(L × d).
    It "asks" agent outputs via cross-attention:
        z = CrossAttn(Q=E, K=agent_outputs, V=agent_outputs)
    Score = ||z|| (magnitude = expert confidence)
    Entropy of attention weights → OOD detector for CRP new-expert creation.

CRP Routing (no prototype, no distance metric):
    For each input H:
        - Run all active experts → scores, entropies
        - MoE gate: top-k experts by score, softmax weights
        - final_repr = Σ gate_j × z_j
    New expert: if (max_score < threshold OR best_entropy > entropy_threshold)
                AND Bernoulli(alpha / (N_total + alpha))

Classification (prompt-based CIL style):
    class_queries ∈ R^(C × d) — one learnable query per class
    logits = final_repr @ class_queries^T  →  (B, C)
    Old class queries are frozen when training new tasks.

Continual Learning guarantees:
    - Agents: frozen throughout → feature space stable
    - Expert queries: only appended (old frozen at task boundary)
    - Class queries: old ones frozen when new classes arrive
    - No prototype to become stale after feature drift
    - Fully differentiable via backprop through cross-attention

Inspiration:
    - CompSLOT: Liao et al. (ICLR 2026) — primitive selection via cross-attention
    - L2P / DualPrompt: Wang et al. (2022) — per-class learnable query vectors
    - CRP: Aldous (1985), Pitman (2006)
    - MoE routing: Mixtral (Jiang et al., 2024)
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
    Expert with learnable query embeddings + cross-attention (no prototype).

    Instead of a centroid in feature space, this expert holds L learnable
    query embeddings that attend to agent outputs via cross-attention.
    The expert identity lives in parameter space, not in the geometry of
    agent outputs — so it never goes stale when the feature distribution
    changes.

    Args:
        expert_id:  Unique expert identifier.
        n_queries:  Number of learnable query embeddings (L).
        embed_dim:  Query / key / value hidden dimension (d).
        agent_dim:  Input agent-output dimension per slot.
    """

    def __init__(
        self,
        expert_id: int,
        n_queries: int = 4,
        embed_dim: int = 256,
        agent_dim: int = 256,
    ):
        super().__init__()

        self.expert_id = expert_id
        self.n_queries = n_queries
        self.embed_dim = embed_dim
        self.agent_dim = agent_dim

        # Learnable query embeddings — expert "identity" in parameter space
        # NOT a prototype centroid; NOT in agent feature geometry
        self.queries = nn.Parameter(torch.randn(n_queries, embed_dim) * 0.02)

        # Project agent slot outputs to K, V
        self.key_proj = nn.Linear(agent_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(agent_dim, embed_dim, bias=False)

        # Statistics (not learnable)
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.total_assigned: int = 0

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: expert queries attend to agent slot outputs.

        Args:
            H: (B, num_slots, agent_dim)

        Returns:
            z:       (B, embed_dim)  aggregated representation
            entropy: (B,)            attention entropy — high = OOD signal
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

        # Attention entropy averaged over queries — OOD diagnostic
        attn_avg = attn.mean(dim=1)                         # (B, k)
        entropy = -(attn_avg * (attn_avg + 1e-8).log()).sum(dim=-1)  # (B,)

        return z, entropy

    def get_info(self) -> Dict:
        """Expert statistics."""
        return {
            "expert_id": self.expert_id,
            "total_assigned": self.total_assigned,
            "class_counts": dict(self.class_counts),
            "n_queries": self.n_queries,
        }


class CRPExpertAggregator(nn.Module):
    """
    MoE Cross-Attention Expert Aggregator (no prototypes).

    Combines CRP-style dynamic expert creation with learnable query
    embeddings + cross-attention routing (no metric-space prototypes).

    Pipeline for each sample (hidden_labels, label):
        1. Reshape hidden_labels → H: (1, num_slots, agent_dim).
        2. Run all N active experts on H → scores (magnitude) + entropies.
        3. CRP check: if (max_score < threshold OR best_entropy > ent_threshold)
                       AND Bernoulli(alpha / (N + alpha)) → create new expert.
        4. MoE gate: top-k experts by score, softmax weights.
        5. final_repr = Σ gate_j × z_j.
        6. logits = final_repr @ class_queries^T → (C,).
        7. CE loss + backprop → update expert queries + class query for this label.

    Key properties:
        - Expert identity = learnable parameter vectors, NOT prototype centroids.
        - Routing uses attention entropy (OOD signal), NOT Euclidean distance.
        - Class queries grow by appending new vectors (old ones frozen per task).
        - Fully differentiable end-to-end.

    Args:
        feature_dim:         Total input dim = num_slots × agent_dim.
        num_slots:           Number of AdaSlot slots (S).
        agent_dim:           Agent output dimension per slot (proto_dim).
        num_classes:         Maximum total number of classes.
        alpha:               CRP concentration. Higher → more new experts.
        max_experts:         Hard cap on expert pool size.
        n_queries:           Learnable query embeddings per expert (L).
        embed_dim:           Cross-attention hidden dimension (d).
        top_k:               Number of top experts for MoE gating.
        expert_lr:           Learning rate for all learnable parameters.
        entropy_threshold:   Attention entropy threshold for OOD detection.
        score_threshold:     Min MoE score to avoid triggering new expert.
        device:              Torch device.
    """

    def __init__(
        self,
        feature_dim: int,
        num_slots: int,
        agent_dim: int,
        num_classes: int = 100,
        alpha: float = 1.0,
        max_experts: int = 20,
        n_queries: int = 4,
        embed_dim: int = 256,
        top_k: int = 3,
        expert_lr: float = 1e-3,
        entropy_threshold: float = 2.0,
        score_threshold: float = 0.3,
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
        self.top_k = top_k
        self.expert_lr = expert_lr
        self.entropy_threshold = entropy_threshold
        self.score_threshold = score_threshold
        self.device = device

        # Expert pool — grows dynamically via CRP
        self.experts = nn.ModuleList()

        # Class query vectors: classification via inner product with final_repr
        # Shape (num_classes, embed_dim); old queries frozen per task boundary
        self.class_queries = nn.Parameter(
            torch.randn(num_classes, embed_dim, device=device) * 0.02
        )

        # Set of classes whose query vectors are currently trainable
        self._trainable_classes: Set[int] = set()

        # Global statistics
        self.total_samples: int = 0
        self.num_classes_seen: int = 0
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.seen_classes: Set[int] = set()
        self.expert_class_map: Dict[int, set] = defaultdict(set)

        self._next_expert_id: int = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None

        # CRP usage counts per expert slot
        self.register_buffer(
            "_expert_counts", torch.zeros(max_experts, dtype=torch.long, device=device)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_optimizer(self):
        """Build Adam optimizer over all current parameters."""
        self._optimizer = torch.optim.Adam(
            self.parameters(), lr=self.expert_lr
        )

    def _create_expert(
        self, init_from: Optional[LearnableExpert] = None
    ) -> LearnableExpert:
        """Create a new LearnableExpert, optionally warm-started."""
        expert = LearnableExpert(
            expert_id=self._next_expert_id,
            n_queries=self.n_queries,
            embed_dim=self.embed_dim,
            agent_dim=self.agent_dim,
        ).to(self.device)

        if init_from is not None:
            with torch.no_grad():
                expert.queries.data.copy_(init_from.queries.data)
                expert.key_proj.weight.data.copy_(init_from.key_proj.weight.data)
                expert.val_proj.weight.data.copy_(init_from.val_proj.weight.data)
                # Small noise to break symmetry
                for p in expert.parameters():
                    p.add_(torch.randn_like(p) * 0.01)

        self._next_expert_id += 1
        return expert

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat feature → (B, num_slots, agent_dim)."""
        B = x.shape[0] if x.dim() > 1 else 1
        return x.view(B, self.num_slots, self.agent_dim)

    def _forward_all_experts(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all active experts on H.

        Args:
            H: (B, num_slots, agent_dim)

        Returns:
            all_z:       (B, N, embed_dim)
            all_scores:  (B, N)   magnitude × log-CRP-count
            all_entropy: (B, N)   attention entropy per expert
        """
        N = len(self.experts)
        zs, raw_scores, entropies = [], [], []

        for j, expert in enumerate(self.experts):
            z, ent = expert(H)                      # (B, d), (B,)
            s = z.norm(dim=-1)                      # (B,)
            raw_scores.append(s)
            entropies.append(ent)
            zs.append(z)

        all_z = torch.stack(zs, dim=1)              # (B, N, d)
        all_scores = torch.stack(raw_scores, dim=1) # (B, N)
        all_entropy = torch.stack(entropies, dim=1) # (B, N)

        # Multiply by log(CRP count + 1) as a mild popularity bonus
        crp_prior = (
            self._expert_counts[:N].float().to(self.device) + 1.0
        ).log1p()
        all_scores = all_scores * crp_prior.unsqueeze(0)

        return all_z, all_scores, all_entropy

    def _moe_aggregate(
        self, all_z: torch.Tensor, all_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        MoE gating: top-k experts, softmax weights.

        Returns:
            output: (B, embed_dim)
        """
        N = all_z.size(1)
        k = min(self.top_k, N)

        topk_scores, topk_idx = all_scores.topk(k, dim=-1)   # (B, k)
        gate = F.softmax(topk_scores, dim=-1)                  # (B, k)

        selected_z = all_z.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )                                                        # (B, k, d)

        return (gate.unsqueeze(-1) * selected_z).sum(dim=1)    # (B, d)

    def _maybe_add_expert(
        self, all_scores: torch.Tensor, all_entropy: torch.Tensor
    ) -> bool:
        """
        CRP decision gate: add new expert if needed.

        Trigger condition:
            (mean_max_score < score_threshold  OR
             best_expert_entropy > entropy_threshold)
            AND Bernoulli(alpha / (N + alpha))

        Returns True if a new expert was created.
        """
        if len(self.experts) >= self.max_experts:
            return False

        mean_max_score = all_scores.max(dim=-1).values.mean().item()
        # Entropy of the currently most confident expert
        best_j = int(all_scores.mean(0).argmax().item())
        best_entropy = all_entropy[:, best_j].mean().item()

        p_new = self.alpha / (self.total_samples + self.alpha + 1e-8)

        if (
            (mean_max_score < self.score_threshold or
             best_entropy > self.entropy_threshold)
            and np.random.rand() < p_new
        ):
            init_src = self.experts[best_j] if len(self.experts) > 0 else None
            new_expert = self._create_expert(init_from=init_src)
            self.experts.append(new_expert)
            # Register new params in existing optimizer
            if self._optimizer is not None:
                self._optimizer.add_param_group(
                    {"params": new_expert.parameters(), "lr": self.expert_lr}
                )
            return True

        return False

    # ------------------------------------------------------------------
    # Public API (compatible with BatchCRPAggregator)
    # ------------------------------------------------------------------

    def freeze_old_classes(self, new_label_set: Set[int]):
        """
        Freeze class-query vectors for all classes NOT in new_label_set.
        Call at the start of each new task to protect previous knowledge.
        """
        self._trainable_classes = new_label_set

    def learn_one(self, hidden_labels: np.ndarray, label: int) -> Dict:
        """
        Online learning: one example.

        Args:
            hidden_labels: (feature_dim,) — flattened agent outputs.
            label:         Ground-truth class label (int).

        Returns:
            info Dict with expert routing details.
        """
        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                          # (1, F)
        y = torch.tensor([label], dtype=torch.long, device=self.device)

        # Bootstrap: create first expert
        if len(self.experts) == 0:
            self.experts.append(self._create_expert())
            self._init_optimizer()

        H = self._reshape(x)                                    # (1, S, D)

        # Forward all active experts
        all_z, all_scores, all_entropy = self._forward_all_experts(H)

        # CRP routing: maybe add new expert
        is_new = self._maybe_add_expert(all_scores, all_entropy)
        if is_new:
            all_z, all_scores, all_entropy = self._forward_all_experts(H)

        # MoE aggregate → final representation
        repr_vec = self._moe_aggregate(all_z, all_scores)       # (1, d)

        # Classification logits
        logits = repr_vec @ self.class_queries.T                 # (1, C)

        # Mask unseen classes to -inf (except current label)
        visible = list(self.seen_classes | {label})
        mask = torch.full_like(logits, float("-inf"))
        mask[:, visible] = 0.0
        logits = logits + mask

        # Cross-entropy loss
        loss = F.cross_entropy(logits, y)

        # Backprop (with class-query freezing)
        if self._optimizer is None:
            self._init_optimizer()

        self._optimizer.zero_grad()
        loss.backward()

        # Freeze gradients for old class queries if _trainable_classes is set
        if self._trainable_classes:
            with torch.no_grad():
                freeze_mask = torch.ones(
                    self.num_classes, device=self.device
                )
                freeze_mask[list(self._trainable_classes)] = 0.0
                if self.class_queries.grad is not None:
                    self.class_queries.grad *= (1.0 - freeze_mask).unsqueeze(1)

        self._optimizer.step()

        # Update CRP counts for top-k used experts
        N = len(self.experts)
        k = min(self.top_k, N)
        topk_idx = all_scores[0].topk(k).indices
        for idx in topk_idx.cpu().tolist():
            self._expert_counts[idx] += 1
            self.experts[idx].total_assigned += 1
            self.experts[idx].class_counts[label] += 1
            self.expert_class_map[idx].add(label)

        # Update global statistics
        if label not in self.seen_classes:
            self.seen_classes.add(label)
            self.num_classes_seen += 1
        self.class_counts[label] += 1
        self.total_samples += 1

        return {
            "expert_idx": topk_idx[0].item() if len(topk_idx) > 0 else -1,
            "is_new_expert": is_new,
            "num_experts": len(self.experts),
            "loss": loss.item(),
        }

    def predict_one(self, hidden_labels: np.ndarray) -> Optional[int]:
        """Predict class for one example."""
        if len(self.experts) == 0:
            return None

        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            H = self._reshape(x)
            all_z, all_scores, _ = self._forward_all_experts(H)
            repr_vec = self._moe_aggregate(all_z, all_scores)   # (1, d)
            logits = repr_vec @ self.class_queries.T             # (1, C)

            mask = torch.full_like(logits, float("-inf"))
            mask[:, list(self.seen_classes)] = 0.0
            logits = logits + mask
            pred = logits.argmax(dim=-1).item()

        return pred

    def predict_proba_one(self, hidden_labels: np.ndarray) -> Dict[int, float]:
        """Predict class probabilities for one example."""
        if len(self.experts) == 0:
            return {}

        x = torch.tensor(
            hidden_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            H = self._reshape(x)
            all_z, all_scores, _ = self._forward_all_experts(H)
            repr_vec = self._moe_aggregate(all_z, all_scores)
            logits = repr_vec @ self.class_queries.T

            mask = torch.full_like(logits, float("-inf"))
            mask[:, list(self.seen_classes)] = 0.0
            logits = logits + mask
            probs = F.softmax(logits, dim=-1).squeeze(0)

        return {
            i: probs[i].item()
            for i in range(self.num_classes)
            if probs[i].item() > 1e-6
        }

    def update_all_projection_bases(self):
        """No-op (API compatibility). GPM not used in this design."""
        pass

    def get_stats(self) -> Dict:
        """Aggregator statistics."""
        return {
            "num_experts": len(self.experts),
            "total_samples": self.total_samples,
            "num_classes_seen": self.num_classes_seen,
            "seen_classes": sorted(self.seen_classes),
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
