"""
Tests for CRP-based Expert Assignment Aggregator (Exclusive Ownership Design).

Tests cover:
1. Initialization
2. Expert creation on first sample
3. Exclusive class ownership
4. Prediction after training
5. Batch API
6. Factory function
7. Routing and contrastive loss
8. get_stats
"""

import sys
import os
import torch
import numpy as np
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.slot_multi_agent.aggregator import (
    LearnableExpert,
    CRPExpertAggregator,
    BatchCRPAggregator,
    create_aggregator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_SLOTS = 4
AGENT_DIM = 32
FEATURE_DIM = NUM_SLOTS * AGENT_DIM  # = 128
NUM_CLASSES = 10


@pytest.fixture
def aggregator():
    """Create a fresh CRPExpertAggregator."""
    return CRPExpertAggregator(
        feature_dim=FEATURE_DIM,
        num_slots=NUM_SLOTS,
        agent_dim=AGENT_DIM,
        num_classes=NUM_CLASSES,
        alpha=1.0,
        max_experts=10,
        n_queries=4,
        embed_dim=64,
        expert_lr=1e-3,
        routing_margin=1.0,
        routing_weight=0.5,
        classes_per_expert=5,
        device="cpu",
    )


@pytest.fixture
def expert():
    """Create a single LearnableExpert."""
    return LearnableExpert(
        expert_id=0,
        n_queries=4,
        embed_dim=64,
        agent_dim=AGENT_DIM,
        max_local_classes=5,
    )


def make_sample(class_id: int, dim: int = FEATURE_DIM, seed: int = 0):
    """Generate a deterministic sample vector for a given class."""
    rng = np.random.RandomState(seed + class_id * 1000)
    base = rng.randn(dim).astype(np.float32)
    # Add class-specific bias so different classes are distinguishable
    base += class_id * 0.5
    return base


# ---------------------------------------------------------------------------
# LearnableExpert Tests
# ---------------------------------------------------------------------------

class TestLearnableExpert:

    def test_init(self, expert):
        assert expert.expert_id == 0
        assert expert.n_queries == 4
        assert expert.embed_dim == 64
        assert expert.agent_dim == AGENT_DIM
        assert len(expert.owned_classes) == 0

    def test_forward(self, expert):
        H = torch.randn(2, NUM_SLOTS, AGENT_DIM)
        z, entropy = expert(H)
        assert z.shape == (2, 64)
        assert entropy.shape == (2,)

    def test_add_class(self, expert):
        expert.add_class(5)
        assert expert.owns_class(5)
        assert expert.global_to_local(5) == 0
        assert expert.local_to_global(0) == 5

        expert.add_class(10)
        assert expert.owns_class(10)
        assert expert.global_to_local(10) == 1
        assert len(expert.owned_classes) == 2

    def test_add_class_idempotent(self, expert):
        expert.add_class(3)
        expert.add_class(3)  # should not duplicate
        assert len(expert.owned_classes) == 1

    def test_classify(self, expert):
        expert.add_class(0)
        expert.add_class(1)
        H = torch.randn(2, NUM_SLOTS, AGENT_DIM)
        z, _ = expert(H)
        logits = expert.classify(z)
        assert logits.shape == (2, 2)  # only 2 owned classes

    def test_get_info(self, expert):
        expert.add_class(5)
        info = expert.get_info()
        assert "expert_id" in info
        assert "owned_classes" in info
        assert info["expert_id"] == 0
        assert 5 in info["owned_classes"]


# ---------------------------------------------------------------------------
# CRPExpertAggregator Tests
# ---------------------------------------------------------------------------

class TestCRPExpertAggregator:

    def test_init_empty(self, aggregator):
        assert len(aggregator.experts) == 0
        assert aggregator.total_samples == 0
        assert aggregator.num_classes_seen == 0

    def test_first_learn_creates_expert(self, aggregator):
        x = make_sample(0)
        info = aggregator.learn_one(x, label=0)
        assert info["num_experts"] == 1
        assert len(aggregator.experts) == 1
        assert aggregator.total_samples == 1
        assert aggregator.experts[0].owns_class(0)

    def test_same_class_goes_to_same_expert(self, aggregator):
        for i in range(10):
            x = make_sample(0, seed=i)
            aggregator.learn_one(x, label=0)
        # All samples of class 0 should be in the same expert
        assert len(aggregator.experts) == 1

    def test_exclusive_class_ownership(self, aggregator):
        # Train classes 0-4 → should go to expert 0 (classes_per_expert=5)
        for cls in range(5):
            for i in range(3):
                aggregator.learn_one(make_sample(cls, seed=i), label=cls)

        # All 5 classes should be owned by the same expert
        expert_0 = aggregator.experts[0]
        for cls in range(5):
            assert expert_0.owns_class(cls)

        # Train class 5 → should create a new expert (expert 0 is full)
        aggregator.learn_one(make_sample(5), label=5)
        assert len(aggregator.experts) == 2
        assert aggregator.experts[1].owns_class(5)
        assert not aggregator.experts[0].owns_class(5)

    def test_no_class_overlap(self, aggregator):
        """No two experts should own the same class."""
        for cls in range(8):
            for i in range(3):
                aggregator.learn_one(make_sample(cls, seed=i), label=cls)

        # Check exclusive ownership
        all_classes = set()
        for expert in aggregator.experts:
            expert_classes = set(expert.owned_classes)
            assert expert_classes.isdisjoint(all_classes), \
                f"Expert {expert.expert_id} has overlapping classes"
            all_classes |= expert_classes

    def test_predict_one_returns_none_when_empty(self, aggregator):
        x = make_sample(0)
        pred = aggregator.predict_one(x)
        assert pred is None

    def test_predict_one_after_training(self, aggregator):
        for i in range(20):
            x = make_sample(0, seed=i)
            aggregator.learn_one(x, label=0)

        x_test = make_sample(0, seed=999)
        pred = aggregator.predict_one(x_test)
        assert pred is not None
        assert isinstance(pred, int)

    def test_predict_proba_one(self, aggregator):
        for i in range(10):
            aggregator.learn_one(make_sample(0, seed=i), label=0)

        proba = aggregator.predict_proba_one(make_sample(0, seed=999))
        assert isinstance(proba, dict)
        if len(proba) > 0:
            assert all(0 <= v <= 1 for v in proba.values())

    def test_max_experts_limit(self):
        agg = CRPExpertAggregator(
            feature_dim=FEATURE_DIM,
            num_slots=NUM_SLOTS,
            agent_dim=AGENT_DIM,
            num_classes=NUM_CLASSES,
            alpha=100.0,
            max_experts=3,
            classes_per_expert=2,
            device="cpu",
        )
        # Train many classes
        for cls in range(8):
            for i in range(3):
                agg.learn_one(make_sample(cls, seed=i), label=cls % NUM_CLASSES)

        assert len(agg.experts) <= 3

    def test_routing_loss_on_misroute(self, aggregator):
        # Train two experts with different classes
        for i in range(20):
            aggregator.learn_one(make_sample(0, seed=i), label=0)
        for cls in range(1, 5):
            aggregator.learn_one(make_sample(cls, seed=0), label=cls)

        # Classes 5+ go to a new expert
        for i in range(5, 8):
            aggregator.learn_one(make_sample(i, seed=0), label=i)

        if len(aggregator.experts) > 1:
            # Train more and check that routing_loss is reported
            info = aggregator.learn_one(make_sample(0, seed=999), label=0)
            assert "routing_loss" in info
            assert "cls_loss" in info

    def test_get_stats(self, aggregator):
        aggregator.learn_one(make_sample(0), label=0)
        stats = aggregator.get_stats()
        assert "num_experts" in stats
        assert "total_samples" in stats
        assert stats["total_samples"] == 1
        assert "expert_infos" in stats

    def test_update_projection_bases(self, aggregator):
        for i in range(10):
            aggregator.learn_one(make_sample(0, seed=i), label=0)
        # Should not raise
        aggregator.update_all_projection_bases()

    def test_freeze_old_classes(self, aggregator):
        # Train expert on classes 0-4
        for cls in range(5):
            for i in range(3):
                aggregator.learn_one(make_sample(cls, seed=i), label=cls)

        # Freeze old classes (emulate starting new task with classes 5-9)
        old_classes = {0, 1, 2, 3, 4}
        aggregator.freeze_old_classes(old_classes)

        # Expert 0 should be frozen
        assert 0 in aggregator._frozen_experts
        for p in aggregator.experts[0].parameters():
            assert p.requires_grad is False


# ---------------------------------------------------------------------------
# BatchCRPAggregator Tests
# ---------------------------------------------------------------------------

class TestBatchCRPAggregator:

    def test_learn_batch(self, aggregator):
        batch_agg = BatchCRPAggregator(aggregator)

        x_batch = torch.tensor(
            np.stack([make_sample(i % 3, seed=i) for i in range(8)]),
            dtype=torch.float32,
        )
        labels_batch = torch.tensor([i % 3 for i in range(8)], dtype=torch.long)

        batch_agg.learn_batch(x_batch, labels_batch)
        assert aggregator.total_samples == 8

    def test_predict_batch(self, aggregator):
        batch_agg = BatchCRPAggregator(aggregator)

        x_batch = torch.tensor(
            np.stack([make_sample(0, seed=i) for i in range(10)]),
            dtype=torch.float32,
        )
        labels_batch = torch.tensor([0] * 10, dtype=torch.long)
        batch_agg.learn_batch(x_batch, labels_batch)

        x_test = torch.tensor(
            np.stack([make_sample(0, seed=i + 100) for i in range(4)]),
            dtype=torch.float32,
        )
        preds = batch_agg.predict_batch(x_test)
        assert len(preds) == 4

    def test_predict_proba_batch(self, aggregator):
        batch_agg = BatchCRPAggregator(aggregator)

        x_batch = torch.tensor(
            np.stack([make_sample(0, seed=i) for i in range(5)]),
            dtype=torch.float32,
        )
        labels_batch = torch.tensor([0] * 5, dtype=torch.long)
        batch_agg.learn_batch(x_batch, labels_batch)

        probas = batch_agg.predict_proba_batch(x_batch)
        assert len(probas) == 5
        assert all(isinstance(p, dict) for p in probas)

    def test_get_stats(self, aggregator):
        batch_agg = BatchCRPAggregator(aggregator)
        stats = batch_agg.get_stats()
        assert "num_experts" in stats


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------

class TestCreateAggregator:

    def test_create_crp(self):
        agg = create_aggregator(
            "crp",
            feature_dim=FEATURE_DIM,
            num_slots=NUM_SLOTS,
            agent_dim=AGENT_DIM,
            num_classes=NUM_CLASSES,
        )
        assert isinstance(agg, BatchCRPAggregator)

    def test_create_crp_default(self):
        agg = create_aggregator(
            feature_dim=FEATURE_DIM,
            num_slots=NUM_SLOTS,
            agent_dim=AGENT_DIM,
            num_classes=NUM_CLASSES,
        )
        assert isinstance(agg, BatchCRPAggregator)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_aggregator("nonexistent_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
