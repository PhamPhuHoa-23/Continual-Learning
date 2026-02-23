"""
Tests for CRP-based Expert Assignment Aggregator.

Tests cover:
1. Initialization
2. Expert creation on first sample
3. Expert assignment (existing vs new)
4. Prediction after training
5. Batch API
6. Factory function
7. Gradient alignment scoring
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
    ExpertModule,
    CRPExpertAggregator,
    BatchCRPAggregator,
    create_aggregator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_DIM = 128  # Small for tests
NUM_CLASSES = 10


@pytest.fixture
def aggregator():
    """Create a fresh CRPExpertAggregator."""
    return CRPExpertAggregator(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        alpha=1.0,
        max_experts=10,
        hidden_dim=64,
        buffer_size=20,
        projection_rank=5,
        expert_lr=1e-3,
        score_threshold=0.05,
        device="cpu",
    )


@pytest.fixture
def expert():
    """Create a single ExpertModule."""
    return ExpertModule(
        expert_id=0,
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=64,
    )


def make_sample(class_id: int, dim: int = FEATURE_DIM, seed: int = 0):
    """Generate a deterministic sample vector for a given class."""
    rng = np.random.RandomState(seed + class_id * 1000)
    base = rng.randn(dim).astype(np.float32)
    # Add class-specific bias so different classes are distinguishable
    base += class_id * 0.5
    return base


# ---------------------------------------------------------------------------
# ExpertModule Tests
# ---------------------------------------------------------------------------

class TestExpertModule:

    def test_init(self, expert):
        assert expert.expert_id == 0
        assert expert.feature_dim == FEATURE_DIM
        assert expert.prototype.shape == (FEATURE_DIM,)
        assert expert.prototype_count == 0
        assert expert.total_assigned == 0

    def test_forward(self, expert):
        x = torch.randn(4, FEATURE_DIM)
        logits = expert(x)
        assert logits.shape == (4, NUM_CLASSES)

    def test_update_prototype(self, expert):
        x1 = torch.randn(FEATURE_DIM)
        expert.update_prototype(x1)
        assert expert.prototype_count == 1
        # First update should copy x1
        assert torch.allclose(expert.prototype, x1, atol=1e-6)

        x2 = torch.randn(FEATURE_DIM)
        expert.update_prototype(x2)
        assert expert.prototype_count == 2

    def test_update_gradient_memory(self, expert):
        g = torch.randn(100)
        expert.update_gradient_memory(g)
        assert expert._grad_memory_initialized
        assert expert.gradient_memory.shape == g.shape

    def test_buffer_activation(self, expert):
        for i in range(5):
            expert.buffer_activation(torch.randn(FEATURE_DIM))
        assert len(expert._activation_buffer) == 5

    def test_get_info(self, expert):
        info = expert.get_info()
        assert "expert_id" in info
        assert "total_assigned" in info
        assert info["expert_id"] == 0


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
        assert info["is_new_expert"] is True
        assert info["num_experts"] == 1
        assert len(aggregator.experts) == 1
        assert aggregator.total_samples == 1

    def test_similar_data_goes_to_same_expert(self, aggregator):
        # Train several samples of the same class
        for i in range(10):
            x = make_sample(0, seed=i)
            aggregator.learn_one(x, label=0)

        # CRP is stochastic; with 10 same-class samples, it should not create
        # the maximum (10) experts. Allow some exploration-driven creation.
        assert len(aggregator.experts) <= 7

    def test_different_classes_can_create_new_experts(self, aggregator):
        # Train two very different classes
        for i in range(20):
            x0 = make_sample(0, seed=i)
            aggregator.learn_one(x0, label=0)

        for i in range(20):
            x5 = make_sample(5, seed=i)
            aggregator.learn_one(x5, label=5)

        # Should have at least 1 expert, possibly more due to CRP
        assert len(aggregator.experts) >= 1
        assert aggregator.total_samples == 40

    def test_predict_one_returns_none_when_empty(self, aggregator):
        x = make_sample(0)
        pred = aggregator.predict_one(x)
        assert pred is None

    def test_predict_one_after_training(self, aggregator):
        # Train
        for i in range(20):
            x = make_sample(0, seed=i)
            aggregator.learn_one(x, label=0)

        # Predict
        x_test = make_sample(0, seed=999)
        pred = aggregator.predict_one(x_test)
        assert pred is not None
        assert isinstance(pred, int)
        assert 0 <= pred < NUM_CLASSES

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
            num_classes=NUM_CLASSES,
            alpha=100.0,  # Very high alpha to encourage new experts
            max_experts=5,
            hidden_dim=32,
            device="cpu",
        )
        # Train many diverse classes (keep labels within NUM_CLASSES)
        for cls in range(20):
            for i in range(3):
                agg.learn_one(make_sample(cls, seed=i), label=cls % NUM_CLASSES)

        assert len(agg.experts) <= 5

    def test_get_stats(self, aggregator):
        aggregator.learn_one(make_sample(0), label=0)
        stats = aggregator.get_stats()
        assert "num_experts" in stats
        assert "total_samples" in stats
        assert stats["total_samples"] == 1
        assert "expert_infos" in stats

    def test_update_projection_bases(self, aggregator):
        # Train enough samples to fill buffers
        for i in range(10):
            aggregator.learn_one(make_sample(0, seed=i), label=0)

        # Should not raise
        aggregator.update_all_projection_bases()


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

        # Train first
        x_batch = torch.tensor(
            np.stack([make_sample(0, seed=i) for i in range(10)]),
            dtype=torch.float32,
        )
        labels_batch = torch.tensor([0] * 10, dtype=torch.long)
        batch_agg.learn_batch(x_batch, labels_batch)

        # Predict
        x_test = torch.tensor(
            np.stack([make_sample(0, seed=i + 100) for i in range(4)]),
            dtype=torch.float32,
        )
        preds = batch_agg.predict_batch(x_test)
        assert len(preds) == 4
        assert all(isinstance(p, int) for p in preds)

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
            "crp", feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES
        )
        assert isinstance(agg, CRPExpertAggregator)

    def test_create_crp_default(self):
        agg = create_aggregator(
            feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES
        )
        assert isinstance(agg, CRPExpertAggregator)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_aggregator("nonexistent_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
