"""
Slot-based Multi-Agent System for Continual Learning.

Architecture:
1. Slot Attention → Decompose image into object-centric slots
2. Performance Estimators → Estimate agent performance on each slot (VAE/MLP)
3. Bandit Selection → Select and weight agents (UCB/Thompson/Epsilon-Greedy)
4. Atomic Agents → Process slots, output hidden labels (DINO-trained)
5. Incremental Tree → Aggregate hidden labels → final class (online learning)

Training Pipeline:
    Phase 1: Train agents with DINO SSL (unsupervised)
        - Student-teacher architecture with EMA
        - Each agent learns K prototypes
        - Outputs: softmax over prototypes (hidden label)
    
    Phase 2: Train tree incrementally (supervised)
        - Freeze agents
        - Extract hidden labels for each example
        - Tree learns: combination of hidden labels → class
        - Supports new classes without retraining agents

Key Features:
    - Object-centric decomposition (Slot Attention)
    - Self-supervised agent training (DINO)
    - Exploration-exploitation (Bandit algorithms)
    - Incremental learning (Hoeffding Tree)
    - No catastrophic forgetting
    - No task ID required at test time

Quick Start:
    >>> from src.slot_multi_agent import (
    ...     create_agent_pool,
    ...     create_estimator_pool,
    ...     create_bandit_selector,
    ...     create_aggregator
    ... )
    >>> 
    >>> # Create components
    >>> student_agents, teacher_agents = create_agent_pool(
    ...     num_agents=50,
    ...     slot_dim=64,
    ...     num_prototypes=256
    ... )
    >>> 
    >>> estimators = create_estimator_pool(
    ...     num_agents=50,
    ...     estimator_type='vae',
    ...     slot_dim=64
    ... )
    >>> 
    >>> bandit_selector = create_bandit_selector(
    ...     strategy='ucb',
    ...     num_agents=50
    ... )
    >>> 
    >>> aggregator = create_aggregator(
    ...     aggregator_type='hoeffding_adaptive'
    ... )
"""

# Atomic Agents (DINO-trained)
from .atomic_agent import (
    ResidualMLPAgent,
    DINOLoss,
    create_agent_pool,
    update_teacher,
    update_all_teachers
)

# Performance Estimators
from .estimators import (
    VAEEstimator,
    MLPEstimator,
    HybridEstimator,
    create_estimator_pool
)

# Legacy Selectors (simple top-k)
from .selector import (
    TopKAgentSelector,
    AdaptiveKSelector,
    create_selector
)

# Bandit-based Selectors (with exploration-exploitation)
from .bandit_selector import (
    BanditSelector,
    UCBSelector,
    ThompsonSamplingSelector,
    EpsilonGreedySelector,
    WeightedTopKSelector,
    create_bandit_selector
)

# Aggregators (Incremental Decision Trees)
from .aggregator import (
    IncrementalTreeAggregator,
    EnsembleTreeAggregator,
    BatchTreeAggregator,
    create_aggregator
)

__all__ = [
    # Atomic Agents
    'ResidualMLPAgent',
    'DINOLoss',
    'create_agent_pool',
    'update_teacher',
    'update_all_teachers',
    
    # Estimators
    'VAEEstimator',
    'MLPEstimator',
    'HybridEstimator',
    'create_estimator_pool',
    
    # Legacy Selectors
    'TopKAgentSelector',
    'AdaptiveKSelector',
    'create_selector',
    
    # Bandit Selectors
    'BanditSelector',
    'UCBSelector',
    'ThompsonSamplingSelector',
    'EpsilonGreedySelector',
    'WeightedTopKSelector',
    'create_bandit_selector',
    
    # Aggregators
    'IncrementalTreeAggregator',
    'EnsembleTreeAggregator',
    'BatchTreeAggregator',
    'create_aggregator',
]
