"""Configuration schemas and validation."""

from typing import Any, Dict
from cont_src.config.base import Config


def validate_config(config: Config) -> bool:
    """
    Validate configuration for consistency.

    Args:
        config: Configuration to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check dimension consistency
    if config.agent.input_dim != config.slot_attention.slot_dim:
        raise ValueError(
            f"Agent input_dim ({config.agent.input_dim}) must match "
            f"slot_attention slot_dim ({config.slot_attention.slot_dim})"
        )

    if config.agent.output_dim != config.aggregator.hidden_dim:
        raise ValueError(
            f"Agent output_dim ({config.agent.output_dim}) must match "
            f"aggregator hidden_dim ({config.aggregator.hidden_dim})"
        )

    # Check loss weights
    if config.loss.weight_primitive < 0:
        raise ValueError("Loss weights must be non-negative")

    # Check task configuration
    if config.data.n_tasks < 1:
        raise ValueError("Number of tasks must be at least 1")

    if config.data.n_classes_per_task < 1:
        raise ValueError("Classes per task must be at least 1")

    # Check clustering
    if config.clustering.n_clusters_task1 < 1:
        raise ValueError("Number of clusters must be at least 1")

    return True


def auto_fix_config(config: Config) -> Config:
    """
    Automatically fix common configuration issues.

    Args:
        config: Configuration to fix

    Returns:
        Fixed configuration
    """
    # Auto-match dimensions
    config.agent.input_dim = config.slot_attention.slot_dim
    config.aggregator.hidden_dim = config.agent.output_dim

    return config
