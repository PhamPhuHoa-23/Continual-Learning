"""Configuration module."""

from cont_src.config.base import (
    Config,
    BackboneConfig,
    SlotAttentionConfig,
    AgentConfig,
    RouterConfig,
    AggregatorConfig,
    ClassifierConfig,
    LossConfig,
    ClusteringConfig,
    DataConfig,
    TrainingConfig,
)
from cont_src.config.defaults import get_config, get_default_config
from cont_src.config.schema import validate_config, auto_fix_config

__all__ = [
    "Config",
    "BackboneConfig",
    "SlotAttentionConfig",
    "AgentConfig",
    "RouterConfig",
    "AggregatorConfig",
    "ClassifierConfig",
    "LossConfig",
    "ClusteringConfig",
    "DataConfig",
    "TrainingConfig",
    "get_config",
    "get_default_config",
    "validate_config",
    "auto_fix_config",
]
