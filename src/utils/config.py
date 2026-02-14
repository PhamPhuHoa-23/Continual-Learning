"""
Configuration utilities for loading and validating config files.

Usage:
    >>> from src.utils.config import load_config, Config
    >>> 
    >>> # Load config from YAML
    >>> cfg = load_config('config.yaml')
    >>> 
    >>> # Access nested values
    >>> print(cfg.agents.num_agents)  # 50
    >>> print(cfg.slot_attention.adaptive)  # True
    >>> print(cfg.selection.strategy)  # 'topk_estimator'
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import torch


class DotDict(dict):
    """
    Dictionary with dot notation access.
    
    Example:
        >>> d = DotDict({'a': {'b': 1}})
        >>> print(d.a.b)  # 1
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, key):
        try:
            value = self[key]
            return value
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")


class Config(DotDict):
    """
    Configuration class with validation.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.validate()
    
    def validate(self):
        """Validate configuration values."""
        
        # Validate device
        if self.device.type not in ['cpu', 'cuda'] and not self.device.type.startswith('cuda:'):
            raise ValueError(f"Invalid device: {self.device.type}")
        
        # Check CUDA availability
        if self.device.type.startswith('cuda') and not torch.cuda.is_available():
            print(f"WARNING: CUDA not available, falling back to CPU")
            self.device.type = 'cpu'
        
        # Validate slot attention
        if self.slot_attention.adaptive:
            assert self.slot_attention.min_slots > 0, "min_slots must be > 0"
            assert self.slot_attention.max_slots >= self.slot_attention.min_slots, \
                "max_slots must be >= min_slots"
        else:
            assert hasattr(self.slot_attention, 'num_slots'), \
                "num_slots required when adaptive=False"
        
        # Validate agents
        assert self.agents.num_agents > 0, "num_agents must be > 0"
        assert self.agents.num_prototypes > 0, "num_prototypes must be > 0"
        assert self.agents.slot_dim == self.slot_attention.slot_dim, \
            "agent slot_dim must match slot_attention slot_dim"
        
        # Validate selection
        assert self.selection.strategy in [
            'topk_estimator',
            'bandit_ucb',
            'bandit_thompson',
            'bandit_epsilon_greedy',
            'weighted_topk'
        ], f"Invalid selection strategy: {self.selection.strategy}"
        
        assert self.selection.k > 0 and self.selection.k <= self.agents.num_agents, \
            f"k must be in [1, {self.agents.num_agents}]"
        
        # Validate aggregator
        assert self.aggregator.type in [
            'hoeffding',
            'hoeffding_adaptive',
            'ensemble'
        ], f"Invalid aggregator type: {self.aggregator.type}"
        
        # Validate data
        assert self.data.dataset in [
            'cifar10',
            'cifar100',
            'tiny_imagenet',
            'imagenet'
        ], f"Invalid dataset: {self.data.dataset}"
        
        assert self.data.continual_learning.scenario in [
            'class_incremental',
            'domain_incremental',
            'task_incremental'
        ], f"Invalid CL scenario: {self.data.continual_learning.scenario}"
        
        print("✓ Configuration validated successfully")
    
    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device.type)
    
    def is_bandit_selection(self) -> bool:
        """Check if using bandit-based selection."""
        return self.selection.strategy.startswith('bandit_')
    
    def get_num_slots(self) -> Optional[int]:
        """Get number of slots (None if adaptive)."""
        if self.slot_attention.adaptive:
            return None
        else:
            return self.slot_attention.num_slots
    
    def get_slot_range(self) -> tuple[int, int]:
        """Get slot range (min, max)."""
        if self.slot_attention.adaptive:
            return (self.slot_attention.min_slots, self.slot_attention.max_slots)
        else:
            n = self.slot_attention.num_slots
            return (n, n)
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n[Project]")
        print(f"  Name: {self.project.name}")
        print(f"  Version: {self.project.version}")
        
        print(f"\n[Device]")
        print(f"  Type: {self.device.type}")
        print(f"  Mixed Precision: {self.device.mixed_precision}")
        
        print(f"\n[Slot Attention]")
        if self.slot_attention.adaptive:
            print(f"  Mode: Adaptive (AdaSlot)")
            print(f"  Slot Range: {self.slot_attention.min_slots}-{self.slot_attention.max_slots}")
        else:
            print(f"  Mode: Fixed")
            print(f"  Num Slots: {self.slot_attention.num_slots}")
        print(f"  Slot Dim: {self.slot_attention.slot_dim}")
        print(f"  Iterations: {self.slot_attention.num_iterations}")
        
        print(f"\n[Agents]")
        print(f"  Num Agents: {self.agents.num_agents}")
        print(f"  Architecture: {self.agents.architecture}")
        print(f"  Hidden Dim: {self.agents.hidden_dim}")
        print(f"  Num Prototypes: {self.agents.num_prototypes}")
        print(f"  DINO Teacher Temp: {self.agents.dino.teacher_temp}")
        print(f"  DINO Student Temp: {self.agents.dino.student_temp}")
        
        print(f"\n[Performance Estimators]")
        print(f"  Type: {self.estimators.type}")
        if self.estimators.type == 'vae':
            print(f"  VAE Latent Dim: {self.estimators.vae.latent_dim}")
        
        print(f"\n[Agent Selection]")
        print(f"  Strategy: {self.selection.strategy}")
        print(f"  Top-k: {self.selection.k}")
        if self.is_bandit_selection():
            strategy_short = self.selection.strategy.replace('bandit_', '')
            if strategy_short == 'ucb':
                print(f"  UCB Exploration: {self.selection.bandit.ucb.exploration_constant}")
            elif strategy_short == 'epsilon_greedy':
                print(f"  Epsilon: {self.selection.bandit.epsilon_greedy.epsilon}")
        
        print(f"\n[Aggregator]")
        print(f"  Type: {self.aggregator.type}")
        print(f"  Grace Period: {self.aggregator.hoeffding.grace_period}")
        print(f"  Split Confidence: {self.aggregator.hoeffding.split_confidence}")
        print(f"  Leaf Prediction: {self.aggregator.hoeffding.leaf_prediction}")
        
        print(f"\n[Training]")
        print(f"  Phase 1 (Agents):")
        print(f"    Epochs: {self.training.phase1_agents.epochs}")
        print(f"    Batch Size: {self.training.phase1_agents.batch_size}")
        print(f"    Learning Rate: {self.training.phase1_agents.learning_rate}")
        print(f"    Weight Decay: {self.training.phase1_agents.weight_decay}")
        print(f"  Phase 2 (Tree): Incremental (online)")
        
        print(f"\n[Data]")
        print(f"  Dataset: {self.data.dataset}")
        print(f"  CL Scenario: {self.data.continual_learning.scenario}")
        print(f"  Num Experiences: {self.data.continual_learning.num_experiences}")
        
        print(f"\n[Experiment]")
        print(f"  Name: {self.experiment.name}")
        print(f"  Description: {self.experiment.description.strip()}")
        
        print("\n" + "="*70 + "\n")


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Config object with dot notation access
    
    Example:
        >>> cfg = load_config('config.yaml')
        >>> print(cfg.agents.num_agents)
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False, indent=2)
    
    print(f"Config saved to: {save_path}")


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge override dict into base config.
    
    Args:
        base_config: Base configuration
        override_dict: Dictionary with overrides
    
    Returns:
        New Config with overrides applied
    
    Example:
        >>> cfg = load_config('config.yaml')
        >>> cfg = merge_configs(cfg, {'agents': {'num_agents': 100}})
    """
    def recursive_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                recursive_merge(base[key], value)
            else:
                base[key] = value
    
    merged = dict(base_config)
    recursive_merge(merged, override_dict)
    
    return Config(merged)


def create_experiment_config(
    base_config_path: str = "config.yaml",
    experiment_name: str = "experiment",
    **overrides
) -> Config:
    """
    Create experiment config with overrides.
    
    Args:
        base_config_path: Path to base config
        experiment_name: Name of experiment
        **overrides: Key-value pairs to override (use '__' for nested, e.g., agents__num_agents=100)
    
    Returns:
        Config with overrides applied
    
    Example:
        >>> cfg = create_experiment_config(
        ...     'config.yaml',
        ...     'ucb_experiment',
        ...     selection__strategy='bandit_ucb',
        ...     selection__k=5
        ... )
    """
    cfg = load_config(base_config_path)
    
    # Update experiment name
    cfg.experiment.name = experiment_name
    
    # Apply overrides (flatten nested keys)
    override_dict = {}
    for key, value in overrides.items():
        parts = key.split('__')
        current = override_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    if override_dict:
        cfg = merge_configs(cfg, override_dict)
    
    return cfg


# ============================================
# QUICK CONFIG TEMPLATES
# ============================================

def get_baseline_config() -> Config:
    """Get baseline configuration."""
    return load_config('config.yaml')


def get_ucb_bandit_config() -> Config:
    """Get config with UCB bandit selection."""
    return create_experiment_config(
        experiment_name='ucb_bandit',
        selection__strategy='bandit_ucb',
        selection__bandit__ucb__exploration_constant=2.0
    )


def get_thompson_bandit_config() -> Config:
    """Get config with Thompson Sampling."""
    return create_experiment_config(
        experiment_name='thompson_sampling',
        selection__strategy='bandit_thompson'
    )


def get_large_prototype_config() -> Config:
    """Get config with larger prototype count."""
    return create_experiment_config(
        experiment_name='large_prototypes',
        agents__num_prototypes=512
    )


def get_ensemble_tree_config() -> Config:
    """Get config with ensemble tree aggregator."""
    return create_experiment_config(
        experiment_name='ensemble_tree',
        aggregator__type='ensemble',
        aggregator__ensemble__n_models=10
    )


if __name__ == "__main__":
    # Test config loading
    print("Testing config loading...\n")
    
    cfg = load_config('config.yaml')
    cfg.print_summary()
    
    print("\nAccessing nested values:")
    print(f"  cfg.agents.num_agents = {cfg.agents.num_agents}")
    print(f"  cfg.selection.strategy = {cfg.selection.strategy}")
    print(f"  cfg.slot_attention.adaptive = {cfg.slot_attention.adaptive}")
    
    print("\nHelper methods:")
    print(f"  cfg.is_bandit_selection() = {cfg.is_bandit_selection()}")
    print(f"  cfg.get_slot_range() = {cfg.get_slot_range()}")
    print(f"  cfg.get_device() = {cfg.get_device()}")
    
    print("\n✓ Config loading test passed!")


