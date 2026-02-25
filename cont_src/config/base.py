"""
Configuration System

Provides structured configuration with validation and defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
import yaml
import json
from pathlib import Path


@dataclass
class BackboneConfig:
    """Configuration for backbone network."""
    type: str = "vit"  # vit, resnet50, etc.
    pretrained: bool = True
    freeze: bool = True
    pretrained_source: str = "imagenet21k"
    output_dim: int = 768  # Output feature dimension
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotAttentionConfig:
    """Configuration for Slot Attention module."""
    num_slots: int = 7
    slot_dim: int = 64
    num_iterations: int = 3
    hidden_dim: int = 128
    freeze_after_task1: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for agent networks."""
    type: str = "mlp"  # mlp, transformer, etc.
    input_dim: int = 64  # slot_dim
    hidden_dim: int = 256
    output_dim: int = 256  # d_h in paper
    num_layers: int = 3
    activation: str = "relu"
    dropout: float = 0.0
    use_layer_norm: bool = True
    freeze_after_training: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    """Configuration for routing mechanism."""
    type: str = "vae"  # vae, cosine, etc.
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    freeze_network: bool = True  # Freeze VAE network, update stats only
    update_stats: bool = True
    threshold_match: float = -10.0  # Mahalanobis distance threshold
    threshold_novel: float = -50.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatorConfig:
    """Configuration for aggregator."""
    type: str = "attention"  # attention, average, etc.
    hidden_dim: int = 256  # d_h
    use_per_agent_keys: bool = True  # Block-diagonal attention
    freeze_keys_with_agents: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    """Configuration for classifier."""
    type: str = "slda"  # slda, linear, etc.
    shrinkage: float = 1e-4  # For SLDA
    incremental: bool = True  # Update incrementally
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Loss types to use
    use_reconstruction: bool = True
    use_primitive: bool = True
    use_supcon: bool = True
    use_agent_recon: bool = False
    use_local_geometry: bool = False

    # Loss weights
    weight_reconstruction: float = 1.0
    weight_primitive: float = 10.0
    weight_supcon: float = 1.0
    weight_agent_recon: float = 1.0
    weight_local_geometry: float = 0.5

    # Loss-specific params
    primitive_temperature: float = 10.0
    supcon_temperature: float = 0.07
    local_k_neighbors: int = 5

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringConfig:
    """Configuration for agent spawning via clustering."""
    algorithm: str = "kmeans"  # kmeans, hdbscan, spectral
    buffer_size_min: int = 100
    cluster_size_min: int = 20
    similarity_threshold: float = 0.7
    n_clusters_task1: int = 10  # Initial agents in Task 1
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for dataset."""
    dataset: str = "cifar100"  # cifar100, tiny_imagenet, etc.
    data_root: str = "./data"
    n_tasks: int = 10
    n_classes_per_task: int = 10
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    # For continual learning
    shuffle_classes: bool = True
    seed: int = 42

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: str = "medium"  # weak, medium, strong

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Task 1 (warmup)
    task1_epochs_slot_attention: int = 100
    task1_epochs_agents: int = 50
    task1_lr: float = 3e-4

    # Incremental tasks
    incremental_epochs: int = 30
    incremental_lr: float = 1e-4

    # Optimization
    optimizer: str = "adam"  # adam, adamw, sgd
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5

    # Checkpointing
    save_interval: int = 10  # Save every N epochs
    keep_best_only: bool = False

    # Device
    device: str = "cuda"  # cuda, cpu
    mixed_precision: bool = False

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container."""
    # Component configs
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    slot_attention: SlotAttentionConfig = field(
        default_factory=SlotAttentionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "default"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # Extra params
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        # Parse nested configs
        backbone = BackboneConfig(**config_dict.get("backbone", {}))
        slot_attention = SlotAttentionConfig(
            **config_dict.get("slot_attention", {}))
        agent = AgentConfig(**config_dict.get("agent", {}))
        router = RouterConfig(**config_dict.get("router", {}))
        aggregator = AggregatorConfig(**config_dict.get("aggregator", {}))
        classifier = ClassifierConfig(**config_dict.get("classifier", {}))
        loss = LossConfig(**config_dict.get("loss", {}))
        clustering = ClusteringConfig(**config_dict.get("clustering", {}))
        data = DataConfig(**config_dict.get("data", {}))
        training = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            backbone=backbone,
            slot_attention=slot_attention,
            agent=agent,
            router=router,
            aggregator=aggregator,
            classifier=classifier,
            loss=loss,
            clustering=clustering,
            data=data,
            training=training,
            experiment_name=config_dict.get("experiment_name", "default"),
            output_dir=config_dict.get("output_dir", "outputs"),
            log_dir=config_dict.get("log_dir", "logs"),
            extras=config_dict.get("extras", {}),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """
        Load config from JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def save_json(self, path: Union[str, Path]):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        return f"Config(experiment='{self.experiment_name}')"
