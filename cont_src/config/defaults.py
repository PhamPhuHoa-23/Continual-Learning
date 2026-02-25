"""Default configurations."""

from cont_src.config.base import Config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_cifar100_config() -> Config:
    """Get configuration for CIFAR-100 experiments."""
    config = Config()

    # Dataset
    config.data.dataset = "cifar100"
    config.data.n_tasks = 10
    config.data.n_classes_per_task = 10
    config.data.batch_size = 64

    # Backbone
    config.backbone.type = "vit"
    config.backbone.freeze = True

    # Slot Attention
    config.slot_attention.num_slots = 7
    config.slot_attention.slot_dim = 64

    # Agents
    config.agent.type = "mlp"
    config.agent.input_dim = 64
    config.agent.hidden_dim = 256
    config.agent.output_dim = 256

    # Losses
    config.loss.use_primitive = True
    config.loss.use_supcon = True
    config.loss.weight_primitive = 10.0
    config.loss.primitive_temperature = 10.0

    # Clustering
    config.clustering.n_clusters_task1 = 10

    # Training
    config.training.task1_epochs_slot_attention = 100
    config.training.task1_epochs_agents = 50
    config.training.incremental_epochs = 30

    config.experiment_name = "cifar100_default"

    return config


def get_tiny_imagenet_config() -> Config:
    """Get configuration for Tiny-ImageNet experiments."""
    config = get_cifar100_config()

    # Override for Tiny-ImageNet
    config.data.dataset = "tiny_imagenet"
    config.data.n_tasks = 20
    config.data.n_classes_per_task = 10

    # Larger model
    config.agent.hidden_dim = 512
    config.agent.output_dim = 512
    config.aggregator.hidden_dim = 512

    # More clusters
    config.clustering.n_clusters_task1 = 20

    # Longer training
    config.training.task1_epochs_agents = 100
    config.training.incremental_epochs = 50

    config.experiment_name = "tiny_imagenet_default"

    return config


# Configuration presets
CONFIGS = {
    "default": get_default_config,
    "cifar100": get_cifar100_config,
    "tiny_imagenet": get_tiny_imagenet_config,
}


def get_config(name: str = "default") -> Config:
    """
    Get configuration by name.

    Args:
        name: Configuration preset name

    Returns:
        Config instance
    """
    if name not in CONFIGS:
        raise ValueError(
            f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}"
        )

    return CONFIGS[name]()
