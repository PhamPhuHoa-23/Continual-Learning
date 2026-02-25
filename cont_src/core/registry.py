"""
Registry Pattern for Component Management

Allows dynamic registration and retrieval of components (models, losses, datasets, etc.)
through decorators and config-driven instantiation.
"""

from typing import Dict, Type, Any, Callable, Optional
import inspect


class Registry:
    """
    Registry for managing component classes.

    Usage:
        # Define a registry
        MODEL_REGISTRY = Registry("models")

        # Register a class
        @MODEL_REGISTRY.register("mlp_agent")
        class MLPAgent:
            pass

        # Retrieve and instantiate
        agent_cls = MODEL_REGISTRY.get("mlp_agent")
        agent = MODEL_REGISTRY.build("mlp_agent", hidden_dim=256)
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class.

        Args:
            name: Registration name. If None, uses class name in snake_case.

        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            reg_name = name if name is not None else self._to_snake_case(
                cls.__name__)

            if reg_name in self._registry:
                raise ValueError(
                    f"'{reg_name}' already registered in {self.name} registry. "
                    f"Existing: {self._registry[reg_name]}, New: {cls}"
                )

            self._registry[reg_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """
        Get registered class by name.

        Args:
            name: Registration name

        Returns:
            Registered class

        Raises:
            KeyError: If name not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )

        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """
        Build instance from registered class.

        Args:
            name: Registration name
            **kwargs: Arguments to pass to class constructor

        Returns:
            Instance of registered class
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def contains(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', registered={len(self._registry)})"


# Global registries for different component types
REGISTRIES = {
    "model": Registry("model"),
    "backbone": Registry("backbone"),
    "agent": Registry("agent"),
    "router": Registry("router"),
    "aggregator": Registry("aggregator"),
    "classifier": Registry("classifier"),
    "loss": Registry("loss"),
    "dataset": Registry("dataset"),
    "transform": Registry("transform"),
    "trainer": Registry("trainer"),
    "clustering": Registry("clustering"),
}


# Convenient access
MODEL_REGISTRY = REGISTRIES["model"]
BACKBONE_REGISTRY = REGISTRIES["backbone"]
AGENT_REGISTRY = REGISTRIES["agent"]
ROUTER_REGISTRY = REGISTRIES["router"]
AGGREGATOR_REGISTRY = REGISTRIES["aggregator"]
CLASSIFIER_REGISTRY = REGISTRIES["classifier"]
LOSS_REGISTRY = REGISTRIES["loss"]
DATASET_REGISTRY = REGISTRIES["dataset"]
TRANSFORM_REGISTRY = REGISTRIES["transform"]
TRAINER_REGISTRY = REGISTRIES["trainer"]
CLUSTERING_REGISTRY = REGISTRIES["clustering"]


def register_module(registry_name: str, name: Optional[str] = None) -> Callable:
    """
    Convenience decorator for registering to a specific registry.

    Args:
        registry_name: Name of registry (e.g., "model", "loss")
        name: Optional custom registration name

    Returns:
        Decorator function
    """
    if registry_name not in REGISTRIES:
        raise ValueError(
            f"Unknown registry '{registry_name}'. "
            f"Available: {list(REGISTRIES.keys())}"
        )

    return REGISTRIES[registry_name].register(name)
