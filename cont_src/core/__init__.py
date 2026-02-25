"""Core framework components."""

from cont_src.core.registry import Registry, REGISTRIES
from cont_src.core.base_module import BaseModule
# from cont_src.core.trainer import BaseTrainer  # TODO: Implement trainer

__all__ = ["Registry", "REGISTRIES", "BaseModule"]
