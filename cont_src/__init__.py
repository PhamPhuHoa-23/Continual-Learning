"""
Continual Learning Framework - Compositional Sub-Concept Routing

A modular, config-driven framework for continual learning research.
"""

__version__ = "0.1.0"

from cont_src.core.registry import Registry

# Expose registry for easy access
__all__ = ["Registry", "__version__"]
