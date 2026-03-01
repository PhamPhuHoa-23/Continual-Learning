"""
cont_src.models.classifiers
============================
Classification heads for the continual learning framework.

Available classifiers
---------------------
SLDAClassifier
    Streaming Linear Discriminant Analysis — incremental, no-gradient,
    ideal for continual learning (zero forgetting on old classes).

LinearClassifier
    Standard trainable linear head — use when gradient-based fine-tuning
    is acceptable.

Registry
--------
Both classifiers are registered under CLASSIFIER_REGISTRY so they can be
instantiated from a config string:

    from cont_src.core.registry import CLASSIFIER_REGISTRY
    clf = CLASSIFIER_REGISTRY.build("slda", feature_dim=64, n_classes=100)
"""

from cont_src.models.classifiers.slda   import SLDAClassifier
from cont_src.models.classifiers.linear import LinearClassifier

# Register in global CLASSIFIER_REGISTRY
try:
    from cont_src.core.registry import CLASSIFIER_REGISTRY

    if "slda"   not in CLASSIFIER_REGISTRY.list():
        CLASSIFIER_REGISTRY.register("slda")(SLDAClassifier)
    if "linear" not in CLASSIFIER_REGISTRY.list():
        CLASSIFIER_REGISTRY.register("linear")(LinearClassifier)
except Exception:
    pass  # registry may not be initialised at import time in some test environments

__all__ = ["SLDAClassifier", "LinearClassifier"]
