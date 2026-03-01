"""Model components."""

# Import all model registries to ensure components are registered
from cont_src.models import agents
from cont_src.models import routers
from cont_src.models import aggregators
from cont_src.models import slot_attention

__all__ = ["agents", "routers", "aggregators", "slot_attention"]
