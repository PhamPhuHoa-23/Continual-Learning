"""
Slot Attention Models for Continual Learning

Provides slot-based object-centric representation learning modules:
- AdaptiveSlotAttention: Adaptive slot attention with Gumbel selection
- PrimitiveSelector: Primitive selection mechanism
- SlotDecoder: Slot decoder for reconstruction
- AdaSlotModule: Complete AdaSlot module
"""

from cont_src.models.slot_attention.adaptive_slot_attention import (
    AdaptiveSlotAttention,
    GumbelSlotSelector,
    sample_slot_lower_bound,
)

from cont_src.models.slot_attention.primitives import (
    PrimitiveSelector,
    SlotDecoder,
    AdaSlotModule,
)

__all__ = [
    "AdaptiveSlotAttention",
    "GumbelSlotSelector",
    "sample_slot_lower_bound",
    "PrimitiveSelector",
    "SlotDecoder",
    "AdaSlotModule",
]
