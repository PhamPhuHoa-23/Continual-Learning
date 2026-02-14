"""
Slot Attention Module for Object-Centric Learning

This package implements the Slot Attention mechanism for unsupervised object discovery.

Based on:
- "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
- Amazon AdaSlot implementation
"""

from .slot_attention import SlotAttention, PositionalEmbedding
from .encoder import CNNEncoder, ResNetEncoder
from .decoder import BroadcastDecoder, MLPDecoder
from .model import SlotAttentionAutoEncoder, build_slot_attention_model

__all__ = [
    # Core modules
    'SlotAttention',
    'PositionalEmbedding',
    
    # Encoders
    'CNNEncoder',
    'ResNetEncoder',
    
    # Decoders
    'BroadcastDecoder',
    'MLPDecoder',
    
    # Complete model
    'SlotAttentionAutoEncoder',
    'build_slot_attention_model',
]

