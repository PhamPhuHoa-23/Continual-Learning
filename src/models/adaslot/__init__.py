"""
AdaSlot: Adaptive Slot Attention with Gumbel Selection.

This module re-implements the AdaSlot architecture from the original repository
to ensure checkpoint compatibility with pretrained .ckpt files.

Architecture mirrors the original `ocl` codebase structure:
  - models.feature_extractor  -> SlotAttentionFeatureExtractor
  - models.conditioning       -> RandomConditioning
  - models.perceptual_grouping -> SlotAttentionGroupingGumbelV1
  - models.object_decoder     -> SlotAttentionDecoder
"""

from .model import AdaSlotModel

__all__ = ["AdaSlotModel"]
