"""
Feature Extractor for SlotAttention / AdaSlot.

CNN variant mirrors: ocl.feature_extractors.SlotAttentionFeatureExtractor
  Checkpoint key prefix: models.feature_extractor.layers.*

ViT variant mirrors: timm ViT-B/16 used in COCO / MOVi-C / MOVi-E checkpoints
  Checkpoint key prefix: models.feature_extractor.model.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def cnn_compute_positions_and_flatten(features: torch.Tensor):
    """Flatten CNN output and compute normalized positions."""
    spatial_dims = features.shape[2:]  # (H, W)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # (B, C, H, W) -> (B, H*W, C)
    flattened = torch.permute(features.view(
        features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return positions, flattened


class SlotAttentionFeatureExtractor(nn.Module):
    """
    CNN feature extractor as used in the original Slot Attention paper.

    4 layers of Conv2d(kernel=5, padding=same, stride=1), 64 channels, ReLU.
    Output: (B, H*W, 64) features + (H*W, 2) positions.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

    @property
    def feature_dim(self):
        return 64

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            positions: (H*W, 2) normalized positions in [0, 1]
            features:  (B, H*W, 64) flattened feature maps
        """
        feat = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(feat)
        return positions, flattened


class SlotAttentionViTFeatureExtractor(nn.Module):
    """
    ViT-B/16 feature extractor matching COCO / MOVi-C / MOVi-E checkpoints.

    Wraps timm's vit_base_patch16_224 under `self.model` so that state_dict
    keys match the checkpoint prefix `models.feature_extractor.model.*`.

    Differences from plain timm ViT:
    - Final norm layer and classification head are NOT present in the checkpoint
      (weights are saved only up to transformer blocks).
    - Positional embeddings are stored for 224×224 (196 patches); at inference
      they are bilinearly interpolated to match the actual input resolution.
    - Returns only patch tokens (CLS token is discarded), shaped (B, N, 768).

    Args:
        pretrained_imagenet: If True, initialise from timm's ImageNet-21k
            pretrained weights BEFORE loading the AdaSlot checkpoint.
            This is a safe default — the AdaSlot checkpoint will overwrite
            these weights via load_state_dict anyway.
    """

    PATCH_SIZE = 16
    HIDDEN_DIM = 768
    TRAIN_GRID = 14   # √196 — number of patches per side at 224×224

    def __init__(self, pretrained_imagenet: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ViT feature extractor. "
                "Install with: pip install timm"
            )

        # Load ViT-B/16 — keys stored as self.model.* to match checkpoint prefix
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained_imagenet,
            num_classes=0,          # drop classification head
            global_pool="",         # keep all tokens (no pooling)
        )

        # Remove the final norm layer that timm includes but checkpoint doesn't
        # (the ckpt ends at blocks.11.mlp.fc2, no norm key present)
        if hasattr(self.model, "norm"):
            del self.model.norm

    @property
    def feature_dim(self) -> int:
        return self.HIDDEN_DIM

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)  — any resolution; 224×224 is native.

        Returns:
            positions: (N, 2)      — normalized [0,1] coords for each patch.
            features:  (B, N, 768) — patch token features (CLS discarded).
        """
        B, C, H, W = images.shape
        h_patches = H // self.PATCH_SIZE
        w_patches = W // self.PATCH_SIZE

        features = self._forward_patches(images, h_patches, w_patches)

        # Build normalized grid positions, same convention as CNN extractor
        positions = torch.cartesian_prod(
            torch.linspace(0.0, 1.0, h_patches, device=images.device),
            torch.linspace(0.0, 1.0, w_patches, device=images.device),
        )  # (N, 2)

        return positions, features

    # ------------------------------------------------------------------
    def _forward_patches(
        self,
        images: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Internal: run ViT and return patch tokens only."""
        m = self.model
        B = images.shape[0]

        # 1. Patch embedding
        x = m.patch_embed(images)           # (B, h*w, 768)

        # 2. Prepend CLS token
        cls = m.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, 1+N, 768)

        # 3. Add positional embedding (interpolate if resolution differs)
        pos = self._interp_pos_embed(m.pos_embed, h_patches, w_patches)
        x = x + pos                         # broadcast over batch

        # 4. Dropout (usually identity at eval)
        x = m.pos_drop(x)

        # 5. Transformer blocks
        for blk in m.blocks:
            x = blk(x)

        # 6. Return patch tokens, drop CLS
        return x[:, 1:, :]                  # (B, N, 768)

    def _interp_pos_embed(
        self,
        pos_embed: torch.Tensor,    # (1, 1+196, 768)
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Interpolate positional embeddings to a different spatial resolution."""
        cls_pos = pos_embed[:, :1, :]         # (1, 1, 768)
        patch_pos = pos_embed[:, 1:, :]         # (1, 196, 768)

        G = self.TRAIN_GRID                     # 14

        if h_patches == G and w_patches == G:
            return pos_embed                    # no-op for native resolution

        # Bicubic interpolation in spatial dimensions
        patch_pos = (
            patch_pos
            .reshape(1, G, G, -1)               # (1, 14, 14, D)
            .permute(0, 3, 1, 2)                # (1, D, 14, 14)
        )
        patch_pos = F.interpolate(
            patch_pos,
            size=(h_patches, w_patches),
            mode="bicubic",
            align_corners=False,
        )                                       # (1, D, h, w)
        patch_pos = (
            patch_pos
            .permute(0, 2, 3, 1)                # (1, h, w, D)
            .reshape(1, h_patches * w_patches, -1)
        )
        return torch.cat([cls_pos, patch_pos], dim=1)
