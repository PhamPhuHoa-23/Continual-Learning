"""
Decoder for AdaSlot.

Mirrors: ocl.decoding.SlotAttentionDecoder with get_slotattention_decoder_backbone
Checkpoint key prefix: models.object_decoder.*
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple, Union


def build_grid_of_positions(resolution: Tuple[int, int]) -> torch.Tensor:
    """Build grid of positions for positional embeddings."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid


def get_slotattention_decoder_backbone(
    object_dim: int, output_dim: int = 4
) -> nn.Sequential:
    """
    CNN decoder backbone from the original Slot Attention paper.

    Architecture: 4x ConvTranspose2d(stride=2) + 1x ConvTranspose2d(stride=1) + final Conv.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(object_dim, 64, 5, stride=2,
                           padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim, 3, stride=1,
                           padding=1, output_padding=0),
    )


def get_activation_fn(name: Union[str, Callable]):
    """Get activation function by name."""
    if callable(name):
        return name
    if name == "identity" or name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class SlotAttentionDecoder(nn.Module):
    """
    Decoder used in the original Slot Attention paper (with Gumbel mask support).

    Broadcasts each slot to an 8x8 spatial grid, optionally adds positional
    embedding, then applies a deconvolution network. Splits output into
    RGB (3ch) and alpha (1ch), normalizes alpha via softmax across slots.
    Optionally applies a hard_keep_decision mask (Gumbel selection).
    """

    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = "identity",
        positional_embedding: Optional[nn.Module] = None,
        mask_type: str = "mask_normalized",
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        self.mask_type = mask_type
        if positional_embedding:
            self.register_buffer(
                "grid", build_grid_of_positions(self.initial_conv_size)
            )

    def forward(
        self,
        object_features: torch.Tensor,
        left_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode slot features into image reconstructions.

        Args:
            object_features: (B, num_slots, slot_dim) slot representations.
            left_mask: (B, num_slots) binary Gumbel keep decisions (optional).

        Returns:
            Dict with 'reconstruction', 'object_reconstructions', 'masks'.
        """
        assert object_features.dim() >= 3
        initial_shape = object_features.shape[:-1]  # (B, num_slots)
        object_features = object_features.flatten(
            0, -2)  # (B*num_slots, slot_dim)

        # Broadcast to spatial grid
        object_features = (
            object_features.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, *self.initial_conv_size)
        )  # (B*num_slots, slot_dim, 8, 8)

        if self.positional_embedding:
            object_features = self.positional_embedding(
                object_features, self.grid.unsqueeze(0)
            )

        # Apply deconvolution
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)  # (B, num_slots, C, H, W)

        # Split RGB and alpha
        # (B, K, 3, H, W), (B, K, 1, H, W)
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)

        # Apply Gumbel keep mask
        if left_mask is not None and self.mask_type != "none":
            if self.mask_type == "logit":
                VANISH = 1e5
                drop_mask = 1 - left_mask
                alpha = alpha - VANISH * \
                    drop_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                alpha = alpha.softmax(dim=-4)
            elif self.mask_type == "mask":
                alpha = alpha.softmax(dim=-4)
                alpha = alpha * \
                    left_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            elif self.mask_type == "mask_normalized":
                MINOR = 1e-5
                alpha = alpha.softmax(dim=-4)
                alpha = alpha * \
                    left_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                alpha = alpha / (alpha.sum(dim=-4, keepdim=True) + MINOR)
        else:
            alpha = alpha.softmax(dim=-4)

        return {
            "reconstruction": (rgb * alpha).sum(-4),
            "object_reconstructions": rgb,
            "masks": alpha.squeeze(-3),
        }


class MLPPatchDecoder(nn.Module):
    """
    MLP-based broadcast decoder used in the COCO / MOVi-C / MOVi-E checkpoints.

    For each slot, learnable positional embeddings are added (broadcast decoder
    style), then a shared MLP is applied to produce per-patch RGB + alpha.

    Checkpoint key mapping:
        object_decoder.pos_embed          [1, N, slot_dim]
        object_decoder.decoder.0.*        Linear(slot_dim, hidden_dim)
        object_decoder.decoder.1          ReLU   (no params)
        object_decoder.decoder.2.*        Linear(hidden_dim, hidden_dim)
        object_decoder.decoder.3          ReLU
        object_decoder.decoder.4.*        Linear(hidden_dim, hidden_dim)
        object_decoder.decoder.5          ReLU
        object_decoder.decoder.6.*        Linear(hidden_dim, patch_size² * 3 + 1)

    Args:
        slot_dim:    Dimension of input slot features (default: 256).
        hidden_dim:  MLP hidden dimension (default: 2048).
        n_patches:   Total number of patches, e.g. 196 for 224×224 / 16×16.
        patch_size:  Spatial size of one patch in pixels (default: 16).
        final_activation: Activation applied to the RGB channels after mixing.
        mask_type:   How the Gumbel hard-keep decision is applied.
            ``"mask_normalized"`` (default): softmax + normalize by kept slots.
            ``"logit"``:  subtract large constant from dropped slots before softmax.
            ``"mask"``:   softmax then zero out dropped slots (un-normalised).
            ``"none"``:   ignore hard_keep_decision entirely.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        hidden_dim: int = 2048,
        n_patches: int = 196,
        patch_size: int = 16,
        final_activation: Union[str, Callable] = "identity",
        mask_type: str = "mask_normalized",
    ):
        super().__init__()

        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3   # e.g. 768 for 16×16
        self.output_dim = self.patch_dim + 1           # RGB + alpha (e.g. 769)

        self.final_activation = get_activation_fn(final_activation)
        self.mask_type = mask_type

        # Learnable positional embeddings: one vector per patch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, slot_dim)
        )

        # MLP decoder — indices 0, 2, 4, 6 are Linear (match checkpoint keys)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),    # 0
            nn.ReLU(inplace=True),               # 1
            nn.Linear(hidden_dim, hidden_dim),   # 2
            nn.ReLU(inplace=True),               # 3
            nn.Linear(hidden_dim, hidden_dim),   # 4
            nn.ReLU(inplace=True),               # 5
            nn.Linear(hidden_dim, self.output_dim),  # 6
        )

    # --- helpers ----------------------------------------------------------

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange flat patch tokens into a spatial image.

        Args:
            x: (B, N, patch_dim)  where patch_dim = C * p * p

        Returns:
            (B, C, H, W)
        """
        B, N, _ = x.shape
        H_p = W_p = int(N ** 0.5)
        C = 3
        p = self.patch_size
        # (B, H_p, W_p, C, p, p)
        x = x.view(B, H_p, W_p, C, p, p)
        # (B, C, H_p, p, W_p, p) → (B, C, H, W)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, H_p * p, W_p * p)

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        object_features: torch.Tensor,
        left_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode slot features into image reconstructions.

        Args:
            object_features: (B, K, slot_dim)
            left_mask:       (B, K) binary Gumbel keep decisions (optional)

        Returns:
            Dict with ``'reconstruction'``, ``'object_reconstructions'``,
            ``'masks'``.
        """
        B, K, _ = object_features.shape

        # Broadcast: add positional embedding to each slot
        # (B, K, 1, D) + (1, 1, N, D) → (B, K, N, D)
        x = object_features.unsqueeze(2) + self.pos_embed.unsqueeze(0)

        # Apply MLP: (B, K, N, D) → (B, K, N, output_dim)
        x = self.decoder(x)

        # Split RGB patches and alpha logits
        # rgb_patches: (B, K, N, patch_dim),  alpha: (B, K, N, 1)
        rgb_patches, alpha = x.split([self.patch_dim, 1], dim=-1)

        # ---- apply Gumbel keep mask ----
        if left_mask is not None and self.mask_type != "none":
            # left_mask: (B, K) → (B, K, 1, 1) for broadcasting
            lm = left_mask.unsqueeze(-1).unsqueeze(-1)   # (B, K, 1, 1)
            if self.mask_type == "logit":
                VANISH = 1e5
                alpha = alpha - VANISH * (1.0 - lm)
                alpha = alpha.softmax(dim=1)
            elif self.mask_type == "mask":
                alpha = alpha.softmax(dim=1) * lm
            else:   # "mask_normalized"  (default)
                MINOR = 1e-5
                alpha = alpha.softmax(dim=1) * lm
                alpha = alpha / (alpha.sum(dim=1, keepdim=True) + MINOR)
        else:
            alpha = alpha.softmax(dim=1)   # sum-to-1 over slots

        # ---- mix: (B, N, patch_dim) ----
        mixed = (rgb_patches * alpha).sum(dim=1)

        # ---- apply final activation ----
        # (operate on the C dimension after unpatchifying for consistency)
        reconstruction = self._unpatchify(mixed)
        reconstruction = self.final_activation(reconstruction)

        # ---- per-slot reconstructions: (B, K, 3, H, W) ----
        H_p = W_p = int(self.n_patches ** 0.5)
        C, p = 3, self.patch_size
        object_recs = (
            rgb_patches
            .view(B, K, H_p, W_p, C, p, p)
            .permute(0, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .view(B, K, C, H_p * p, W_p * p)
        )

        # ---- alpha masks: (B, K, H, W) ----
        # alpha is (B, K, N, 1) — one scalar per patch; tile p×p to full resolution
        masks = (
            alpha
            .squeeze(-1)                            # (B, K, N)
            .view(B, K, H_p, W_p)                  # (B, K, H_p, W_p)
            .repeat_interleave(p, dim=-2)           # (B, K, H, W_p)
            .repeat_interleave(p, dim=-1)           # (B, K, H, W)
        )

        return {
            "reconstruction": reconstruction,
            "object_reconstructions": object_recs,
            "masks": masks,
        }
