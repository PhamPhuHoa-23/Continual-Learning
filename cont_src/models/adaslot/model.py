"""
AdaSlot Combined Model.

Wraps feature_extractor, conditioning, perceptual_grouping, and object_decoder
under a `models` sub-module so that state_dict keys match the checkpoint format:
    models.feature_extractor.*
    models.conditioning.*
    models.perceptual_grouping.*
    models.object_decoder.*
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .feature_extractor import SlotAttentionFeatureExtractor, SlotAttentionViTFeatureExtractor
from .conditioning import RandomConditioning
from .perceptual_grouping import SlotAttentionGroupingGumbelV1
from .decoder import SlotAttentionDecoder, get_slotattention_decoder_backbone, MLPPatchDecoder
from .positional_embedding import SoftPositionEmbed


class _NoOpPositionalEmbed(nn.Module):
    """
    No-op positional embedding placeholder for ViT checkpoints.

    ViT checkpoints do not include SoftPositionEmbed weights at
    ``positional_embedding.layers.0``, so this class is stored at layers[0]
    but has no parameters and simply returns inputs unchanged.
    """

    def forward(self, inputs: torch.Tensor, positions=None) -> torch.Tensor:
        return inputs


class _PositionalEmbeddingWithMLP(nn.Module):
    """
    Positional embedding followed by an MLP, wrapped in a ModuleList called `layers`.

    This matches the checkpoint structure:
        positional_embedding.layers.0.dense.{weight,bias}   <- SoftPositionEmbed
        positional_embedding.layers.1.0.{weight,bias}       <- LayerNorm
        positional_embedding.layers.1.1.{weight,bias}       <- Linear (up)
        positional_embedding.layers.1.3.{weight,bias}       <- Linear (down)

    The MLP is: nn.Sequential(LayerNorm, Linear, ReLU, Linear)
    """

    def __init__(
        self,
        n_spatial_dims: int,
        feature_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        use_soft_pos_embed: bool = True,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = feature_dim

        if use_soft_pos_embed:
            pos_layer: nn.Module = SoftPositionEmbed(
                n_spatial_dims=n_spatial_dims,
                feature_dim=feature_dim,
                cnn_channel_order=False,
                savi_style=False,
            )
        else:
            pos_layer = _NoOpPositionalEmbed()

        self.layers = nn.ModuleList([
            pos_layer,
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            ),
        ])

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](inputs, positions)
        x = self.layers[1](x)
        return x


class _FFResidualMLP(nn.Module):
    """
    Feed-forward residual MLP for slot attention, stored as `module` (nn.Sequential).

    Matches checkpoint structure:
        ff_mlp.module.0.{weight,bias}  <- LayerNorm
        ff_mlp.module.1.{weight,bias}  <- Linear (up)
        ff_mlp.module.3.{weight,bias}  <- Linear (down)

    Architecture: LayerNorm -> Linear(dim, hidden) -> ReLU -> Linear(hidden, dim)
    Applied residually: output = mlp(x) + x
    """

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs) + inputs


class _Models(nn.Module):
    """
    Container that holds sub-modules under the 'models' prefix,
    matching the checkpoint key structure.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        object_decoder: nn.Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder


class AdaSlotModel(nn.Module):
    """
    Complete AdaSlot model matching the original checkpoint structure.

    The state_dict keys will have the prefix `models.XXX.YYY` which matches
    the checkpoint files from the AdaSlot repository.

    Args:
        resolution: Input image resolution (H, W).
        num_slots: Maximum number of slots.
        slot_dim: Dimension of slot embeddings.
        num_iterations: Number of slot attention iterations.
        feature_dim: Output feature dimension of the CNN encoder (default: 64).
            Ignored when ``encoder_type='vit'``; ViT always outputs 768-dim patches.
        kvq_dim: Key/Query/Value projection dimension (default: 128).
        low_bound: Minimum number of slots to keep in Gumbel selection.
        encoder_type: ``'cnn'`` (default) or ``'vit'``.
        vit_pretrained: If True and ``encoder_type='vit'``, initialise from
            timm's ImageNet-21k pretrained ViT-B/16 weights before the AdaSlot
            checkpoint is loaded.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (128, 128),
        num_slots: int = 11,
        slot_dim: int = 64,
        num_iterations: int = 3,
        feature_dim: int = 64,
        kvq_dim: int = 128,
        low_bound: int = 1,
        encoder_type: str = "cnn",
        vit_pretrained: bool = True,
    ):
        super().__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.encoder_type = encoder_type

        if encoder_type == "vit":
            # ── ViT path (COCO / MOVi-C / MOVi-E checkpoints) ────────────────
            # feature_dim is fixed at 768 for ViT-B/16
            # hidden dims scale with slot_dim: ff = 4×, gumbel = 4×, decoder = 8×
            vit_feature_dim = 768
            ff_hidden = 4 * slot_dim
            gumbel_hidden = 4 * slot_dim
            decoder_hidden = 8 * slot_dim

            # 1. Feature Extractor
            feature_extractor = SlotAttentionViTFeatureExtractor(
                pretrained_imagenet=vit_pretrained
            )

            # 2. Conditioning
            conditioning = RandomConditioning(
                object_dim=slot_dim,
                n_slots=num_slots,
            )

            # 3. Perceptual Grouping
            # Pos-embed: no SoftPositionEmbed (ViT has no layers.0 params in ckpt)
            # layers.1: LayerNorm(768) → Linear(768,768) → ReLU → Linear(768,slot_dim)
            pos_embed_grouping = _PositionalEmbeddingWithMLP(
                n_spatial_dims=2,
                feature_dim=vit_feature_dim,
                hidden_dim=vit_feature_dim,   # hidden=768
                output_dim=slot_dim,           # project to slot_dim
                use_soft_pos_embed=False,      # ← no SoftPositionEmbed for ViT
            )

            ff_mlp = _FFResidualMLP(dim=slot_dim, hidden_dim=ff_hidden)

            single_gumbel_score_network = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, gumbel_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gumbel_hidden, 2),
            )

            # After pos_embed_grouping, features are slot_dim-dimensional
            perceptual_grouping = SlotAttentionGroupingGumbelV1(
                feature_dim=slot_dim,       # output of pos_embed projection
                object_dim=slot_dim,
                kvq_dim=kvq_dim,
                n_heads=1,
                iters=num_iterations,
                eps=1e-8,
                ff_mlp=ff_mlp,
                positional_embedding=pos_embed_grouping,
                use_projection_bias=False,
                use_implicit_differentiation=False,
                single_gumbel_score_network=single_gumbel_score_network,
                low_bound=low_bound,
            )

            # 4. Object Decoder: MLP patch decoder
            # n_patches = (resolution / patch_size)² = (224/16)² = 196
            n_patches = (resolution[0] // 16) * (resolution[1] // 16)
            object_decoder = MLPPatchDecoder(
                slot_dim=slot_dim,
                hidden_dim=decoder_hidden,
                n_patches=n_patches,
                patch_size=16,
                final_activation="identity",
            )

        else:
            # ── CNN path (CLEVR10 checkpoint) ─────────────────────────────────
            ff_hidden = 128
            gumbel_hidden = 256

            # 1. Feature Extractor
            feature_extractor = SlotAttentionFeatureExtractor()

            # 2. Conditioning
            conditioning = RandomConditioning(
                object_dim=slot_dim,
                n_slots=num_slots,
            )

            # 3. Perceptual Grouping
            # Positional embedding with MLP (matching layers.0 / layers.1 structure)
            pos_embed_grouping = _PositionalEmbeddingWithMLP(
                n_spatial_dims=2,
                feature_dim=feature_dim,
                hidden_dim=128,
                output_dim=feature_dim,   # CNN: same dim in and out
            )

            # ff_mlp: residual MLP with LayerNorm -> Linear -> ReLU -> Linear
            ff_mlp = _FFResidualMLP(dim=slot_dim, hidden_dim=ff_hidden)

            # Gumbel score network: LayerNorm(64) -> Linear(64,256) -> ReLU -> Linear(256,2)
            single_gumbel_score_network = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, gumbel_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gumbel_hidden, 2),
            )

            perceptual_grouping = SlotAttentionGroupingGumbelV1(
                feature_dim=feature_dim,
                object_dim=slot_dim,
                kvq_dim=kvq_dim,
                n_heads=1,
                iters=num_iterations,
                eps=1e-8,
                ff_mlp=ff_mlp,
                positional_embedding=pos_embed_grouping,
                use_projection_bias=False,
                use_implicit_differentiation=False,
                single_gumbel_score_network=single_gumbel_score_network,
                low_bound=low_bound,
            )

            # 4. Object Decoder
            decoder_backbone = get_slotattention_decoder_backbone(
                object_dim=slot_dim, output_dim=4
            )

            pos_embed_decoder = SoftPositionEmbed(
                n_spatial_dims=2,
                feature_dim=slot_dim,
                cnn_channel_order=True,
                savi_style=False,
            )

            object_decoder = SlotAttentionDecoder(
                decoder=decoder_backbone,
                final_activation="identity",
                positional_embedding=pos_embed_decoder,
            )

        # Wrap under `models` prefix
        self.models = _Models(
            feature_extractor=feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            object_decoder=object_decoder,
        )

    def forward(
        self,
        image: torch.Tensor,
        global_step: Optional[int] = None,
        dynamic_slots: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full AdaSlot pipeline.

        Args:
            image: (B, C, H, W) input image.
            global_step: Optional global step for Gumbel temperature scheduling.
            dynamic_slots: If False, all slots are kept (Gumbel gate disabled).
                Use this for stable fine-tuning experiments. Can be toggled
                back to True at any time.

        Returns:
            Dict with 'reconstruction', 'object_reconstructions', 'masks',
            'slots', 'attn', 'slots_keep_prob', 'hard_keep_decision'.
        """
        B = image.size(0)

        # 1. Extract features
        positions, features = self.models.feature_extractor(image)

        # 2. Generate initial slot conditioning
        conditioning = self.models.conditioning(B)

        # 3. Perceptual grouping (Slot Attention + Gumbel)
        grouping_out = self.models.perceptual_grouping(
            features=features,
            positions=positions,
            conditioning=conditioning,
            global_step=global_step,
            dynamic_slots=dynamic_slots,
        )

        slots = grouping_out["objects"]
        attn = grouping_out["feature_attributions"]
        slots_keep_prob = grouping_out["slots_keep_prob"]
        hard_keep_decision = grouping_out["hard_keep_decision"]

        # 4. Decode (pass hard_keep_decision for Gumbel masking)
        decoder_out = self.models.object_decoder(
            slots, left_mask=hard_keep_decision)

        return {
            "reconstruction": decoder_out["reconstruction"],
            "object_reconstructions": decoder_out["object_reconstructions"],
            "masks": decoder_out["masks"],
            "slots": slots,
            "attn": attn,
            "slots_keep_prob": slots_keep_prob,
            "hard_keep_decision": hard_keep_decision,
        }

    def encode(self, image: torch.Tensor, global_step: Optional[int] = None,
               dynamic_slots: bool = True) -> torch.Tensor:
        """
        Encode image to slot representations only.

        Args:
            image: (B, C, H, W) input image.
            global_step: Optional global step.

        Returns:
            slots: (B, num_slots, slot_dim)
        """
        B = image.size(0)
        positions, features = self.models.feature_extractor(image)
        conditioning = self.models.conditioning(B)
        grouping_out = self.models.perceptual_grouping(
            features=features,
            positions=positions,
            conditioning=conditioning,
            global_step=global_step,
            dynamic_slots=dynamic_slots,
        )
        return grouping_out["objects"]
