"""
AdaSlot Checkpoint Registry
============================
Centralized config for different pretrained AdaSlot checkpoints.

Each checkpoint comes from the original AdaSlot repository and was trained
on a specific dataset with specific hyperparameters.

Usage:
    from cont_src.models.adaslot_configs import get_adaslot_config, build_adaslot_from_checkpoint
    
    cfg = get_adaslot_config("clevr10")
    model = build_adaslot_from_checkpoint("clevr10", ckpt_path="CLEVR10.ckpt")
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from pathlib import Path


@dataclass
class AdaSlotConfig:
    """Configuration for an AdaSlot checkpoint."""

    name: str
    # Model architecture
    resolution: Tuple[int, int]
    num_slots: int
    slot_dim: int
    num_iterations: int
    feature_dim: int
    kvq_dim: int
    low_bound: int
    encoder_type: str  # "cnn" or "vit"

    # Training context (informational)
    source_dataset: str
    description: str

    # Default checkpoint path (relative to project root)
    default_ckpt_path: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# Registry: maps checkpoint name → config
# ═════════════════════════════════════════════════════════════════════════════

ADASLOT_CONFIGS: Dict[str, AdaSlotConfig] = {
    # ─── CLEVR10 (CNN encoder - FULLY SUPPORTED) ───────────────────────────
    "clevr10": AdaSlotConfig(
        name="clevr10",
        resolution=(128, 128),
        num_slots=11,           # CLEVR: up to 10 objects + 1 background
        slot_dim=64,
        num_iterations=3,
        feature_dim=64,
        kvq_dim=128,
        low_bound=1,
        encoder_type="cnn",     # ✓ Supported by current code
        source_dataset="CLEVR",
        description="Pretrained on CLEVR dataset (synthetic 3D scenes, 10 objects max). CNN encoder.",
        default_ckpt_path="checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt",
    ),

    # ─── COCO (ViT encoder - SUPPORTED) ───────────────────────────────────
    "coco": AdaSlotConfig(
        name="coco",
        resolution=(224, 224),      # ViT trained at 224×224
        num_slots=7,                # COCO: typically fewer salient objects per image
        slot_dim=256,               # ViT checkpoint: slot_dim=256
        num_iterations=3,
        feature_dim=768,            # ViT-B/16 output dim
        kvq_dim=256,                # ViT checkpoint: kvq_dim=256
        low_bound=1,
        encoder_type="vit",
        source_dataset="COCO",
        description="Pretrained on MS COCO (real-world photos). ViT-B/16 encoder.",
        default_ckpt_path="checkpoints/slot_attention/adaslot_real/COCO.ckpt",
    ),

    # ─── MOVi-C (ViT encoder - SUPPORTED) ─────────────────────────────────
    "movic": AdaSlotConfig(
        name="movic",
        resolution=(224, 224),
        num_slots=11,           # MOVi-C: 3–10 objects
        slot_dim=128,
        num_iterations=3,
        feature_dim=768,
        kvq_dim=128,
        low_bound=1,
        encoder_type="vit",
        source_dataset="MOVi-C",
        description="Pretrained on MOVi-C (video, 3–10 textured objects). ViT-B/16 encoder.",
        default_ckpt_path="checkpoints/slot_attention/adaslot_real/MOVi-C.ckpt",
    ),

    # ─── MOVi-E (ViT encoder - SUPPORTED) ─────────────────────────────────
    "movie": AdaSlotConfig(
        name="movie",
        resolution=(224, 224),
        num_slots=24,           # MOVi-E: up to 23 objects
        slot_dim=128,
        num_iterations=3,
        feature_dim=768,
        kvq_dim=128,
        low_bound=1,
        encoder_type="vit",
        source_dataset="MOVi-E",
        description="Pretrained on MOVi-E (video, up to 23 textured objects). ViT-B/16 encoder.",
        default_ckpt_path="checkpoints/slot_attention/adaslot_real/MOVi-E.ckpt",
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def get_adaslot_config(name: str) -> AdaSlotConfig:
    """
    Retrieve config for a registered checkpoint.

    Args:
        name: Checkpoint identifier (e.g., "clevr10", "coco").

    Returns:
        AdaSlotConfig instance.

    Raises:
        KeyError if the checkpoint is not registered.
    """
    name_lower = name.lower()
    if name_lower not in ADASLOT_CONFIGS:
        available = ", ".join(ADASLOT_CONFIGS.keys())
        raise KeyError(
            f"Unknown checkpoint '{name}'. Available: {available}"
        )
    return ADASLOT_CONFIGS[name_lower]


def list_available_checkpoints() -> list[str]:
    """Return a list of all registered checkpoint names."""
    return list(ADASLOT_CONFIGS.keys())


def build_adaslot_from_checkpoint(
    checkpoint_name: str,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
    strict_load: bool = True,
    reset_gumbel_gate: bool = False,
) -> nn.Module:
    """
    Build an AdaSlotModel and load pretrained weights.

    Args:
        checkpoint_name: Name of the checkpoint config (e.g., "clevr10").
        ckpt_path: Path to the .ckpt file. If None, uses default_ckpt_path.
        device: Device to load the model onto.
        strict_load: Whether to enforce strict state_dict matching.
        reset_gumbel_gate: If True, re-initialise the Gumbel gate weights
            (useful when transferring to a very different domain).

    Returns:
        Loaded AdaSlotModel.
    """
    from cont_src.models.adaslot.model import AdaSlotModel

    cfg = get_adaslot_config(checkpoint_name)

    # Resolve checkpoint path
    if ckpt_path is None:
        if cfg.default_ckpt_path is None:
            raise ValueError(
                f"No default checkpoint path for '{checkpoint_name}'. "
                "Please provide 'ckpt_path' explicitly."
            )
        ckpt_path = cfg.default_ckpt_path

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model
    model = AdaSlotModel(
        resolution=cfg.resolution,
        num_slots=cfg.num_slots,
        slot_dim=cfg.slot_dim,
        num_iterations=cfg.num_iterations,
        feature_dim=cfg.feature_dim,
        kvq_dim=cfg.kvq_dim,
        low_bound=cfg.low_bound,
        encoder_type=cfg.encoder_type,
        vit_pretrained=False,   # don't pull ImageNet weights; ckpt overrides everything
    )

    # Load weights
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"  Config: {cfg.name}  ({cfg.source_dataset})")
    print(
        f"  Resolution: {cfg.resolution}  |  Slots: {cfg.num_slots}  |  Dim: {cfg.slot_dim}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["state_dict"], strict=strict_load)

    if missing or unexpected:
        print(
            f"  ⚠ Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")
        if strict_load and (missing or unexpected):
            raise RuntimeError(
                "Strict load failed. Check model architecture vs checkpoint.")
    else:
        print("  ✓ Weights loaded successfully")

    # Optional: reset Gumbel gate
    if reset_gumbel_gate:
        _reset_gumbel_gate(model)
        print("  ⚠ Gumbel gate reset (use with caution)")

    model.to(device)
    return model


def _reset_gumbel_gate(model: nn.Module) -> None:
    """
    Re-initialize the Gumbel gate network.

    WARNING: This discards learned slot selection behaviour. Use only when
    transferring to a very different domain (e.g., CLEVR → CIFAR-100).
    """
    for name, module in model.named_modules():
        if "single_gumbel_score_network" in name or "gumbel_score" in name:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            print(f"    Gate reset: {name}")


# ═════════════════════════════════════════════════════════════════════════════
# Testing
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("AdaSlot Checkpoint Registry")
    print("=" * 70)

    for name, cfg in ADASLOT_CONFIGS.items():
        print(f"\n{name.upper()}")
        print(f"  Dataset    : {cfg.source_dataset}")
        print(f"  Encoder    : {cfg.encoder_type.upper()}")
        print(f"  Resolution : {cfg.resolution}")
        print(f"  Num slots  : {cfg.num_slots}")
        print(f"  Slot dim   : {cfg.slot_dim}")
        print(f"  Default ckpt: {cfg.default_ckpt_path or 'N/A'}")
        print(f"  Description: {cfg.description}")

    print("\n" + "=" * 70)
    print(f"Total registered checkpoints: {len(ADASLOT_CONFIGS)}")
    print("=" * 70)
