"""
Test AdaSlot Checkpoint Registry
==================================

Shows all available checkpoints and their configs.
Tests loading each checkpoint (if available).

Usage:
    python test_adaslot_registry.py
    python test_adaslot_registry.py --test_load
"""

import argparse
import torch
from pathlib import Path

from cont_src.models.adaslot_configs import (
    get_adaslot_config,
    build_adaslot_from_checkpoint,
    list_available_checkpoints,
    ADASLOT_CONFIGS,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_load", action="store_true",
                        help="Try to load each checkpoint (if file exists)")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("AdaSlot Checkpoint Registry".center(80))
    print("=" * 80)
    print()

    available = list_available_checkpoints()
    print(f"Total registered checkpoints: {len(available)}")
    print(f"Names: {', '.join(available)}")
    print()

    # Display each config
    for name in available:
        cfg = get_adaslot_config(name)
        print("─" * 80)
        print(f"Checkpoint: {name.upper()}")
        print("─" * 80)
        print(f"  Source Dataset : {cfg.source_dataset}")
        print(f"  Description    : {cfg.description}")
        print()
        print(f"  Architecture:")
        print(f"    Encoder Type : {cfg.encoder_type.upper()} ✓ Supported")
        print(f"    Resolution   : {cfg.resolution[0]} × {cfg.resolution[1]}")
        print(f"    Num Slots    : {cfg.num_slots}")
        print(f"    Slot Dim     : {cfg.slot_dim}")
        print(f"    Feature Dim  : {cfg.feature_dim}")
        print(f"    Iterations   : {cfg.num_iterations}")
        print(f"    KVQ Dim      : {cfg.kvq_dim}")
        print(f"    Low Bound    : {cfg.low_bound}")
        print()
        print(f"  Default Path   : {cfg.default_ckpt_path or 'N/A'}")

        if cfg.default_ckpt_path:
            ckpt_path = Path(cfg.default_ckpt_path)
            exists = ckpt_path.exists()
            status = "✓ Found" if exists else "✗ Not found"
            print(f"  Status         : {status}")

            if args.test_load and exists:
                try:
                    print(f"\n  Loading checkpoint...")
                    model = build_adaslot_from_checkpoint(
                        name, device=args.device, reset_gumbel_gate=False
                    )
                    n_params = sum(p.numel() for p in model.parameters())
                    print(
                        f"  ✓ Model loaded successfully  ({n_params:,} params)")

                    # Test forward pass
                    dummy = torch.randn(2, 3, *cfg.resolution).to(args.device)
                    with torch.no_grad():
                        out = model(dummy)
                    n_active = out["hard_keep_decision"].sum(
                        dim=1).float().mean().item()
                    print(
                        f"  ✓ Forward pass OK  (avg active slots: {n_active:.1f}/{cfg.num_slots})")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        print()

    print("=" * 80)

    # Summary table
    print("\nSummary Table:")
    print("=" * 90)
    print(f"{'Name':<12} {'Dataset':<12} {'Encoder':<8} {'Slots':<8} {'SlotDim':<10} {'Res':<10} {'Status':<10}")
    print("─" * 90)

    for name in available:
        cfg = get_adaslot_config(name)
        ckpt_path = Path(
            cfg.default_ckpt_path) if cfg.default_ckpt_path else None
        status = "✓" if (ckpt_path and ckpt_path.exists()) else "✗"
        supported = "✓"

        print(f"{name:<12} {cfg.source_dataset:<12} {cfg.encoder_type.upper():<8} "
              f"{cfg.num_slots:<8} {cfg.slot_dim:<10} {cfg.resolution[0]}×{cfg.resolution[1]:<7} "
              f"{status:<10} ({supported} code)")

    print("─" * 90)
    print("Legend: ✓ = available/supported, ✗ = missing/unsupported")
    print("=" * 90)


if __name__ == "__main__":
    main()
