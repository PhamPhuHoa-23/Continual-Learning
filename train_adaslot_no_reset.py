"""
AdaSlot Training Script (NO Gate Reset)
========================================

Fine-tune a pretrained AdaSlot checkpoint WITHOUT resetting the Gumbel gate.

Use this when you trust the pretrained gate's slot selection behaviour and
want to preserve it while adapting to a new domain.

Supports all registered checkpoints:
  - clevr10  (CLEVR, 11 slots)
  - coco     (COCO, 7 slots)
  - movic    (MOVi-C, 11 slots)
  - movie    (MOVi-E, 24 slots)

Usage:
    # Local training (CLEVR10 checkpoint on CIFAR-100)
    python train_adaslot_no_reset.py --checkpoint clevr10 --dataset cifar100 --epochs 5
    
    # Use COCO checkpoint instead
    python train_adaslot_no_reset.py --checkpoint coco --ckpt_path path/to/COCO.ckpt
    
    # MOVi-E (24 slots) - for very complex scenes
    python train_adaslot_no_reset.py --checkpoint movie --dataset tiny_imagenet
"""

from cont_src.training import AdaSlotTrainer, AdaSlotTrainerConfig
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.models.adaslot_configs import (
    get_adaslot_config,
    build_adaslot_from_checkpoint,
    list_available_checkpoints,
)
import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import CIFAR100

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ═════════════════════════════════════════════════════════════════════════════
# Args
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune AdaSlot (no gate reset) on CIFAR-100 or Tiny-ImageNet"
    )

    # Checkpoint selection
    available = list_available_checkpoints()
    parser.add_argument(
        "--checkpoint", type=str, default="clevr10",
        choices=available,
        help=f"Which pretrained checkpoint to use: {', '.join(available)}"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to .ckpt file. If None, uses default from registry."
    )
    parser.add_argument(
        "--reset_gate", action="store_true",
        help="OVERRIDE: Reset Gumbel gate (destroys learned slot selection)"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar100", "tiny_imagenet"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--w_recon", type=float, default=1.0)
    parser.add_argument("--w_sparse", type=float, default=10.0)
    parser.add_argument("--w_prim", type=float, default=5.0)

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/adaslot_finetuned")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_final", action="store_true",
                        help="Save final checkpoint after training")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def get_dataloaders(args, img_size: int):
    """Build train/val/test loaders for the selected dataset."""

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "cifar100":
        train_full = CIFAR100(args.data_root, train=True,
                              download=True, transform=train_tf)
        test_ds = CIFAR100(args.data_root, train=False, transform=val_tf)
    else:
        raise NotImplementedError(
            f"Dataset '{args.dataset}' not implemented yet")

    # Split train into train + val
    n_val = int(len(train_full) * args.val_split)
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# Wrapper model
# ═════════════════════════════════════════════════════════════════════════════

class SlotModelWrapper(nn.Module):
    """Maps AdaSlotModel outputs to trainer convention."""

    def __init__(self, backbone, prim_sel=None):
        super().__init__()
        self.backbone = backbone
        self.prim_sel = prim_sel

    def forward(self, images, **kw):
        out = self.backbone(images, **kw)
        result = {
            "recon": out["reconstruction"],
            "mask": out["hard_keep_decision"],
            "slots": out["slots"],
            **out,
        }
        if self.prim_sel is not None:
            H = self.prim_sel(
                out["slots"], slot_mask=out["hard_keep_decision"])
            result["primitives"] = H.unsqueeze(1)
        return result


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.checkpoint}_{args.dataset}_{timestamp}"

    run_dir = Path(args.output_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load checkpoint config + build model (NO gate reset by default)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset   : {args.dataset}")
    logger.info(f"Device    : {args.device}")
    logger.info("=" * 70)

    cfg = get_adaslot_config(args.checkpoint)
    logger.info(f"AdaSlot Config: {cfg.name}  ({cfg.source_dataset})")
    logger.info(
        f"  Resolution: {cfg.resolution}  |  Slots: {cfg.num_slots}  |  Dim: {cfg.slot_dim}")

    backbone = build_adaslot_from_checkpoint(
        checkpoint_name=args.checkpoint,
        ckpt_path=args.ckpt_path,
        device=args.device,
        strict_load=True,
        reset_gumbel_gate=args.reset_gate,  # False by default!
    )

    if args.reset_gate:
        logger.warning(
            "⚠ Gate was RESET — pretrained slot selection discarded")
    else:
        logger.info(
            "✓ Gate PRESERVED — using pretrained slot selection behaviour")

    # PrimitiveSelector
    prim_sel = PrimitiveSelector(
        slot_dim=cfg.slot_dim,
        hidden_dim=cfg.slot_dim,
    ).to(args.device)

    slot_model = SlotModelWrapper(backbone, prim_sel).to(args.device)

    n_params = sum(p.numel() for p in slot_model.parameters())
    logger.info(f"Total params: {n_params:,}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Data
    # ─────────────────────────────────────────────────────────────────────────
    img_size = cfg.resolution[0]  # assume square
    train_loader, val_loader, test_loader = get_dataloaders(args, img_size)

    logger.info(
        f"Train batches: {len(train_loader)}  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Train
    # ─────────────────────────────────────────────────────────────────────────
    trainer_cfg = AdaSlotTrainerConfig(
        lr=args.lr,
        max_steps=0,
        max_epochs=args.epochs,
        w_recon=args.w_recon,
        w_sparse=args.w_sparse,
        w_prim=args.w_prim,
        checkpoint_dir=str(run_dir),
        log_every_n_steps=args.log_every,
    )

    trainer = AdaSlotTrainer(
        config=trainer_cfg,
        slot_model=slot_model,
        primitive_predictor=None,  # wrapper already has prim_sel
    )

    logger.info("Starting training...")
    metrics = trainer.train(train_loader)

    logger.info("=" * 70)
    logger.info("Training complete!")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 70)

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump({k: ([v] if not isinstance(v, list) else v)
                  for k, v in metrics.items()}, f)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Optional: save final checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    if args.save_final:
        final_ckpt = run_dir / "final.pt"
        torch.save({
            "epoch": args.epochs,
            "model": slot_model.state_dict(),
            "config": vars(args),
            "metrics": metrics,
        }, final_ckpt)
        logger.info(f"Final checkpoint saved: {final_ckpt}")

    logger.info(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
