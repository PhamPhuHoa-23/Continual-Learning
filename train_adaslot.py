"""
Simple AdaSlot Training Script

Train AdaSlot on Task 1 only - no compositional pipeline overhead.
Supports both primitive loss (CompSLOT paper) and clustering loss.
"""

import argparse
import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.models.adaslot.model import AdaSlotModel
from src.data.continual_cifar100 import get_continual_cifar100_loaders
from src.losses.primitive import PrimitiveSelector, ConceptLearningLoss
from src.losses.contrastive import SlotClusteringLoss


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger("adaslot_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─── Args ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AdaSlot on Task 1 (simple script)"
    )

    # Model
    parser.add_argument("--slot_dim",  type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=7)

    # Data
    parser.add_argument("--n_tasks", type=int, default=10,
                        help="Total tasks (only task 1 will be used)")
    parser.add_argument("--n_classes_per_task", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--adaslot_resolution", type=int, default=128,
                        help="Input image resolution for AdaSlot")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sparse_weight", type=float, default=1.0,
                        help="Weight for sparsity penalty")

    # Primitive Loss (CompSLOT Paper)
    parser.add_argument("--use_primitive_loss", action="store_true",
                        help="Use primitive loss (recommended)")
    parser.add_argument("--primitive_alpha", type=float, default=10.0,
                        help="Weight α for primitive loss")
    parser.add_argument("--primitive_temp", type=float, default=10.0,
                        help="Temperature τ_p for primitive similarity")

    # Clustering Loss (Legacy, optional)
    parser.add_argument("--use_clustering_loss", action="store_true",
                        help="Use clustering loss (alternative)")
    parser.add_argument("--clustering_loss_type", type=str, default="contrastive",
                        choices=["contrastive", "prototype"])
    parser.add_argument("--clustering_weight", type=float, default=0.5)
    parser.add_argument("--clustering_temp", type=float, default=0.07)
    parser.add_argument("--slot_aggregation", type=str, default="mean",
                        choices=["mean", "max", "attention"])

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/adaslot_runs")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Test mode
    parser.add_argument("--test_mode", action="store_true",
                        help="Quick test with limited data")
    parser.add_argument("--max_samples", type=int, default=200)

    return parser.parse_args()


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {args.device} | Seed: {args.seed}")

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {config_path}")

    # ─── Load Data (Task 1 only) ────────────────────────────────────────────

    logger.info("Loading CIFAR-100 Task 1...")
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
        max_samples_per_task=args.max_samples if args.test_mode else None,
    )

    # Only use task 1
    train_loader = train_loaders[0]
    test_loader = test_loaders[0]
    task1_classes = class_order[:args.n_classes_per_task]
    
    logger.info(f"Task 1 classes: {task1_classes}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # ─── Initialize Model ───────────────────────────────────────────────────

    logger.info(f"Initializing AdaSlot (slots={args.num_slots}, dim={args.slot_dim})...")
    model = AdaSlotModel(num_slots=args.num_slots, slot_dim=args.slot_dim)
    model.to(args.device)

    # ─── Setup Loss Functions ───────────────────────────────────────────────

    primitive_selector = None
    concept_loss_fn = None
    clustering_loss = None

    if args.use_primitive_loss:
        logger.info("Using Primitive Loss (CompSLOT)")
        logger.info(f"  α={args.primitive_alpha}, τ_p={args.primitive_temp}")
        
        primitive_selector = PrimitiveSelector(
            slot_dim=args.slot_dim,
            temperature=None  # Auto: 100/√D_s
        ).to(args.device)
        
        concept_loss_fn = ConceptLearningLoss(
            alpha=args.primitive_alpha,
            temperature_p=args.primitive_temp
        ).to(args.device)
        
        # Optimize both model and selector
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(primitive_selector.parameters()),
            lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_clustering_loss:
        logger.info(f"Using Clustering Loss: {args.clustering_loss_type}")
        logger.info(f"  weight={args.clustering_weight}, temp={args.clustering_temp}")
        
        clustering_loss = SlotClusteringLoss(
            loss_type=args.clustering_loss_type,
            temperature=args.clustering_temp,
            num_classes=args.n_classes_per_task,
            embedding_dim=args.slot_dim,
            aggregation=args.slot_aggregation
        ).to(args.device)

    if not args.use_primitive_loss and not args.use_clustering_loss:
        logger.warning("No slot loss specified! Using reconstruction only.")

    # ─── Training Loop ──────────────────────────────────────────────────────

    logger.info(f"Starting training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        if primitive_selector is not None:
            primitive_selector.train()

        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'sparse': 0.0,
            'primitive': 0.0,
            'clustering': 0.0
        }
        n_batches = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for X, y in pbar:
            X = X.to(args.device)
            y = y.to(args.device)

            # Resize to AdaSlot resolution
            if X.shape[-1] != args.adaslot_resolution:
                X = F.interpolate(
                    X, size=(args.adaslot_resolution, args.adaslot_resolution),
                    mode="bilinear", align_corners=False
                )

            # Forward
            out = model(X, global_step=epoch)

            # ─── Compute Loss ───────────────────────────────────────────────

            if args.use_primitive_loss:
                # Primitive loss approach
                primitives, weights = primitive_selector(out["slots"])
                
                losses = concept_loss_fn(
                    reconstructed=out["reconstruction"],
                    target=X,
                    primitives=primitives,
                    labels=y
                )
                
                loss = losses['total']
                loss_recon = losses['recon']
                loss_prim = losses['primitive']
                
                # Sparsity penalty
                loss_sparse = args.sparse_weight * out["hard_keep_decision"].float().mean()
                loss = loss + loss_sparse
                
                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['recon'] += loss_recon.item()
                epoch_losses['primitive'] += loss_prim.item()
                epoch_losses['sparse'] += loss_sparse.item()
                
                pbar.set_postfix({
                    'recon': f"{loss_recon.item():.2f}",
                    'prim': f"{loss_prim.item():.3f}",
                    'sparse': f"{loss_sparse.item():.3f}"
                })
            
            else:
                # Standard reconstruction
                loss_recon = F.mse_loss(out["reconstruction"], X, reduction="sum") / X.size(0)
                loss_sparse = args.sparse_weight * out["hard_keep_decision"].float().mean()
                loss = loss_recon + loss_sparse
                
                epoch_losses['recon'] += loss_recon.item()
                epoch_losses['sparse'] += loss_sparse.item()
                
                # Optional clustering loss
                if clustering_loss is not None:
                    loss_cluster = clustering_loss(
                        slots=out["slots"],
                        labels=y,
                        masks=out.get("masks", None)
                    )
                    loss = loss + args.clustering_weight * loss_cluster
                    epoch_losses['clustering'] += loss_cluster.item()
                    
                    pbar.set_postfix({
                        'recon': f"{loss_recon.item():.2f}",
                        'cluster': f"{loss_cluster.item():.3f}",
                        'sparse': f"{loss_sparse.item():.3f}"
                    })
                else:
                    pbar.set_postfix({'loss': f"{loss.item():.3f}"})
                
                epoch_losses['total'] += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            if primitive_selector is not None:
                torch.nn.utils.clip_grad_norm_(primitive_selector.parameters(), 100.0)
            optimizer.step()

            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Log progress
        loss_str = " | ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items() if v > 0)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | {loss_str}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(run_dir, f"adaslot_epoch{epoch+1}.pt")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': epoch_losses,
                'args': vars(args)
            }
            
            if primitive_selector is not None:
                save_dict['primitive_selector_state_dict'] = primitive_selector.state_dict()
            
            torch.save(save_dict, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")
            
            # Update best
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                best_path = os.path.join(run_dir, "adaslot_best.pt")
                torch.save(save_dict, best_path)
                logger.info(f"  Best model updated: {best_path}")

    # ─── Final Save ─────────────────────────────────────────────────────────

    final_path = os.path.join(run_dir, "adaslot_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'primitive_selector_state_dict': primitive_selector.state_dict() if primitive_selector else None,
        'args': vars(args)
    }, final_path)
    logger.info(f"Final checkpoint saved: {final_path}")

    # ─── Simple Evaluation ──────────────────────────────────────────────────

    logger.info("Evaluating reconstruction quality on test set...")
    model.eval()
    test_recon_loss = 0.0
    test_batches = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating", leave=False):
            X = X.to(args.device)
            
            if X.shape[-1] != args.adaslot_resolution:
                X = F.interpolate(
                    X, size=(args.adaslot_resolution, args.adaslot_resolution),
                    mode="bilinear", align_corners=False
                )
            
            out = model(X, global_step=args.epochs)
            loss = F.mse_loss(out["reconstruction"], X, reduction="sum") / X.size(0)
            test_recon_loss += loss.item()
            test_batches += 1

    avg_test_loss = test_recon_loss / max(test_batches, 1)
    logger.info(f"Test reconstruction loss: {avg_test_loss:.4f}")

    logger.info("Training complete!")
    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
