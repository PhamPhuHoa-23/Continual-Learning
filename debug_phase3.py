"""
Debug Phase 3 — Feed TinyImageNet directly into CRP Aggregator.

Bypass Phase 1 (AdaSlot) and Phase 2 (Agents) entirely.
Use a pretrained ResNet18 as a simple feature extractor, project features
to the shape CRP expects (num_slots × agent_dim), and feed them straight
into the CRP aggregator.

Purpose:
    Analyze Phase 3 in isolation.  If Phase 3 works well with "good" CNN
    features, the problem lies in earlier phases.  Run each phase backwards:
        1. This script  → test Phase 3 alone
        2. Add agents   → test Phase 2 + 3
        3. Add AdaSlot  → test Phase 1 + 2 + 3

Usage:
    # Quick test on CPU (small config)
    python debug_phase3.py --device cpu --batch_size 8 --num_classes 20

    # GPU run with full config
    python debug_phase3.py --device cuda --num_classes 20 --epochs 5

    # Use a different backbone
    python debug_phase3.py --backbone resnet34 --device cuda
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────
#  Simple Feature Extractor (replaces AdaSlot + Agents)
# ─────────────────────────────────────────────────────────────────────

class SimpleFeatureExtractor(nn.Module):
    """
    Pretrained CNN → project to (num_slots × agent_dim) flat features.

    This replaces the entire Phase 1 (AdaSlot) + Phase 2 (Agents) pipeline
    with a single pretrained backbone + a learned linear projection.

    The output is shaped exactly like what Phase 3 expects:
        (B, num_slots * agent_dim)  →  reshaped inside CRP to  (B, num_slots, agent_dim)
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_slots: int = 11,
        agent_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.agent_dim = agent_dim
        self.output_dim = num_slots * agent_dim

        # Load pretrained backbone
        import torchvision.models as models
        backbone_fn = getattr(models, backbone_name, None)
        if backbone_fn is None:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        backbone = backbone_fn(weights="DEFAULT")

        # Remove the classification head → use as feature extractor
        if hasattr(backbone, "fc"):
            backbone_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            backbone_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Cannot find classification head in {backbone_name}")

        self.backbone = backbone
        self.backbone_dim = backbone_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Project backbone features → (num_slots * agent_dim)
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, num_slots * agent_dim)
        """
        with torch.no_grad():
            feat = self.backbone(images)  # (B, backbone_dim)
        feat = self.projector(feat)       # (B, num_slots * agent_dim)
        return feat


# ─────────────────────────────────────────────────────────────────────
#  Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────

def print_tensor_stats(name: str, t: torch.Tensor):
    """Print basic statistics for a tensor."""
    t_flat = t.detach().float().cpu()
    print(f"  {name:30s} | shape={list(t.shape)} "
          f"| min={t_flat.min().item():.4f} max={t_flat.max().item():.4f} "
          f"| mean={t_flat.mean().item():.4f} std={t_flat.std().item():.4f} "
          f"| nan={t_flat.isnan().any().item()} inf={t_flat.isinf().any().item()}")


def diagnose_features(features: torch.Tensor, label: str = "features"):
    """Detailed diagnostics on feature vectors."""
    print(f"\n{'─' * 60}")
    print(f"  DIAGNOSTICS: {label}")
    print(f"{'─' * 60}")
    print_tensor_stats("raw", features)

    # Per-sample norms
    norms = features.norm(dim=-1)
    print_tensor_stats("per-sample L2 norm", norms)

    # Sparsity (fraction of near-zero elements)
    sparsity = (features.abs() < 1e-6).float().mean().item()
    print(f"  {'Sparsity (|x| < 1e-6)':30s} | {sparsity:.4f}")

    # Cosine similarity between samples in batch
    if features.shape[0] > 1:
        normed = F.normalize(features.detach(), dim=-1)
        cos_sim = (normed @ normed.T)
        # Mask diagonal
        mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool)
        off_diag = cos_sim[mask]
        print(f"  {'Pairwise cos-sim (off-diag)':30s} | "
              f"mean={off_diag.mean().item():.4f} "
              f"std={off_diag.std().item():.4f}")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────
#  Main debug pipeline
# ─────────────────────────────────────────────────────────────────────

def run_phase3_debug(args):
    device = args.device
    num_slots = args.num_slots
    agent_dim = args.agent_dim
    feature_dim = num_slots * agent_dim

    # ── 1. Load TinyImageNet data ──────────────────────────────────────
    print("=" * 60)
    print("PHASE 3 DEBUG — Direct Image → CRP (no AdaSlot, no Agents)")
    print("=" * 60)
    print(f"  Backbone      : {args.backbone}")
    print(f"  num_slots     : {num_slots}")
    print(f"  agent_dim     : {agent_dim}")
    print(f"  feature_dim   : {feature_dim}")
    print(f"  num_classes   : {args.num_classes}")
    print(f"  device        : {device}")
    print(f"  epochs        : {args.epochs}")
    print()

    print("[1/4] Loading TinyImageNet data...")
    from src.data.continual_tinyimagenet import get_continual_tinyimagenet_loaders

    _classes_per_task = args.num_classes
    if 200 % _classes_per_task != 0:
        raise ValueError(
            f"--num_classes {_classes_per_task} must divide 200 evenly."
        )
    n_tasks = 200 // _classes_per_task

    train_loaders, test_loaders, class_order = get_continual_tinyimagenet_loaders(
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        resolution=args.resolution,
        root="./data",
    )
    print(f"  → {n_tasks} tasks, {_classes_per_task} classes/task")
    print(f"  → class_order shape: {class_order.shape}")

    # ── 2. Build feature extractor ─────────────────────────────────────
    print(f"\n[2/4] Building feature extractor ({args.backbone})...")
    feature_extractor = SimpleFeatureExtractor(
        backbone_name=args.backbone,
        num_slots=num_slots,
        agent_dim=agent_dim,
        freeze_backbone=True,
    ).to(device)
    print(f"  → backbone_dim: {feature_extractor.backbone_dim}")
    print(f"  → output_dim: {feature_extractor.output_dim}")

    # Quick sanity check with a dummy batch
    dummy = torch.randn(2, 3, args.resolution, args.resolution, device=device)
    dummy_out = feature_extractor(dummy)
    print(f"  → Sanity check: input {list(dummy.shape)} → output {list(dummy_out.shape)}")
    del dummy, dummy_out

    # ── 3. Build CRP Aggregator ────────────────────────────────────────
    print("\n[3/4] Building CRP Aggregator...")
    from src.slot_multi_agent.aggregator import create_aggregator

    total_classes = n_tasks * _classes_per_task  # = 200
    aggregator = create_aggregator(
        aggregator_type="crp",
        feature_dim=feature_dim,
        num_slots=num_slots,
        agent_dim=agent_dim,
        num_classes=total_classes,
        device=device,
    )
    print(f"  → feature_dim: {feature_dim}")
    print(f"  → total_classes: {total_classes}")

    # ── 4. Optionally train projector end-to-end with CRP ──────────────
    # The projector is the ONLY trainable part (backbone is frozen).
    # This lets the projector learn good features for CRP.
    proj_optimizer = torch.optim.Adam(
        feature_extractor.projector.parameters(), lr=args.proj_lr
    )

    # ── 5. Per-task Phase 3 loop ───────────────────────────────────────
    print(f"\n[4/4] Running Phase 3 per-task loop ({n_tasks} tasks)...")
    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    seen_classes = set()

    for task_id in range(n_tasks):
        task_cls = class_order[
            task_id * _classes_per_task : (task_id + 1) * _classes_per_task
        ].tolist()

        print(f"\n{'=' * 60}")
        print(f"TASK {task_id + 1}/{n_tasks}  —  classes: {task_cls}")
        print(f"{'=' * 60}")

        # Freeze old class queries
        if hasattr(aggregator, '_agg') and seen_classes:
            aggregator._agg.freeze_old_classes(seen_classes)

        # ── TRAIN ──
        task_loader = train_loaders[task_id]
        total_correct = 0
        total_samples = 0
        t0 = time.time()

        for epoch in range(args.epochs):
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, batch in enumerate(task_loader):
                images = batch[0].to(device)
                targets = batch[1]

                # Extract features (replaces AdaSlot + Agents)
                features = feature_extractor(images)  # (B, feature_dim)

                # Diagnose first batch of first epoch
                if batch_idx == 0 and epoch == 0 and task_id == 0:
                    diagnose_features(features, f"Task {task_id} — CNN features")

                B = features.shape[0]

                # Predict
                try:
                    preds = aggregator.predict_batch(features.detach())
                    correct = sum(
                        1 for p, t in zip(preds, targets.cpu().numpy()) if p == t
                    )
                    epoch_correct += correct
                    total_correct += correct
                except Exception:
                    pass

                total_samples += B
                epoch_samples += B

                # Learn (CRP aggregator)
                aggregator.learn_batch(features.detach(), targets.cpu())

                # Also backprop through projector to improve features
                # We compute a proxy cross-entropy loss using the CRP's
                # current expert state
                if args.train_projector:
                    _train_projector_step(
                        feature_extractor, aggregator, images, targets,
                        proj_optimizer, device, num_slots, agent_dim,
                    )

                if (batch_idx + 1) % args.log_every == 0:
                    acc = epoch_correct / max(epoch_samples, 1) * 100
                    elapsed = time.time() - t0
                    print(
                        f"    Batch {batch_idx + 1:>5d} | "
                        f"samples={epoch_samples} | "
                        f"acc={acc:.2f}% | "
                        f"{elapsed:.0f}s"
                    )

            if args.epochs > 1:
                epoch_acc = epoch_correct / max(epoch_samples, 1) * 100
                print(f"  [Epoch {epoch + 1}/{args.epochs}] acc={epoch_acc:.2f}%")

        train_acc = total_correct / max(total_samples, 1) * 100
        print(f"  [TRAIN DONE] Task {task_id} accuracy: {train_acc:.2f}%")

        # ── EVALUATE ──
        print(f"  [EVAL] Per-task accuracy after task {task_id}:")
        row_accs = []
        for eval_task in range(task_id + 1):
            acc = _eval_task(
                feature_extractor, aggregator,
                test_loaders[eval_task], device,
            )
            acc_matrix[task_id][eval_task] = acc
            row_accs.append(acc)

        row_str = "  | ".join(
            f"T{j}: {acc_matrix[task_id][j]:.1f}%"
            for j in range(task_id + 1)
        )
        print(f"  ├─ {row_str}")
        avg_acc = sum(row_accs) / len(row_accs)
        print(f"  └─ Avg so far: {avg_acc:.2f}%")

        # Aggregator stats
        if hasattr(aggregator, 'get_stats'):
            stats = aggregator.get_stats()
            print(f"  [CRP Stats] experts={stats['num_experts']}, "
                  f"classes_seen={stats['num_classes_seen']}, "
                  f"total_samples={stats['total_samples']}")

        seen_classes |= set(task_cls)

    # ── Final summary ──
    _print_final_summary(acc_matrix, n_tasks)


def _train_projector_step(
    feature_extractor, aggregator, images, targets,
    optimizer, device, num_slots, agent_dim,
):
    """Optional: backprop through projector using CRP's class queries."""
    if not hasattr(aggregator, 'aggregator'):
        return
    crp = aggregator.aggregator
    if len(crp.experts) == 0 or len(crp.seen_classes) == 0:
        return

    features = feature_extractor(images)  # (B, F)
    H = features.view(-1, num_slots, agent_dim)  # (B, S, D)

    # Forward through CRP experts
    all_z, all_scores, _ = crp._forward_all_experts(H)
    repr_vec = crp._moe_aggregate(all_z, all_scores)  # (B, d)
    logits = repr_vec @ crp.class_queries.T  # (B, C)

    # Mask unseen classes
    mask = torch.full_like(logits, float("-inf"))
    mask[:, list(crp.seen_classes)] = 0.0
    logits = logits + mask

    targets_dev = targets.to(device)
    loss = F.cross_entropy(logits, targets_dev)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _eval_task(feature_extractor, aggregator, test_loader, device):
    """Evaluate accuracy on a single task's test set."""
    feature_extractor.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            targets = batch[1]

            features = feature_extractor(images)
            B = features.shape[0]

            try:
                preds = aggregator.predict_batch(features)
                correct = sum(
                    1 for p, t in zip(preds, targets.cpu().numpy()) if p == t
                )
                total_correct += correct
            except Exception:
                pass

            total_samples += B

    feature_extractor.train()
    acc = total_correct / max(total_samples, 1) * 100
    return acc


def _print_final_summary(acc_matrix, n_tasks):
    """Print CL metrics summary."""
    print(f"\n{'=' * 60}")
    print("FINAL CL METRICS SUMMARY")
    print(f"{'=' * 60}")

    # Accuracy matrix
    print("\nAccuracy Matrix (row=trained up to task, col=evaluated on task):")
    header = "       " + "".join(f"  T{j:<4d}" for j in range(n_tasks))
    print(header)
    for i in range(n_tasks):
        row = f"  T{i:<3d}"
        for j in range(n_tasks):
            val = acc_matrix[i][j]
            if val is not None:
                row += f" {val:5.1f}%"
            else:
                row += "    — "
        print(row)

    # Average accuracy after final task
    final_accs = [
        acc_matrix[n_tasks - 1][j]
        for j in range(n_tasks)
        if acc_matrix[n_tasks - 1][j] is not None
    ]
    if final_accs:
        avg_final = sum(final_accs) / len(final_accs)
        print(f"\n  Average accuracy (final task): {avg_final:.2f}%")

    # Forgetting
    if n_tasks > 1:
        forgetting_vals = []
        for j in range(n_tasks - 1):
            best_acc = max(
                acc_matrix[i][j]
                for i in range(j, n_tasks)
                if acc_matrix[i][j] is not None
            )
            final_acc = acc_matrix[n_tasks - 1][j]
            if final_acc is not None:
                forgetting_vals.append(best_acc - final_acc)
        if forgetting_vals:
            avg_forget = sum(forgetting_vals) / len(forgetting_vals)
            print(f"  Average forgetting:            {avg_forget:.2f}%")

    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Debug Phase 3 — Direct Image → CRP (bypass AdaSlot + Agents)"
    )
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model (resnet18, resnet34, resnet50, etc.)")
    parser.add_argument("--num_slots", type=int, default=11,
                        help="Number of 'virtual slots' for CRP reshaping")
    parser.add_argument("--agent_dim", type=int, default=256,
                        help="Agent output dimension per slot (proto_dim)")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Classes per task (must divide 200)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per task")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--proj_lr", type=float, default=1e-3,
                        help="Learning rate for the projector head")
    parser.add_argument("--train_projector", action="store_true",
                        help="Also train the projector via CRP backprop "
                             "(end-to-end through CRP experts)")

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()
    run_phase3_debug(args)


if __name__ == "__main__":
    main()
