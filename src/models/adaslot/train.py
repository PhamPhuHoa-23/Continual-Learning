"""
End-to-End Training Pipeline for Slot-based Multi-Agent Continual Learning.

Full training flow (3 phases):
  Phase 1: AdaSlot Pretraining
      image → AdaSlot → reconstruction
      Loss = MSE_reconstruction + λ_sparse × SparsePenalty(hard_keep_decision)
      → Teaches the model to decompose scenes into adaptive object slots

  Phase 2: Agent Training (DINO-style self-supervised)
      frozen_AdaSlot(image) → slots
      for each slot → student_agent → logits
                     → teacher_agent → logits (no grad, EMA)
      Loss = DINO_cross_entropy(student, teacher)
      → Teaches agents to produce meaningful hidden labels per slot

  Phase 3: Tree Fitting (Incremental / Continual)
      frozen_AdaSlot(image) → slots
      frozen_agents(slots) → hidden_labels
      concatenated_hidden_labels → Hoeffding Tree → prediction
      → Incremental decision tree grows online, never resets

Usage:
    python -m src.models.adaslot.train --phase 1 --data_dir data/clevr
    python -m src.models.adaslot.train --phase 2 --adaslot_ckpt checkpoints/adaslot.pth
    python -m src.models.adaslot.train --phase 3 --adaslot_ckpt ... --agent_ckpt ...
    python -m src.models.adaslot.train --phase all  # run all 3 sequentially
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ───────────────────────────────────────────────────────
#  Phase 1: AdaSlot Losses
# ───────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """MSE-sum reconstruction loss (from original Slot Attention paper)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) reconstructed image
            target: (B, 3, H, W) original image
        Returns:
            Scalar MSE-sum loss (sum over pixels, mean over batch).
        """
        return F.mse_loss(pred, target, reduction='sum') / pred.shape[0]


class SparsePenalty(nn.Module):
    """
    Penalty to encourage the model to DROP unnecessary slots.

    L_sparse = linear_weight × mean(hard_keep_decision)
             + quadratic_weight × (mean(hard_keep_decision) - bias)²

    A high linear_weight pushes the model to keep fewer slots (lower mean).
    """

    def __init__(
        self,
        linear_weight: float = 10.0,
        quadratic_weight: float = 0.0,
        quadratic_bias: float = 0.5,
    ):
        super().__init__()
        self.linear_weight = linear_weight
        self.quadratic_weight = quadratic_weight
        self.quadratic_bias = quadratic_bias

    def forward(self, hard_keep_decision: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hard_keep_decision: (B, num_slots) binary decisions (1=keep, 0=drop)
        """
        sparse_degree = hard_keep_decision.mean()
        loss = self.linear_weight * sparse_degree
        loss += self.quadratic_weight * (sparse_degree - self.quadratic_bias) ** 2
        return loss


# ───────────────────────────────────────────────────────
#  LR Scheduler (exp decay with linear warmup)
# ───────────────────────────────────────────────────────

def exp_decay_with_warmup(step: int, decay_rate: float, decay_steps: int, warmup_steps: int) -> float:
    """
    LR multiplier: linear warmup → exponential decay.

    warmup:  factor = step / warmup_steps   (0→1 linearly)
    decay:   factor *= decay_rate ^ (step / decay_steps)
    """
    if warmup_steps > 0 and step < warmup_steps:
        warmup_factor = step / warmup_steps
    else:
        warmup_factor = 1.0
    return warmup_factor * (decay_rate ** (step / decay_steps))


def build_scheduler(optimizer, decay_rate=0.5, decay_steps=100000, warmup_steps=10000):
    """Build LambdaLR scheduler with exp-decay + warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: exp_decay_with_warmup(step, decay_rate, decay_steps, warmup_steps),
    )


# ───────────────────────────────────────────────────────
#  Phase 1: AdaSlot Pretraining
# ───────────────────────────────────────────────────────

def train_phase1_adaslot(
    model,
    dataloader: DataLoader,
    num_steps: int = 500000,
    lr: float = 4e-4,
    sparse_linear_weight: float = 10.0,
    save_dir: str = "checkpoints/adaslot",
    save_every: int = 50000,
    log_every: int = 100,
    device: str = "cuda",
    resume_ckpt: Optional[str] = None,
):
    """
    Phase 1: Train AdaSlot to decompose images into object-centric slots.

    Loss = MSE(reconstruction, image) + sparse_penalty(hard_keep_decision)
    """
    print("=" * 60)
    print("PHASE 1: AdaSlot Pretraining")
    print(f"  Steps: {num_steps}, LR: {lr}, Sparse weight: {sparse_linear_weight}")
    print("=" * 60)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, decay_rate=0.5, decay_steps=100000, warmup_steps=10000)

    recon_loss_fn = ReconstructionLoss()
    sparse_loss_fn = SparsePenalty(linear_weight=sparse_linear_weight)

    start_step = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f"  Resumed from step {start_step}")

    os.makedirs(save_dir, exist_ok=True)
    data_iter = iter(dataloader)
    t0 = time.time()

    for step in range(start_step, num_steps):
        # Get next batch (infinite loop over dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)

        # Forward
        out = model(images, global_step=step)

        # Losses
        loss_recon = recon_loss_fn(out['reconstruction'], images)
        loss_sparse = sparse_loss_fn(out['hard_keep_decision'])
        loss_total = loss_recon + loss_sparse

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - t0
            slots_kept = out['hard_keep_decision'].sum(dim=-1).mean().item()
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"  Step {step:>7d} | "
                f"loss={loss_total.item():.4f} "
                f"(recon={loss_recon.item():.4f}, sparse={loss_sparse.item():.4f}) | "
                f"slots_kept={slots_kept:.1f} | "
                f"lr={lr_now:.2e} | "
                f"{elapsed:.0f}s"
            )

        # Checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"adaslot_step{step + 1}.pth")
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"  [SAVED] {ckpt_path}")

    # Save final
    final_path = os.path.join(save_dir, "adaslot_final.pth")
    torch.save({'step': num_steps, 'model': model.state_dict()}, final_path)
    print(f"  [DONE] Phase 1 complete. Final checkpoint: {final_path}")
    return final_path


# ───────────────────────────────────────────────────────
#  Phase 2: Agent DINO Training
# ───────────────────────────────────────────────────────

def train_phase2_agents(
    adaslot_model,
    student_agents: nn.ModuleList,
    teacher_agents: nn.ModuleList,
    dataloader: DataLoader,
    num_steps: int = 100000,
    lr: float = 1e-3,
    ema_momentum: float = 0.996,
    student_temp: float = 0.1,
    teacher_temp: float = 0.07,
    save_dir: str = "checkpoints/agents",
    save_every: int = 10000,
    log_every: int = 100,
    device: str = "cuda",
):
    """
    Phase 2: Train agents with DINO-style self-supervised learning on extracted slots.

    AdaSlot is FROZEN. For each image:
      1. Extract slots via frozen AdaSlot
      2. Each agent processes each kept slot
      3. Student-teacher cross-entropy loss (DINO)
      4. Update teacher via EMA
    """
    from src.slot_multi_agent.atomic_agent import DINOLoss, update_all_teachers

    print("=" * 60)
    print("PHASE 2: Agent DINO Training")
    print(f"  Steps: {num_steps}, LR: {lr}, EMA: {ema_momentum}")
    print(f"  Agents: {len(student_agents)} students + {len(teacher_agents)} teachers")
    print("=" * 60)

    adaslot_model = adaslot_model.to(device).eval()
    for p in adaslot_model.parameters():
        p.requires_grad_(False)

    student_agents = student_agents.to(device)
    teacher_agents = teacher_agents.to(device)
    teacher_agents.requires_grad_(False)

    num_prototypes = student_agents[0].num_prototypes
    dino_loss_fn = DINOLoss(
        num_prototypes=num_prototypes,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
    ).to(device)

    optimizer = torch.optim.AdamW(student_agents.parameters(), lr=lr, weight_decay=0.04)
    scheduler = build_scheduler(optimizer, decay_rate=0.5, decay_steps=50000, warmup_steps=5000)

    os.makedirs(save_dir, exist_ok=True)
    data_iter = iter(dataloader)
    t0 = time.time()

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)

        # 1. Extract slots (frozen AdaSlot)
        with torch.no_grad():
            adaslot_out = adaslot_model(images)
            slots = adaslot_out['slots']           # (B, num_slots, slot_dim)
            keep = adaslot_out['hard_keep_decision']  # (B, num_slots)

        B, num_slots, slot_dim = slots.shape

        # 2. DINO training: each agent processes all kept slots
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0

        # Randomly select a subset of agents to train each step (efficiency)
        num_agents_to_train = min(5, len(student_agents))
        agent_indices = torch.randperm(len(student_agents))[:num_agents_to_train]

        for agent_idx in agent_indices:
            student = student_agents[agent_idx]
            teacher = teacher_agents[agent_idx]

            # Flatten kept slots across batch
            # Use keep mask to select active slots
            flat_slots = slots.reshape(B * num_slots, slot_dim)
            flat_keep = keep.reshape(B * num_slots)
            active_slots = flat_slots[flat_keep > 0.5]  # (N_active, slot_dim)

            if active_slots.shape[0] < 2:
                continue

            # Student forward
            student_logits = student(active_slots, return_logits=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(active_slots, return_logits=True)

            # DINO loss
            loss = dino_loss_fn(student_logits, teacher_logits)
            total_loss = total_loss + loss
            num_losses += 1

        if num_losses > 0:
            total_loss = total_loss / num_losses

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_agents.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # EMA update teachers
            update_all_teachers(student_agents, teacher_agents, momentum=ema_momentum)

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"  Step {step:>7d} | "
                f"dino_loss={total_loss.item():.4f} | "
                f"lr={lr_now:.2e} | "
                f"{elapsed:.0f}s"
            )

        # Checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"agents_step{step + 1}.pth")
            torch.save({
                'step': step + 1,
                'student_agents': student_agents.state_dict(),
                'teacher_agents': teacher_agents.state_dict(),
                'dino_center': dino_loss_fn.center,
            }, ckpt_path)
            print(f"  [SAVED] {ckpt_path}")

    final_path = os.path.join(save_dir, "agents_final.pth")
    torch.save({
        'student_agents': student_agents.state_dict(),
        'teacher_agents': teacher_agents.state_dict(),
    }, final_path)
    print(f"  [DONE] Phase 2 complete. Final: {final_path}")
    return final_path


# ───────────────────────────────────────────────────────
#  Phase 3: CRP Expert Assignment Aggregator
# ───────────────────────────────────────────────────────

def train_phase3_crp(
    adaslot_model,
    student_agents: nn.ModuleList,
    dataloader: DataLoader,
    num_classes: int = 10,
    aggregator_type: str = "crp",
    aggregate_mode: str = "concat",
    k: int = 3,
    device: str = "cuda",
    log_every: int = 100,
):
    """
    Phase 3: Incrementally fit the aggregator on agent hidden labels.

    AdaSlot + Agents are FROZEN. For each batch:
      1. Extract slots
      2. Each agent produces hidden labels per slot
      3. Concatenate/mean-pool across slots
      4. Aggregator learns incrementally (learn_batch / partial_fit depending on type)
    """
    from src.slot_multi_agent.aggregator import create_aggregator
    import numpy as np

    print("=" * 60)
    print("PHASE 3: Incremental Aggregator Fitting")
    print(f"  Classes: {num_classes}, Aggregator: {aggregator_type}")
    print(f"  Mode: {aggregate_mode}, k={k}")
    print("=" * 60)

    adaslot_model = adaslot_model.to(device).eval()
    student_agents = student_agents.to(device).eval()
    for p in adaslot_model.parameters():
        p.requires_grad_(False)
    for p in student_agents.parameters():
        p.requires_grad_(False)

    num_agents = len(student_agents)
    num_slots = adaslot_model.num_slots
    slot_dim = adaslot_model.slot_dim
    hidden_dim = student_agents[0].num_prototypes

    # Compute aggregator input dimension
    if aggregate_mode == "concat":
        input_dim = num_slots * k * hidden_dim
    else:
        input_dim = k * hidden_dim

    # Prepare kwargs based on aggregator type
    agg_kwargs = {
        "num_classes": num_classes,
        "device": device,
    }
    if aggregator_type == "crp":
        agg_kwargs["feature_dim"] = input_dim
    else:
        agg_kwargs["input_dim"] = input_dim

    aggregator = create_aggregator(
        aggregator_type=aggregator_type,
        **agg_kwargs
    )

    total_correct = 0
    total_samples = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)
        targets = batch[1] if isinstance(batch, (list, tuple)) else batch['label']

        with torch.no_grad():
            # 1. Slots
            adaslot_out = adaslot_model(images)
            slots = adaslot_out['slots']              # (B, S, D)

            B, S, D = slots.shape

            # 2. Agent hidden labels
            all_features = []
            for b in range(B):
                sample_hidden = []
                for slot_idx in range(S):
                    slot = slots[b, slot_idx].unsqueeze(0)  # (1, D)
                    # Use top-k agents (simplification: use first k agents)
                    slot_labels = []
                    for a_idx in range(min(k, num_agents)):
                        agent = student_agents[a_idx]
                        label = agent(slot)  # (1, num_prototypes)
                        slot_labels.append(label.squeeze(0))
                    sample_hidden.append(torch.cat(slot_labels))  # (k * proto,)

                if aggregate_mode == "concat":
                    feat = torch.cat(sample_hidden)       # (S * k * proto,)
                else:
                    feat = torch.stack(sample_hidden).mean(0)  # (k * proto,)
                all_features.append(feat)

            # Features are tensors for CRP, we only convert to numpy when calling trees
            features_t = torch.stack(all_features)  # (B, dim) on CPU

        # 3. Aggregator: try to predict, then learn
        try:
            if hasattr(aggregator, 'predict_batch'):
                preds = aggregator.predict_batch(features_t)
                correct = sum(1 for p, t in zip(preds, targets.cpu().numpy()) if p == t)
            else:
                preds = aggregator.predict(features_t.numpy())
                correct = (preds == targets.cpu().numpy()).sum()
            total_correct += correct
        except Exception:
            # Predict might fail on the very first batch before learning
            pass

        total_samples += B
        
        if hasattr(aggregator, 'learn_batch'):
            aggregator.learn_batch(features_t, targets.cpu())
        else:
            aggregator.partial_fit(features_t.numpy(), targets.cpu().numpy())

        if (batch_idx + 1) % log_every == 0:
            acc = total_correct / max(total_samples, 1) * 100
            elapsed = time.time() - t0
            print(
                f"  Batch {batch_idx + 1:>5d} | "
                f"samples={total_samples} | "
                f"acc={acc:.2f}% | "
                f"{elapsed:.0f}s"
            )

    final_acc = total_correct / max(total_samples, 1) * 100
    print(f"  [DONE] Phase 3 complete. Final accuracy: {final_acc:.2f}%")
    return aggregator


# ───────────────────────────────────────────────────────
#  Dummy Dataset (for testing)
# ───────────────────────────────────────────────────────

class DummyImageDataset(Dataset):
    """Random images for smoke-testing the pipeline."""

    def __init__(self, num_samples: int = 1000, resolution: int = 128, num_classes: int = 10):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.resolution, self.resolution)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


# ───────────────────────────────────────────────────────
#  Main Entry Point
# ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Slot-based Multi-Agent System")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Training phase: 1=AdaSlot, 2=Agents, 3=Tree, all=sequential")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--adaslot_ckpt", type=str, default=None, help="AdaSlot checkpoint path")
    parser.add_argument("--agent_ckpt", type=str, default=None, help="Agent checkpoint path")
    parser.add_argument("--num_slots", type=int, default=11)
    parser.add_argument("--slot_dim", type=int, default=64)
    parser.add_argument("--num_agents", type=int, default=50)
    parser.add_argument("--num_prototypes", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
        
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--resolution", type=int, default=128)

    # Phase-specific
    parser.add_argument("--p1_steps", type=int, default=500000, help="Phase 1 training steps")
    parser.add_argument("--p1_lr", type=float, default=4e-4)
    parser.add_argument("--p2_steps", type=int, default=100000, help="Phase 2 training steps")
    parser.add_argument("--p2_lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default="CLEVR10",
                        choices=["CLEVR10", "COCO", "MOVi-C", "MOVi-E", "none"],
                        help="Pretrained AdaSlot checkpoint to load (default: CLEVR10). "
                             "Use 'none' to train from scratch.")

    args = parser.parse_args()

    # ── Build models ──
    from . import AdaSlotModel

    adaslot = AdaSlotModel(
        resolution=(args.resolution, args.resolution),
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
    )

    # ── Load pretrained weights ──
    if args.adaslot_ckpt and os.path.exists(args.adaslot_ckpt):
        # Explicit checkpoint path provided
        ckpt = torch.load(args.adaslot_ckpt, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        result = adaslot.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded AdaSlot weights from {args.adaslot_ckpt}")
        print(f"       Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    elif args.pretrained != "none":
        # Auto-detect pretrained checkpoint
        pretrained_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "checkpoints", "slot_attention", "adaslot_real", "AdaSlotCkpt"
        )
        ckpt_path = os.path.join(pretrained_dir, f"{args.pretrained}.ckpt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
            result = adaslot.load_state_dict(state, strict=False)
            print(f"[INFO] ✅ Loaded pretrained AdaSlot ({args.pretrained}) from {ckpt_path}")
            print(f"       Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
        else:
            print(f"[WARN] Pretrained checkpoint not found: {ckpt_path}")
            print(f"       Download from: https://drive.google.com/drive/folders/1SRKE9Q5XF2UeYj1XB8kyjxORDmB7c7Mz")
            print(f"       Training from scratch instead.")

    # ── Build dataloader ──
    if args.data_dir:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
    else:
        print("[INFO] No data_dir specified, downloading and using CIFAR-100 continual benchmark.")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "continual_cifar100", 
            "src/data/continual_cifar100.py"
        )
        data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        get_continual_cifar100_loaders = data_module.get_continual_cifar100_loaders
        # Get loaders for task 0 as our main training data
        train_loaders, _, _ = get_continual_cifar100_loaders(
            n_tasks=int(100 / args.num_classes) if args.num_classes > 0 else 5,
            batch_size=args.batch_size,
            num_workers=0,
            seed=42,
            resolution=args.resolution
        )
        dataloader = train_loaders[0]  # Just use the first task loader

    # ── Execute phases ──
    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]

    for phase in phases:
        if phase == "1":
            adaslot_ckpt_path = train_phase1_adaslot(
                model=adaslot,
                dataloader=dataloader,
                num_steps=args.p1_steps,
                lr=args.p1_lr,
                device=args.device,
            )
        elif phase == "2":
            # Load AdaSlot
            if args.adaslot_ckpt:
                ckpt = torch.load(args.adaslot_ckpt, map_location='cpu', weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                adaslot.load_state_dict(state, strict=False)

            # Create agent pool
            from src.slot_multi_agent.atomic_agent import create_agent_pool
            student_agents, teacher_agents = create_agent_pool(
                num_agents=args.num_agents,
                slot_dim=args.slot_dim,
                num_prototypes=args.num_prototypes,
                hidden_dim=256,
                num_blocks=3,
                device=args.device,
            )

            train_phase2_agents(
                adaslot_model=adaslot,
                student_agents=student_agents,
                teacher_agents=teacher_agents,
                dataloader=dataloader,
                num_steps=args.p2_steps,
                lr=args.p2_lr,
                device=args.device,
            )

        elif phase == "3":
            # Load AdaSlot
            if args.adaslot_ckpt:
                ckpt = torch.load(args.adaslot_ckpt, map_location='cpu', weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                adaslot.load_state_dict(state, strict=False)

            # Load Agents
            from src.slot_multi_agent.atomic_agent import create_agent_pool
            student_agents, _ = create_agent_pool(
                num_agents=args.num_agents,
                slot_dim=args.slot_dim,
                num_prototypes=args.num_prototypes,
                device=args.device,
            )
            if args.agent_ckpt:
                ckpt = torch.load(args.agent_ckpt, map_location='cpu', weights_only=False)
                student_agents.load_state_dict(ckpt['student_agents'])

            # Pass student_agents to phase 3 aggregator
            from src.slot_multi_agent.aggregator import create_aggregator
            aggregator = train_phase3_crp(
                adaslot_model=adaslot,
                student_agents=student_agents,
                dataloader=dataloader,
                num_classes=args.num_classes,
                aggregator_type="crp",
                device=args.device,
            )

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
