"""
End-to-End Training Pipeline for Slot-based Multi-Agent Continual Learning.

Full training flow (4 phases):
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

  Phase 2.5: Estimator Training (VAE + MLP)
      frozen_AdaSlot(image) → slots
      frozen_agents(slots) → quality scores (confidence = 1 − normalized entropy)
      VAE_i trained with quality-weighted reconstruction loss per agent i
      MLP trained supervised: (slot, agent_id) → quality_score
      → Enables fast filtering of agents suited to each slot

  Phase 3: Continual Classification (Filter → UCB MoE → CRP)
      frozen_AdaSlot(image) → slots
      frozen_agents + estimators:
          for each slot:
              VAE/MLP scores → filter top-K agents
              UCB Weighted MoE → committee weights for all K agents
              weighted_output = Σ w_i × agent_i(slot)       (MoE gate)
          concat all slot outputs → CRP Expert Aggregator → prediction
      UCB updates: reward ∝ correctness × weight
      → Online continual classification with dynamic agent selection

Usage:
    python -m src.models.adaslot.train --phase 1
    python -m src.models.adaslot.train --phase 2   --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth
    python -m src.models.adaslot.train --phase 2.5 --adaslot_ckpt ... --agent_ckpt ...
    python -m src.models.adaslot.train --phase 3   --adaslot_ckpt ... --agent_ckpt ... --estimator_ckpt ...
    python -m src.models.adaslot.train --phase all  # run all 4 sequentially
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
#  Phase 2.5: Performance Estimator Training (VAE + MLP)
# ───────────────────────────────────────────────────────

def train_phase2b_estimators(
    adaslot_model,
    student_agents: nn.ModuleList,
    dataloader: DataLoader,
    num_steps: int = 20000,
    lr: float = 1e-3,
    vae_beta: float = 0.5,
    agents_per_step: int = 10,
    save_dir: str = "checkpoints/estimators",
    save_every: int = 5000,
    log_every: int = 100,
    device: str = "cuda",
):
    """
    Phase 2.5: Train VAE and MLP performance estimators.

    AdaSlot + Agents are FROZEN.  For each training step:
      1. Extract slots via frozen AdaSlot.
      2. Run a random subset of agents on active slots to compute
         per-agent *quality scores*:
             quality_i(slot) = 1 − H(agent_i(slot)) / log(num_prototypes)
         where H is the entropy of the agent's softmax output.
         High quality → peaked output → the agent "understands" this slot.
      3. Train per-agent **VAE estimators** with quality-weighted
         reconstruction loss.  The VAE learns to reconstruct slots that
         the agent handles well, so at inference time a low
         reconstruction error ↔ high predicted performance.
      4. Train a shared **MLP estimator** with supervised loss:
             (slot, agent_id) → quality   (MSE regression).

    These estimators are later used in Phase 3 to **fast-filter** the
    50-agent pool down to a small top-K set per slot, before the UCB
    Weighted MoE assigns committee weights.

    Args:
        adaslot_model: Frozen AdaSlot model.
        student_agents: Frozen student agents (from Phase 2).
        dataloader: Training data (same as Phase 2).
        num_steps: Number of training steps.
        lr: Learning rate for estimator parameters.
        vae_beta: Weight for KL divergence term in VAE loss.
        agents_per_step: How many agents to sample per step (efficiency).
        save_dir: Directory to save estimator checkpoints.
        save_every: Save a checkpoint every N steps.
        log_every: Print a log line every N steps.
        device: Torch device.

    Returns:
        final_path: Path to the saved checkpoint.
        vae_estimators: Trained nn.ModuleList of VAEEstimators.
        mlp_estimator: Trained MLPEstimator.
    """
    from src.slot_multi_agent.estimators import VAEEstimator, MLPEstimator

    num_agents = len(student_agents)
    slot_dim = student_agents[0].slot_dim
    num_prototypes = student_agents[0].num_prototypes

    print("=" * 60)
    print("PHASE 2.5: Performance Estimator Training (VAE + MLP)")
    print(f"  Steps: {num_steps}, LR: {lr}, VAE β: {vae_beta}")
    print(f"  Agents: {num_agents}, slot_dim: {slot_dim}, "
          f"prototypes: {num_prototypes}")
    print(f"  Agents sampled per step: {agents_per_step}")
    print("=" * 60)

    # Freeze upstream
    adaslot_model = adaslot_model.to(device).eval()
    student_agents = student_agents.to(device).eval()
    for p in adaslot_model.parameters():
        p.requires_grad_(False)
    for p in student_agents.parameters():
        p.requires_grad_(False)

    # Create estimators
    vae_estimators = nn.ModuleList([
        VAEEstimator(agent_id=i, slot_dim=slot_dim)
        for i in range(num_agents)
    ]).to(device)

    mlp_estimator = MLPEstimator(
        num_agents=num_agents,
        slot_dim=slot_dim,
    ).to(device)

    # Optimizers
    vae_optimizer = torch.optim.Adam(vae_estimators.parameters(), lr=lr)
    mlp_optimizer = torch.optim.Adam(mlp_estimator.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    data_iter = iter(dataloader)
    max_entropy = float(torch.tensor(float(num_prototypes)).log().item())
    t0 = time.time()

    for step in range(num_steps):
        # --- Get batch ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = (batch[0].to(device) if isinstance(batch, (list, tuple))
                  else batch['image'].to(device))

        # --- 1. Extract slots (frozen AdaSlot) ---
        with torch.no_grad():
            adaslot_out = adaslot_model(images)
            slots = adaslot_out['slots']                # (B, S, D)
            keep = adaslot_out['hard_keep_decision']    # (B, S)

        B, S, D = slots.shape
        flat_slots = slots.reshape(B * S, D)
        flat_keep = keep.reshape(B * S)
        active_slots = flat_slots[flat_keep > 0.5]      # (N_active, D)

        if active_slots.shape[0] < 2:
            continue

        # --- 2. Sample agents & compute quality ---
        n_train = min(agents_per_step, num_agents)
        agent_indices = torch.randperm(num_agents)[:n_train]

        total_vae_loss = torch.tensor(0.0, device=device)
        total_mlp_loss = torch.tensor(0.0, device=device)

        for agent_idx in agent_indices:
            agent_idx_int = agent_idx.item()
            agent = student_agents[agent_idx_int]

            # Quality: confidence = 1 − normalised entropy
            with torch.no_grad():
                output = agent(active_slots)               # (N, proto)
                entropy = -(output * torch.log(output + 1e-10)).sum(dim=-1)
                quality = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)

            # --- 3. VAE loss (quality-weighted reconstruction) ---
            vae = vae_estimators[agent_idx_int]
            recon, mu, logvar = vae(active_slots)

            recon_per = F.mse_loss(recon, active_slots, reduction='none').mean(dim=-1)
            weighted_recon = (recon_per * quality).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = weighted_recon + vae_beta * kl
            total_vae_loss = total_vae_loss + vae_loss

            # --- 4. MLP loss (supervised quality regression) ---
            pred_quality = mlp_estimator(active_slots, agent_idx_int)
            mlp_loss = F.mse_loss(pred_quality, quality)
            total_mlp_loss = total_mlp_loss + mlp_loss

        # --- Backprop VAE ---
        total_vae_loss = total_vae_loss / n_train
        vae_optimizer.zero_grad()
        total_vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae_estimators.parameters(), max_norm=1.0)
        vae_optimizer.step()

        # --- Backprop MLP ---
        total_mlp_loss = total_mlp_loss / n_train
        mlp_optimizer.zero_grad()
        total_mlp_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_estimator.parameters(), max_norm=1.0)
        mlp_optimizer.step()

        # --- Logging ---
        if step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  Step {step:>7d} | "
                f"vae_loss={total_vae_loss.item():.4f} | "
                f"mlp_loss={total_mlp_loss.item():.4f} | "
                f"{elapsed:.0f}s"
            )

        # --- Checkpoint ---
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"estimators_step{step + 1}.pth")
            torch.save({
                'step': step + 1,
                'vae_estimators': vae_estimators.state_dict(),
                'mlp_estimator': mlp_estimator.state_dict(),
            }, ckpt_path)
            print(f"  [SAVED] {ckpt_path}")

    # Save final
    final_path = os.path.join(save_dir, "estimators_final.pth")
    torch.save({
        'vae_estimators': vae_estimators.state_dict(),
        'mlp_estimator': mlp_estimator.state_dict(),
    }, final_path)
    print(f"  [DONE] Phase 2.5 complete. Final: {final_path}")
    return final_path, vae_estimators, mlp_estimator


# ───────────────────────────────────────────────────────
#  Phase 3: Filter → UCB Weighted MoE → CRP
# ───────────────────────────────────────────────────────

def _estimate_agent_scores(
    slot: torch.Tensor,
    vae_estimators: nn.ModuleList,
    mlp_estimator,
    num_agents: int,
    vae_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute hybrid (VAE + MLP) performance score for every agent on a slot.

    Args:
        slot: (slot_dim,) single slot vector.
        vae_estimators: Per-agent VAE estimators.
        mlp_estimator: Shared MLP estimator.
        num_agents: Total agents.
        vae_weight: Weight for VAE score in the combination.

    Returns:
        scores: (num_agents,) tensor – higher is better.
    """
    scores = torch.zeros(num_agents, device=slot.device)
    for i in range(num_agents):
        vae_score = vae_estimators[i].estimate_performance(slot)
        mlp_score = mlp_estimator.estimate_performance(slot, i)
        combined = vae_weight * vae_score + (1 - vae_weight) * mlp_score
        scores[i] = combined
    return scores


def _extract_weighted_features_one_sample(
    slots: torch.Tensor,
    student_agents: nn.ModuleList,
    vae_estimators,
    mlp_estimator,
    ucb_moe,
    filter_k: int,
    num_agents: int,
    use_estimator_pipeline: bool,
) -> torch.Tensor:
    """
    Extract the feature vector for ONE sample using the full pipeline.

    For each slot:
      - If estimators available: score all agents → top-filter_k →
        UCB weights → weighted sum of agent outputs.
      - Otherwise: fall back to first-k agents (concat).

    Args:
        slots: (S, D) slots for this sample.
        student_agents: Agent pool.
        vae_estimators: VAE estimators (or None).
        mlp_estimator: MLP estimator (or None).
        ucb_moe: UCBWeightedMoE instance (or None).
        filter_k: How many agents to keep after filtering.
        num_agents: Total agents.
        use_estimator_pipeline: Whether to use estimate → UCB → weighted.

    Returns:
        feature: (S * proto_dim,)  if use_estimator_pipeline  (weighted MoE)
            or   (S * k * proto_dim,) in legacy mode.
    """
    S, D = slots.shape
    sample_hidden = []

    for s_idx in range(S):
        slot = slots[s_idx]  # (D,)

        if use_estimator_pipeline:
            # ── Step 1: Fast filter via VAE + MLP ──
            scores = _estimate_agent_scores(
                slot, vae_estimators, mlp_estimator, num_agents
            )
            _, topk_ids = torch.topk(scores, k=min(filter_k, num_agents))
            filtered_ids = topk_ids.cpu().tolist()

            # ── Step 2: UCB committee weights ──
            _, weights = ucb_moe.get_weights(filtered_ids)
            weights_t = torch.tensor(
                weights, dtype=torch.float32, device=slot.device
            )

            # ── Step 3: Weighted sum of agent outputs (MoE gate) ──
            weighted_output = torch.zeros(
                student_agents[0].num_prototypes, device=slot.device
            )
            for w, aid in zip(weights_t, filtered_ids):
                agent_out = student_agents[aid](slot.unsqueeze(0)).squeeze(0)
                weighted_output += w * agent_out

            sample_hidden.append(weighted_output)  # (proto_dim,)

        else:
            # ── Legacy: first-k agents, concat ──
            slot_labels = []
            for a_idx in range(min(filter_k, num_agents)):
                label = student_agents[a_idx](
                    slot.unsqueeze(0)
                ).squeeze(0)
                slot_labels.append(label)
            sample_hidden.append(torch.cat(slot_labels))  # (k * proto_dim,)

    return torch.cat(sample_hidden)  # concat across all slots


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
    aggregator=None,  # Pass existing aggregator to continue training across tasks
    # ── NEW: estimator + UCB MoE pipeline ──
    vae_estimators=None,
    mlp_estimator=None,
    ucb_moe=None,
    filter_k: int = 10,
):
    """
    Phase 3: Incrementally fit the CRP aggregator on agent hidden labels.

    If *vae_estimators*, *mlp_estimator*, and *ucb_moe* are provided the
    full pipeline is used:

        slot → VAE/MLP scores → top-filter_k agents
             → UCB weighted MoE (all filter_k members)
             → weighted output per slot
             → concat across slots → CRP aggregator → prediction

    After CRP predicts, the binary reward (correct / incorrect) is
    distributed back to UCB so that it can refine agent weights online.

    When estimators are not available (backward-compatible), falls back
    to the legacy first-k-agents concatenation.

    Args:
        adaslot_model: Frozen AdaSlot.
        student_agents: Frozen student agents.
        dataloader: Training data for the current task.
        num_classes: Maximum number of classes.
        aggregator_type: "crp" (default).
        aggregate_mode: "concat" (default, used only in legacy mode).
        k: Number of agents in legacy first-k mode.
        device: Torch device.
        log_every: Logging frequency.
        aggregator: Existing aggregator to reuse across tasks.
        vae_estimators: Trained VAE estimators (from Phase 2.5).
        mlp_estimator: Trained MLP estimator (from Phase 2.5).
        ucb_moe: UCBWeightedMoE instance (persisted across tasks).
        filter_k: Number of agents to keep after fast filtering.

    Returns:
        aggregator: Updated aggregator (BatchCRPAggregator).
    """
    from src.slot_multi_agent.aggregator import create_aggregator
    import numpy as np

    use_estimator_pipeline = (
        vae_estimators is not None
        and mlp_estimator is not None
        and ucb_moe is not None
    )

    pipeline_label = "Filter→UCB→CRP" if use_estimator_pipeline else "Legacy (first-k)"
    print("=" * 60)
    print(f"PHASE 3: Incremental Aggregator  [{pipeline_label}]")
    print(f"  Classes: {num_classes}, Aggregator: {aggregator_type}")
    if use_estimator_pipeline:
        print(f"  Filter K: {filter_k}, UCB burn-in: {ucb_moe.burn_in}")
    else:
        print(f"  Legacy k={k}, Mode: {aggregate_mode}")
    print("=" * 60)

    adaslot_model = adaslot_model.to(device).eval()
    student_agents = student_agents.to(device).eval()
    for p in adaslot_model.parameters():
        p.requires_grad_(False)
    for p in student_agents.parameters():
        p.requires_grad_(False)

    if vae_estimators is not None:
        vae_estimators = vae_estimators.to(device).eval()
    if mlp_estimator is not None:
        mlp_estimator = mlp_estimator.to(device).eval()

    num_agents = len(student_agents)
    num_slots = adaslot_model.num_slots
    slot_dim = adaslot_model.slot_dim
    proto_dim = student_agents[0].num_prototypes

    # Compute aggregator input dimension
    if use_estimator_pipeline:
        # Weighted MoE → single proto_dim output per slot → concat S slots
        input_dim = num_slots * proto_dim
    else:
        if aggregate_mode == "concat":
            input_dim = num_slots * k * proto_dim
        else:
            input_dim = k * proto_dim

    # Prepare kwargs based on aggregator type
    agg_kwargs = {
        "num_classes": num_classes,
        "device": device,
    }
    if aggregator_type == "crp":
        agg_kwargs["feature_dim"] = input_dim
    else:
        agg_kwargs["input_dim"] = input_dim

    if aggregator is None:
        aggregator = create_aggregator(
            aggregator_type=aggregator_type,
            **agg_kwargs
        )
    # else: reuse existing aggregator — continual learning continues from previous task

    total_correct = 0
    total_samples = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = (batch[0].to(device) if isinstance(batch, (list, tuple))
                  else batch['image'].to(device))
        targets = batch[1] if isinstance(batch, (list, tuple)) else batch['label']

        with torch.no_grad():
            # 1. Slots
            adaslot_out = adaslot_model(images)
            slots = adaslot_out['slots']              # (B, S, D)

            B, S, D = slots.shape

            # 2. Feature extraction (full pipeline or legacy)
            all_features = []
            all_committee_info = []  # for UCB reward update

            for b in range(B):
                feat = _extract_weighted_features_one_sample(
                    slots=slots[b],
                    student_agents=student_agents,
                    vae_estimators=vae_estimators,
                    mlp_estimator=mlp_estimator,
                    ucb_moe=ucb_moe,
                    filter_k=filter_k if use_estimator_pipeline else k,
                    num_agents=num_agents,
                    use_estimator_pipeline=use_estimator_pipeline,
                )
                all_features.append(feat)

                # Store committee info for UCB update (only if pipeline)
                if use_estimator_pipeline:
                    # Collect the committees for every slot of this sample
                    slot_committees = []
                    for s_idx in range(S):
                        slot = slots[b, s_idx]
                        scores = _estimate_agent_scores(
                            slot, vae_estimators, mlp_estimator, num_agents
                        )
                        _, topk_ids = torch.topk(
                            scores, k=min(filter_k, num_agents)
                        )
                        fids = topk_ids.cpu().tolist()
                        _, ws = ucb_moe.get_weights(fids)
                        slot_committees.append((fids, ws))
                    all_committee_info.append(slot_committees)

            features_t = torch.stack(all_features)  # (B, dim)

        # 3. Aggregator: predict, then learn
        try:
            if hasattr(aggregator, 'predict_batch'):
                preds = aggregator.predict_batch(features_t)
                correct = sum(
                    1 for p, t in zip(preds, targets.cpu().numpy()) if p == t
                )
            else:
                preds = aggregator.predict(features_t.numpy())
                correct = (preds == targets.cpu().numpy()).sum()
            total_correct += correct

            # ── UCB reward update ──
            if use_estimator_pipeline and preds is not None:
                for b_idx, (p, t) in enumerate(
                    zip(preds, targets.cpu().numpy())
                ):
                    reward = 1.0 if p == t else 0.0
                    for fids, ws in all_committee_info[b_idx]:
                        ucb_moe.update_batch(fids, ws, reward)

        except Exception:
            pass

        total_samples += B

        if hasattr(aggregator, 'learn_batch'):
            aggregator.learn_batch(features_t, targets.cpu())
        else:
            aggregator.partial_fit(features_t.numpy(), targets.cpu().numpy())

        if (batch_idx + 1) % log_every == 0:
            acc = total_correct / max(total_samples, 1) * 100
            elapsed = time.time() - t0
            extra = ""
            if use_estimator_pipeline:
                extra = f" | ucb_rounds={ucb_moe.total_count}"
            print(
                f"  Batch {batch_idx + 1:>5d} | "
                f"samples={total_samples} | "
                f"acc={acc:.2f}%{extra} | "
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
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "2.5", "3", "all"],
                        help="Training phase: 1=AdaSlot, 2=Agents, "
                             "2.5=Estimators, 3=CRP, all=sequential")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory")
    parser.add_argument("--adaslot_ckpt", type=str, default=None,
                        help="AdaSlot checkpoint path")
    parser.add_argument("--agent_ckpt", type=str, default=None,
                        help="Agent checkpoint path")
    parser.add_argument("--estimator_ckpt", type=str, default=None,
                        help="Estimator checkpoint path (Phase 2.5 output)")
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
    parser.add_argument("--p1_steps", type=int, default=500000,
                        help="Phase 1 training steps")
    parser.add_argument("--p1_lr", type=float, default=4e-4)
    parser.add_argument("--p2_steps", type=int, default=100000,
                        help="Phase 2 training steps")
    parser.add_argument("--p2_lr", type=float, default=1e-3)
    parser.add_argument("--p2b_steps", type=int, default=20000,
                        help="Phase 2.5 estimator training steps")
    parser.add_argument("--p2b_lr", type=float, default=1e-3)
    parser.add_argument("--filter_k", type=int, default=10,
                        help="Number of agents to keep after fast filtering")
    parser.add_argument("--ucb_exploration", type=float, default=1.414,
                        help="UCB exploration constant (default sqrt(2))")
    parser.add_argument("--ucb_burn_in", type=int, default=100,
                        help="UCB burn-in rounds with uniform weights")
    parser.add_argument("--pretrained", type=str, default="CLEVR10",
                        choices=["CLEVR10", "COCO", "MOVi-C", "MOVi-E", "none"],
                        help="Pretrained AdaSlot checkpoint to load "
                             "(default: CLEVR10). Use 'none' for scratch.")

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
        ckpt = torch.load(args.adaslot_ckpt, map_location='cpu',
                          weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        result = adaslot.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded AdaSlot weights from {args.adaslot_ckpt}")
        print(f"       Missing: {len(result.missing_keys)}, "
              f"Unexpected: {len(result.unexpected_keys)}")
    elif args.pretrained != "none":
        pretrained_dir = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            )),
            "checkpoints", "slot_attention", "adaslot_real", "AdaSlotCkpt"
        )
        ckpt_path = os.path.join(pretrained_dir, f"{args.pretrained}.ckpt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu',
                              weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model', ckpt))
            result = adaslot.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded pretrained AdaSlot ({args.pretrained}) "
                  f"from {ckpt_path}")
            print(f"       Missing: {len(result.missing_keys)}, "
                  f"Unexpected: {len(result.unexpected_keys)}")
        else:
            print(f"[WARN] Pretrained checkpoint not found: {ckpt_path}")
            print("       Download from: https://drive.google.com/drive/"
                  "folders/1SRKE9Q5XF2UeYj1XB8kyjxORDmB7c7Mz")
            print("       Training from scratch instead.")

    # ── Build dataloader ──
    train_loaders = None    # will be set when using CIFAR-100
    test_loaders = None
    class_order = None

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
        print("[INFO] No data_dir specified, downloading and using "
              "CIFAR-100 continual benchmark.")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "continual_cifar100",
            "src/data/continual_cifar100.py"
        )
        data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        get_loaders = data_module.get_continual_cifar100_loaders
        _n_tasks = (int(100 / args.num_classes)
                    if args.num_classes > 0 else 5)
        train_loaders, test_loaders, class_order = get_loaders(
            n_tasks=_n_tasks,
            batch_size=args.batch_size,
            num_workers=0,
            seed=42,
            resolution=args.resolution
        )
        # Phase 1, 2, 2.5 use ALL tasks combined for richer training
        from torch.utils.data import ConcatDataset
        _combined = ConcatDataset(
            [loader.dataset for loader in train_loaders]
        )
        dataloader = DataLoader(
            _combined, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )

    # ── Shared objects that persist across phases ──
    student_agents = None
    teacher_agents = None
    vae_estimators = None
    mlp_estimator = None

    # ── Execute phases ──
    phases = (["1", "2", "2.5", "3"] if args.phase == "all"
              else [args.phase])

    for phase in phases:
        # ==============================================================
        #  PHASE 1: AdaSlot Pretraining
        # ==============================================================
        if phase == "1":
            train_phase1_adaslot(
                model=adaslot,
                dataloader=dataloader,
                num_steps=args.p1_steps,
                lr=args.p1_lr,
                device=args.device,
            )

        # ==============================================================
        #  PHASE 2: Agent DINO Training
        # ==============================================================
        elif phase == "2":
            if args.adaslot_ckpt:
                ckpt = torch.load(args.adaslot_ckpt, map_location='cpu',
                                  weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                adaslot.load_state_dict(state, strict=False)

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

        # ==============================================================
        #  PHASE 2.5: Estimator Training (VAE + MLP)
        # ==============================================================
        elif phase == "2.5":
            # Ensure AdaSlot is loaded
            if args.adaslot_ckpt:
                ckpt = torch.load(args.adaslot_ckpt, map_location='cpu',
                                  weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                adaslot.load_state_dict(state, strict=False)

            # Ensure agents are loaded
            if student_agents is None:
                from src.slot_multi_agent.atomic_agent import create_agent_pool
                student_agents, _ = create_agent_pool(
                    num_agents=args.num_agents,
                    slot_dim=args.slot_dim,
                    num_prototypes=args.num_prototypes,
                    device=args.device,
                )
            if args.agent_ckpt:
                ckpt = torch.load(args.agent_ckpt, map_location='cpu',
                                  weights_only=False)
                student_agents.load_state_dict(ckpt['student_agents'])

            _, vae_estimators, mlp_estimator = train_phase2b_estimators(
                adaslot_model=adaslot,
                student_agents=student_agents,
                dataloader=dataloader,
                num_steps=args.p2b_steps,
                lr=args.p2b_lr,
                device=args.device,
            )

        # ==============================================================
        #  PHASE 3: Filter → UCB Weighted MoE → CRP
        # ==============================================================
        elif phase == "3":
            # ── Load AdaSlot ──
            if args.adaslot_ckpt:
                ckpt = torch.load(args.adaslot_ckpt, map_location='cpu',
                                  weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                adaslot.load_state_dict(state, strict=False)

            # ── Load Agents ──
            if student_agents is None:
                from src.slot_multi_agent.atomic_agent import create_agent_pool
                student_agents, _ = create_agent_pool(
                    num_agents=args.num_agents,
                    slot_dim=args.slot_dim,
                    num_prototypes=args.num_prototypes,
                    device=args.device,
                )
            if args.agent_ckpt:
                ckpt = torch.load(args.agent_ckpt, map_location='cpu',
                                  weights_only=False)
                student_agents.load_state_dict(ckpt['student_agents'])

            # ── Load Estimators (if available) ──
            if vae_estimators is None and args.estimator_ckpt:
                from src.slot_multi_agent.estimators import (
                    VAEEstimator, MLPEstimator
                )
                vae_estimators = nn.ModuleList([
                    VAEEstimator(agent_id=i, slot_dim=args.slot_dim)
                    for i in range(args.num_agents)
                ])
                mlp_estimator = MLPEstimator(
                    num_agents=args.num_agents,
                    slot_dim=args.slot_dim,
                )
                est_ckpt = torch.load(args.estimator_ckpt,
                                      map_location='cpu', weights_only=False)
                vae_estimators.load_state_dict(est_ckpt['vae_estimators'])
                mlp_estimator.load_state_dict(est_ckpt['mlp_estimator'])
                print(f"[INFO] Loaded estimators from "
                      f"{args.estimator_ckpt}")

            # ── Create UCB Weighted MoE (if estimators available) ──
            ucb_moe = None
            if vae_estimators is not None and mlp_estimator is not None:
                from src.slot_multi_agent.bandit_selector import (
                    UCBWeightedMoE
                )
                ucb_moe = UCBWeightedMoE(
                    num_agents=args.num_agents,
                    exploration_constant=args.ucb_exploration,
                    burn_in=args.ucb_burn_in,
                )
                print(f"[INFO] UCB Weighted MoE enabled "
                      f"(filter_k={args.filter_k}, "
                      f"c={args.ucb_exploration}, "
                      f"burn_in={args.ucb_burn_in})")
            else:
                print("[INFO] No estimators found → legacy first-k mode")

            # ── Task loaders ──
            if train_loaders is not None:
                _task_train_loaders = train_loaders
                _task_test_loaders = test_loaders
                _task_class_order = class_order
            else:
                _task_train_loaders = [dataloader]
                _task_test_loaders = [dataloader]
                _task_class_order = [list(range(args.num_classes))]

            # ── Continual task loop ──
            aggregator = None
            for task_id, task_loader in enumerate(_task_train_loaders):
                print(f"\n{'=' * 60}")
                print(f"PHASE 3 — TASK {task_id + 1}/"
                      f"{len(_task_train_loaders)}")
                if (isinstance(_task_class_order, list)
                        and task_id < len(_task_class_order)):
                    print(f"  Classes: {_task_class_order[task_id]}")
                print(f"{'=' * 60}")

                aggregator = train_phase3_crp(
                    adaslot_model=adaslot,
                    student_agents=student_agents,
                    dataloader=task_loader,
                    num_classes=args.num_classes,
                    aggregator_type="crp",
                    device=args.device,
                    aggregator=aggregator,
                    # Full pipeline (None ⇒ legacy fallback)
                    vae_estimators=vae_estimators,
                    mlp_estimator=mlp_estimator,
                    ucb_moe=ucb_moe,
                    filter_k=args.filter_k,
                )

                # ── Evaluate on all classes seen so far ──
                _eval_after_task(
                    task_id=task_id,
                    adaslot=adaslot,
                    student_agents=student_agents,
                    aggregator=aggregator,
                    test_loader=_task_test_loaders[task_id],
                    device=args.device,
                    vae_estimators=vae_estimators,
                    mlp_estimator=mlp_estimator,
                    ucb_moe=ucb_moe,
                    filter_k=args.filter_k,
                )

            # ── Save UCB state ──
            if ucb_moe is not None:
                ucb_path = os.path.join("checkpoints", "ucb_moe_state.npz")
                os.makedirs(os.path.dirname(ucb_path), exist_ok=True)
                ucb_moe.save(ucb_path)
                print(f"  [SAVED] UCB state → {ucb_path}")
                print(f"  [UCB STATS] {ucb_moe.get_stats()['total_count']}"
                      f" total rounds, "
                      f"{len(ucb_moe.get_stats()['per_agent'])} agents used")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)


# ───────────────────────────────────────────────────────
#  Cumulative Evaluation Helper
# ───────────────────────────────────────────────────────

def _eval_after_task(
    task_id: int,
    adaslot,
    student_agents,
    aggregator,
    test_loader,
    device: str,
    vae_estimators=None,
    mlp_estimator=None,
    ucb_moe=None,
    filter_k: int = 10,
):
    """
    Evaluate cumulative accuracy after a task using the same pipeline
    as training (estimator → UCB → CRP or legacy first-k).
    """
    use_pipeline = (
        vae_estimators is not None
        and mlp_estimator is not None
        and ucb_moe is not None
    )

    adaslot = adaslot.to(device).eval()
    student_agents = student_agents.to(device).eval()
    if vae_estimators is not None:
        vae_estimators = vae_estimators.to(device).eval()
    if mlp_estimator is not None:
        mlp_estimator = mlp_estimator.to(device).eval()

    num_agents = len(student_agents)
    total_correct = 0
    total_samples = 0

    for eval_batch in test_loader:
        eval_images = eval_batch[0].to(device)
        eval_targets = eval_batch[1]

        with torch.no_grad():
            adaslot_out = adaslot(eval_images)
            eval_slots = adaslot_out['slots']   # (B, S, D)
            B, S, D = eval_slots.shape

            all_feats = []
            for b in range(B):
                feat = _extract_weighted_features_one_sample(
                    slots=eval_slots[b],
                    student_agents=student_agents,
                    vae_estimators=vae_estimators,
                    mlp_estimator=mlp_estimator,
                    ucb_moe=ucb_moe,
                    filter_k=filter_k if use_pipeline else 3,
                    num_agents=num_agents,
                    use_estimator_pipeline=use_pipeline,
                )
                all_feats.append(feat)
            feats_t = torch.stack(all_feats)

        try:
            preds = aggregator.predict_batch(feats_t)
            total_correct += sum(
                1 for p, t in zip(preds, eval_targets.numpy()) if p == t
            )
        except Exception:
            pass
        total_samples += B

    cum_acc = total_correct / max(total_samples, 1) * 100
    pipeline_tag = "Filter→UCB→CRP" if use_pipeline else "legacy"
    print(f"  [EVAL] Cumulative accuracy after task {task_id} "
          f"({pipeline_tag}): {cum_acc:.2f}%  "
          f"({total_correct}/{total_samples})")


if __name__ == "__main__":
    main()
