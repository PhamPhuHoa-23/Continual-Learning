"""
Agent Phase A Trainer  –  Warm-up
==================================
Trains agents using only the agent reconstruction loss.

    L = γ · L_agent

- SlotAttention backbone:  **frozen**
- SlotVAE routers:         **frozen**
- Routing:                 **hard** (argmax over VAE scores) by default
- Agents:                  **trainable**

Each slot is assigned to exactly one agent; that agent's
``reconstruction_loss`` is backpropagated.

Usage
-----
    from cont_src.training.configs       import PhaseAConfig
    from cont_src.training.agent_phase_a import AgentPhaseATrainer

    cfg     = PhaseAConfig(max_steps=50, gamma=1.0)
    trainer = AgentPhaseATrainer(
        cfg,
        slot_model = backbone,
        vaes       = vae_list,
        agents     = agent_list,
    )
    trainer.train(dataloader)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent
from cont_src.models.routers.slot_vae import SlotVAE
from cont_src.training.base_trainer import BaseTrainer
from cont_src.training.configs import PhaseAConfig

logger = logging.getLogger(__name__)


class AgentPhaseATrainer(BaseTrainer):
    """
    Phase A: agent warm-up with reconstruction loss only.

    Parameters
    ----------
    config : PhaseAConfig
    slot_model : nn.Module
        Frozen backbone that returns ``{"slots": (B, K, D), ...}``.
    vaes : List[SlotVAE]
        One VAE per agent cluster (frozen, used for routing).
    agents : List[ResidualMLPAgent]
        Agents to train.
    """

    def __init__(
        self,
        config: PhaseAConfig,
        slot_model: nn.Module,
        vaes: List[SlotVAE],
        agents: List[ResidualMLPAgent],
    ):
        # Register trainable components so BaseTrainer's optimiser can find them
        self.model_components = {f"agent_{i}": a for i, a in enumerate(agents)}
        # Register backbone as non-trainable reference (won't be in optimizer
        # but will be moved to device)
        self.model_components["slot_model"] = slot_model

        super().__init__(config)
        self.config: PhaseAConfig  # type narrowing

        self.slot_model = slot_model
        self.vaes = vaes
        self.agents = agents

        # Freeze backbone
        self.freeze("slot_model")

        logger.info(
            f"[AgentPhaseATrainer] {len(agents)} agents  "
            f"routing={config.routing_mode}  γ={config.gamma}"
        )

    # ------------------------------------------------------------------
    # Optimiser: only agent parameters
    # ------------------------------------------------------------------

    def setup_optimizer(self) -> None:
        params = []
        for agent in self.agents:
            params += [p for p in agent.parameters() if p.requires_grad]

        if not params:
            logger.warning(
                "[AgentPhaseATrainer] No trainable agent parameters.")
            self.optimizer = None
            return

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        cfg = self.config
        dur = f"{cfg.max_steps} steps" if cfg.max_steps > 0 else f"{cfg.max_epochs} epochs"
        logger.info(f"[AgentPhaseATrainer] Phase A warm-up — {dur}")

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_slots(self, images: torch.Tensor):
        """Forward backbone, return (slots (B,K,D), n_active_slots float)."""
        self.slot_model.eval()
        out = self.slot_model(images)
        slots = out["slots"]
        mask = out.get("hard_keep_decision", out.get("mask"))
        n_active = (mask > 0.5).float().sum(
            dim=-1).mean().item() if mask is not None else float(slots.shape[1])
        return slots, n_active

    def _route(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Assign each slot to an agent.

        Parameters
        ----------
        slots : (B, K, D)

        Returns
        -------
        assignments : (B, K) LongTensor  — agent index for each slot
        """
        B, K, D = slots.shape
        n_agents = len(self.vaes)

        if n_agents == 0:
            # No VAEs available — assign all slots to a single dummy agent 0
            return torch.zeros(B, K, dtype=torch.long, device=slots.device)

        if n_agents == 1 or self.config.routing_mode == "single":
            return torch.zeros(B, K, dtype=torch.long, device=slots.device)

        # vae.score() expects (N, D_s) not (B, K, D) — flatten then reshape back
        slots_flat = slots.reshape(B * K, D)   # (B*K, D)
        scores = torch.stack(
            [vae.score(slots_flat) for vae in self.vaes],
            dim=-1,
        )  # (B*K, n_agents)
        scores = scores.reshape(B, K, n_agents)  # (B, K, n_agents)

        if self.config.routing_mode == "hard":
            return scores.argmax(dim=-1)                   # (B, K)

        # soft → still pick argmax for assignment; grad needed only in phase B
        return scores.argmax(dim=-1)

    def train_step(self, batch: Any) -> Dict[str, float]:
        images, _ = self._unpack(batch)
        images = images.to(self.device)

        # 1. Extract slots (no grad needed for backbone)
        slots, n_active_slots = self._extract_slots(
            images)   # (B, K, D), float

        # 2. Route: hard argmax
        with torch.no_grad():
            assignments = self._route(slots)         # (B, K)

        # 3. Compute L_agent per agent, accumulate
        total_loss = torch.tensor(0.0, device=self.device)
        per_agent_losses: Dict[str, float] = {}
        active_agents = 0

        for m, agent in enumerate(self.agents):
            agent.train()
            # Gather slots assigned to this agent across all samples
            mask_m = (assignments == m)              # (B, K) bool
            if not mask_m.any():
                continue

            # (N_assigned, D)
            slots_m = slots[mask_m]
            l_m = agent.reconstruction_loss(slots_m.unsqueeze(0))
            # unsqueeze(0) gives shape (1, N_assigned, D) matching
            # ResidualMLPAgent.forward which expects (B, K, D)
            # but reconstruction_loss works on any batch shape — re-check:
            # Actually ResidualMLPAgent.reconstruction_loss(slots) where
            # slots is (B, K, D) or we can just pass (1, N, D)
            l_m = agent.reconstruction_loss(slots_m.unsqueeze(0))

            total_loss = total_loss + self.config.gamma * l_m
            per_agent_losses[f"l_agent_{m}"] = l_m.detach().item()
            active_agents += 1

        metrics = {
            "loss_total":     total_loss,
            "l_agent_mean":   total_loss.detach().item() / max(active_agents, 1),
            "active_agents":  float(active_agents),
            "n_active_slots": n_active_slots,
            **per_agent_losses,
        }
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack(batch):
        if isinstance(batch, (list, tuple)):
            return batch[0], (batch[1] if len(batch) > 1 else None)
        if isinstance(batch, dict):
            return batch["image"], batch.get("label")
        return batch, None
