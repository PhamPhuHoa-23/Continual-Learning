"""
Agent Phase B Trainer  –  Full Training with Label Losses
==========================================================
Full agent training with soft routing and label-supervised losses:

    L = γ·L_agent  +  α·L_prim  +  β·L_SupCon

Key differences from Phase A
------------------------------
- Routing uses **temperature-annealed softmax** → gradients flow to agents
  through the soft mixture weights.
- Labels are required (batches without labels are skipped for L_prim / L_SupCon).
- Temperature is annealed from ``init_temperature`` to ``final_temperature``
  over the course of training (cosine or linear schedule).
- Hard argmax is still used at inference (``model.eval()`` mode).
- Aggregated hidden H = Σ_m w_m · h_m feeds into primitive loss and SupCon.

Frozen / trainable
------------------
- SlotAttention backbone:    **frozen**
- SlotVAE routers:           **frozen** by default (``freeze_routers=True``)
- Agents:                    **trainable**
- AttentionAggregator:       **trainable** (if provided)

Usage
-----
    from cont_src.training.configs       import PhaseBConfig
    from cont_src.training.agent_phase_b import AgentPhaseBTrainer

    cfg     = PhaseBConfig(max_steps=200, gamma=1.0, alpha=0.3, beta=0.3)
    trainer = AgentPhaseBTrainer(
        cfg,
        slot_model  = backbone,
        vaes        = vae_list,
        agents      = agent_list,
        aggregator  = attention_agg,
    )
    trainer.train(dataloader)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent
from cont_src.models.routers.slot_vae import SlotVAE
from cont_src.training.base_trainer import BaseTrainer
from cont_src.training.configs import PhaseBConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive loss (inline, avoids coupling to losses.py for portability)
# ---------------------------------------------------------------------------

def _primitive_loss(hidden: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """Matrix-level KL  d^y || d^H  (same formula as PrimitiveLoss in losses.py)."""
    h = F.normalize(hidden, p=2, dim=-1)
    sim = torch.mm(h, h.t()) * tau          # (B, B)
    d_H = F.softmax(sim, dim=1)

    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    d_y  = same / (same.sum(dim=1, keepdim=True) + 1e-8)

    kl = (d_y * torch.log((d_y + 1e-8) / (d_H + 1e-8))).sum() / hidden.shape[0]
    return kl


# ---------------------------------------------------------------------------
# Supervised contrastive loss (inline)
# ---------------------------------------------------------------------------

def _supcon_loss(hidden: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """Khosla et al. SupCon loss on L2-normalised features."""
    B = hidden.shape[0]
    if B < 2:
        return hidden.new_zeros(()).squeeze()

    h  = F.normalize(hidden, p=2, dim=-1)          # (B, D)
    sim = torch.mm(h, h.t()) / tau                  # (B, B)

    # Mask out diagonal (self)
    diag_mask = torch.eye(B, dtype=torch.bool, device=h.device)
    sim = sim.masked_fill(diag_mask, float("-inf"))

    # Rows where every element is -inf (shouldn't happen for B>=2, but be safe)
    row_max = sim.max(dim=1).values
    valid_rows = torch.isfinite(row_max)            # (B,)
    if not valid_rows.any():
        return hidden.new_zeros(()).squeeze()

    log_prob = torch.zeros_like(sim)
    log_prob[valid_rows] = (
        sim[valid_rows]
        - torch.logsumexp(sim[valid_rows], dim=1, keepdim=True)
    )

    # Positive pairs (same class, excluding self)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag_mask
    pos_mask = pos_mask.float()
    n_pos = pos_mask.sum(dim=1).clamp(min=1)

    loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
    return loss[valid_rows].mean()


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def _compute_temperature(step: int, total_steps: int, cfg: PhaseBConfig) -> float:
    """Interpolate temperature from init → final."""
    if cfg.temp_anneal == "constant" or total_steps <= 0:
        return cfg.init_temperature

    ratio = min(step / total_steps, 1.0)

    if cfg.temp_anneal == "linear":
        return cfg.init_temperature + ratio * (cfg.final_temperature - cfg.init_temperature)

    # cosine
    cos = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.final_temperature + cos * (cfg.init_temperature - cfg.final_temperature)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AgentPhaseBTrainer(BaseTrainer):
    """
    Phase B: full agent training with soft routing + label losses.

    Parameters
    ----------
    config : PhaseBConfig
    slot_model : nn.Module
        Frozen backbone → ``{"slots": (B, K, D), ...}``.
    vaes : List[SlotVAE]
        Routers (frozen by default).
    agents : List[ResidualMLPAgent]
        Trainable agents.
    aggregator : nn.Module, optional
        Maps (B, K, D) → (B, D_h) for L_prim / L_SupCon.
        If None, a simple mean across the K dimension is used.
    """

    def __init__(
        self,
        config: PhaseBConfig,
        slot_model: nn.Module,
        vaes: List[SlotVAE],
        agents: List[ResidualMLPAgent],
        aggregator: Optional[nn.Module] = None,
    ):
        self.model_components = {f"agent_{i}": a for i, a in enumerate(agents)}
        self.model_components["slot_model"] = slot_model
        if aggregator is not None:
            self.model_components["aggregator"] = aggregator

        super().__init__(config)
        self.config: PhaseBConfig  # type narrowing

        self.slot_model = slot_model
        self.vaes       = vaes
        self.agents     = agents
        self.aggregator = aggregator

        # Freeze backbone always
        self.freeze("slot_model")

        # Optionally freeze VAE routers (they have no .parameters(),
        # but this guards future changes)
        # SlotVAE is not nn.Module; freeze is a no-op here — fine.

        # Total steps for temperature schedule
        self._total_steps = config.max_steps if config.max_steps > 0 else int(1e6)

        logger.info(
            f"[AgentPhaseBTrainer] {len(agents)} agents  "
            f"γ={config.gamma} α={config.alpha} β={config.beta}  "
            f"T: {config.init_temperature}→{config.final_temperature} "
            f"({config.temp_anneal})"
        )

    # ------------------------------------------------------------------
    # Optimiser: agents + optional aggregator only
    # ------------------------------------------------------------------

    def setup_optimizer(self) -> None:
        params = []
        for agent in self.agents:
            params += [p for p in agent.parameters() if p.requires_grad]
        if self.aggregator is not None:
            params += [p for p in self.aggregator.parameters() if p.requires_grad]

        if not params:
            logger.warning("[AgentPhaseBTrainer] No trainable parameters.")
            self.optimizer = None
            return

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        dur = (
            f"{self.config.max_steps} steps"
            if self.config.max_steps > 0
            else f"{self.config.max_epochs} epochs"
        )
        logger.info(f"[AgentPhaseBTrainer] Phase B full training — {dur}")
        # Register agent IDs with AttentionAggregator (creates per-agent keys)
        if self.aggregator is not None and hasattr(self.aggregator, "register_agent"):
            for i in range(len(self.agents)):
                self.aggregator.register_agent(i)

    def on_after_step(self, step: int, metrics: Dict) -> None:
        """Log current temperature every log interval."""
        metrics["temperature"] = _compute_temperature(
            step, self._total_steps, self.config
        )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_slots(self, images: torch.Tensor):
        """Return (slots (B,K,D), n_active_slots float)."""
        self.slot_model.eval()
        out   = self.slot_model(images)
        slots = out["slots"]
        mask  = out.get("hard_keep_decision", out.get("mask"))
        n_active = (mask > 0.5).float().sum(dim=-1).mean().item() if mask is not None else float(slots.shape[1])
        return slots, n_active

    def _soft_route(
        self, slots: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Soft routing weights via temperature softmax over VAE scores.

        Parameters
        ----------
        slots : (B, K, D)
        temperature : float

        Returns
        -------
        weights : (B, K, n_agents)   — soft assignment weights
        """
        n = len(self.vaes)
        if n == 0:
            return torch.ones(*slots.shape[:2], 1, device=slots.device)
        if n == 1:
            return torch.ones(*slots.shape[:2], 1, device=slots.device)

        # vae.score() expects (N, D_s) not (B, K, D) — flatten then reshape back
        B, K, D = slots.shape
        slots_flat = slots.reshape(B * K, D)              # (B*K, D)
        scores = torch.stack(
            [vae.score(slots_flat) for vae in self.vaes], dim=-1
        )  # (B*K, n_agents)
        scores = scores.reshape(B, K, n)                  # (B, K, n_agents)
        return F.softmax(scores / temperature, dim=-1)

    def _compute_hidden(
        self, slots: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-slot hidden representations as a soft mixture of agents.

        h_k = Σ_m w_{k,m} · agent_m(slot_k)

        Parameters
        ----------
        slots   : (B, K, D)
        weights : (B, K, n_agents)

        Returns
        -------
        hidden_slots : (B, K, D_h)
        l_agent      : scalar reconstruction loss
        """
        B, K, D = slots.shape

        # Get hidden from every agent: list of (B, K, D_h)
        agent_hiddens = []
        l_agent_total = torch.tensor(0.0, device=slots.device)

        for m, agent in enumerate(self.agents):
            agent.train()
            out = agent(slots)                          # {"hidden": ..., "reconstructed": ...}
            h_m = out["hidden"]                         # (B, K, D_h)
            agent_hiddens.append(h_m)

            # L_agent for this agent
            if "reconstructed" in out and out["reconstructed"] is not None:
                l_m = F.mse_loss(out["reconstructed"], slots)
                l_agent_total = l_agent_total + l_m

        # Stack → (B, K, n_agents, D_h)
        H_stack = torch.stack(agent_hiddens, dim=2)

        # Soft mix: weights (B, K, n_agents) → unsqueeze → (B, K, n_agents, 1)
        w = weights.unsqueeze(-1)
        hidden_slots = (H_stack * w).sum(dim=2)        # (B, K, D_h)

        l_agent_avg = l_agent_total / max(len(self.agents), 1)
        return hidden_slots, l_agent_avg

    def _aggregate(
        self,
        hidden_slots: torch.Tensor,
        assignments: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate (B, K, D_h) → (B, D_h).
        Uses self.aggregator if available, otherwise mean pooling.
        """
        if self.aggregator is not None:
            if assignments is not None:
                result = self.aggregator(hidden_slots, assignments)
                # AttentionAggregator returns a dict with key 'aggregated'
                if isinstance(result, dict):
                    return result.get("aggregated",
                           result.get("pooled",
                           result.get("output", hidden_slots.mean(dim=1))))
                return result
            return hidden_slots.mean(dim=1)
        return hidden_slots.mean(dim=1)                # (B, D_h)

    def train_step(self, batch: Any) -> Dict[str, float]:
        images, labels = self._unpack(batch)
        images = images.to(self.device)
        has_labels = labels is not None
        if has_labels:
            labels = labels.to(self.device)

        # Current temperature
        temp = _compute_temperature(self.global_step, self._total_steps, self.config)

        # 1. Extract slots (frozen backbone)
        slots, n_active_slots = self._extract_slots(images)   # (B, K, D), float

        # 2. Soft routing weights
        weights = self._soft_route(slots, temp)          # (B, K, n_agents)

        # 3. Compute per-slot hidden and L_agent
        hidden_slots, l_agent = self._compute_hidden(slots, weights)   # (B, K, D_h), scalar

        # 4. Aggregate into one vector per sample
        # hard assignments for aggregator keys
        assignments = weights.argmax(dim=-1)             # (B, K)
        H = self._aggregate(hidden_slots, assignments=assignments)   # (B, D_h)

        # Sanitize H: NaN/Inf from upstream explosions would poison label losses
        if not torch.isfinite(H).all():
            H = torch.nan_to_num(H, nan=0.0, posinf=1.0, neginf=-1.0)

        # 5. Label losses
        l_prim   = torch.tensor(0.0, device=self.device)
        l_supcon = torch.tensor(0.0, device=self.device)

        if has_labels:
            l_prim   = _primitive_loss(H, labels, self.config.prim_temperature)
            l_supcon = _supcon_loss(H, labels, self.config.supcon_temperature)

        # 6. Total loss
        cfg = self.config
        loss_total = (
            cfg.gamma * l_agent
            + cfg.alpha * l_prim
            + cfg.beta  * l_supcon
        )

        return {
            "loss_total":     loss_total,
            "l_agent":        l_agent.detach(),
            "l_prim":         l_prim.detach()   if isinstance(l_prim,   torch.Tensor) else l_prim,
            "l_supcon":       l_supcon.detach() if isinstance(l_supcon, torch.Tensor) else l_supcon,
            "temperature":    temp,
            "n_active_slots": n_active_slots,
        }

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
