"""
AdaSlot Trainer  –  Phase 0
============================
Trains (or fine-tunes) the SlotAttention backbone with:

    L_total = w_recon · L_recon  +  w_sparse · L_sparse  +  w_prim · L_prim

Reconstruction loss follows AdaSlot's original convention:

    L_recon = F.mse_loss(pred, target, reduction="sum") / B

so the raw value scales as (C · H · W) ≈ 49 152 for 128×128×3 images.
``w_recon = 1/(C·H·W)`` normalises it back to a per-pixel unit.

Usage
-----
    from cont_src.training.configs import AdaSlotTrainerConfig
    from cont_src.training.adaslot_trainer import AdaSlotTrainer

    cfg = AdaSlotTrainerConfig(max_steps=500, w_recon=2.03e-5)
    trainer = AdaSlotTrainer(cfg, slot_model, primitive_predictor=prim_net)
    trainer.train(dataloader)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_src.training.base_trainer import BaseTrainer
from cont_src.training.configs import AdaSlotTrainerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: AdaSlot-style mse_sum
# ---------------------------------------------------------------------------

def mse_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss matching the original AdaSlot implementation.

        L = sum_{b,c,h,w} (pred - target)^2  /  B

    This is 49 152× larger than ``reduction="mean"`` for 128×128×3 inputs.
    Use ``w_recon = 1/(C·H·W)`` in the config to normalise back to per-pixel.
    """
    B = pred.shape[0]
    return F.mse_loss(pred, target, reduction="sum") / B


def linear_sparse_penalty(mask: torch.Tensor) -> torch.Tensor:
    """
    Sparsity loss: mean expected number of active slots.

        L_sparse = (1/B) Σ_b Σ_k mask_{b,k}

    where mask_{b,k} ∈ [0, 1] is the soft activity of slot k in sample b.
    """
    return mask.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AdaSlotTrainer(BaseTrainer):
    """
    AdaSlot backbone trainer.

    Parameters
    ----------
    config : AdaSlotTrainerConfig
    slot_model : nn.Module
        Full AdaSlot model.  Must accept a batch of images and return a dict
        with at least:
            "recon"      – reconstructed image (B, C, H, W)
            "slots"      – slot embeddings     (B, K, D)
            "mask"       – slot activity mask  (B, K)
        Optionally:
            "primitives" – primitive predictions (B, K, P) for L_prim
    primitive_predictor : nn.Module, optional
        Separate MLP that maps slots → primitive weights.  If ``None`` and the
        slot_model does not supply "primitives", L_prim is skipped.
    labels_key : str
        Key in the batch dict (or position in a tuple batch) for labels.
        Used only for L_prim.
    """

    def __init__(
        self,
        config: AdaSlotTrainerConfig,
        slot_model: nn.Module,
        primitive_predictor: Optional[nn.Module] = None,
        labels_key: str = "label",
    ):
        # Register components BEFORE calling super().__init__
        # so setup_optimizer() can scan them.
        self.model_components = {"slot_model": slot_model}
        if primitive_predictor is not None:
            self.model_components["primitive_predictor"] = primitive_predictor

        super().__init__(config)
        self.config: AdaSlotTrainerConfig  # type narrowing

        self.labels_key = labels_key

        if config.freeze_slot_attn == "frozen":
            self._freeze_slot_attention(slot_model)

        logger.info(
            f"[AdaSlotTrainer] device={self.device} "
            f"w_recon={config.w_recon}  w_sparse={config.w_sparse}  "
            f"w_prim={config.w_prim}"
        )

    # ------------------------------------------------------------------
    # Hook: log config at start
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        cfg = self.config
        use_steps = cfg.max_steps > 0
        duration = f"{cfg.max_steps} steps" if use_steps else f"{cfg.max_epochs} epochs"
        logger.info(f"[AdaSlotTrainer] Starting AdaSlot training — {duration}")

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        One mini-batch update.

        Batch can be:
          - a tuple  ``(images, labels)``
          - a dict   ``{"image": ..., "label": ...}``
          - just a tensor of images (L_prim will be skipped)
        """
        images, labels = self._unpack_batch(batch)
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # AdaSlot decoder always outputs 128×128 (initial_conv_size=(8,8) hardcoded).
        # Resize input to match, regardless of dataloader IMG_SIZE.
        if images.shape[-1] != 128 or images.shape[-2] != 128:
            images = F.interpolate(
                images, size=(128, 128), mode="bilinear", align_corners=False
            )

        slot_model = self.model_components["slot_model"]
        slot_model.train()

        output = slot_model(images)
        recon: torch.Tensor = output["recon"]          # (B, C, H, W)
        mask:  torch.Tensor = output["mask"]           # (B, K)

        # ---- L_recon ----
        l_recon_raw = mse_sum(recon, images)
        l_recon = self.config.w_recon * l_recon_raw

        # ---- L_sparse ----
        l_sparse_raw = linear_sparse_penalty(mask)
        l_sparse = self.config.w_sparse * l_sparse_raw

        # ---- L_prim ----
        l_prim = torch.tensor(0.0, device=self.device)
        if labels is not None:
            primitives = self._get_primitives(output, images)
            if primitives is not None:
                # primitives: (B, K, P)  or  (B, P)
                # aggregate over slots if needed
                if primitives.dim() == 3:
                    primitives = primitives.mean(dim=1)   # (B, P)
                l_prim = self._primitive_loss(primitives, labels)
                l_prim = self.config.w_prim * l_prim

        loss_total = l_recon + l_sparse + l_prim

        return {
            "loss_total": loss_total,
            "l_recon":    l_recon.detach(),
            "l_recon_raw": l_recon_raw.detach(),
            "l_sparse":   l_sparse.detach(),
            "l_prim":     l_prim.detach() if isinstance(l_prim, torch.Tensor) else l_prim,
            "n_active_slots": (mask > 0.5).float().sum(dim=1).mean().item(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """Returns (images, labels_or_None)."""
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            return batch[0], None
        if isinstance(batch, dict):
            return batch["image"], batch.get(self.labels_key)
        return batch, None

    def _get_primitives(self, output, images):
        """
        Returns primitive weight tensor or None.
        Priority: output["primitives"] > separate primitive_predictor.
        """
        if "primitives" in output:
            return output["primitives"]
        prim_net = self.model_components.get("primitive_predictor")
        if prim_net is not None:
            prim_net.train()
            slots = output["slots"]          # (B, K, D)
            # predict per-slot then aggregate
            return prim_net(slots.view(-1, slots.shape[-1])).view(
                slots.shape[0], slots.shape[1], -1
            )
        return None

    def _primitive_loss(
        self,
        primitives: torch.Tensor,  # (B, P)
        labels: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        """
        Soft cross-entropy / KL between label-similarity matrix and
        primitive-similarity matrix.

        This matches PrimitiveLoss in losses.py but uses the primitives
        directly as the representation (d_H is computed on primitives).
        """
        prim_norm = F.normalize(primitives, p=2, dim=1)
        sim = torch.mm(prim_norm, prim_norm.t()) * self.config.prim_temperature
        d_H = F.softmax(sim, dim=1)

        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        d_y = same / (same.sum(dim=1, keepdim=True) + 1e-8)

        kl = (d_y * torch.log((d_y + 1e-8) / (d_H + 1e-8))
              ).sum() / primitives.shape[0]
        return kl

    @staticmethod
    def _freeze_slot_attention(model: nn.Module) -> None:
        """
        Freeze sub-modules whose name contains 'slot_attention'.
        Override for custom freeze strategies.
        """
        for name, mod in model.named_modules():
            if "slot_attn" in name or "slot_attention" in name:
                for p in mod.parameters():
                    p.requires_grad_(False)
        logger.info("[AdaSlotTrainer] SlotAttention parameters frozen.")
