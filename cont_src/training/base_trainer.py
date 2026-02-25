"""
Base Trainer
============
Abstract base class that every phase-specific trainer inherits from.

Extension pattern
-----------------
To add a new training phase:

    class MyPhaseTrainer(BaseTrainer):
        def train_step(self, batch) -> dict:
            ...  # return {"loss_total": ..., "my_loss": ...}

        # optional hook overrides:
        def on_train_start(self): ...
        def on_epoch_end(self, epoch, metrics): ...

Hooks (in call order per step):
    on_train_start()
    for batch in loader:
        on_before_step(step, batch)
        metrics = train_step(batch)
        on_after_step(step, metrics)
        [log if step % log_every]
        [save if step % save_every]
    on_train_end(total_metrics)
"""

from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from cont_src.training.configs import PhaseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """
    Rolling average of scalar metrics.  Reset between epochs / intervals.

        acc = MetricAccumulator()
        acc.update({"loss": 0.4, "recon": 0.3})
        acc.mean()   # → {"loss": 0.4, "recon": 0.3}
    """

    def __init__(self):
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
            self._sums[k] += v
            self._counts[k] += 1

    def mean(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------

class BaseTrainer(ABC):
    """
    Abstract trainer.  Subclasses implement ``train_step`` and optionally
    override any of the hooks.

    Attributes
    ----------
    config : PhaseConfig
        Phase configuration dataclass.
    model_components : dict
        Named nn.Module objects that this trainer owns / optimises.
        Subclasses populate this in ``__init__`` **before** calling
        ``super().__init__``.
    optimizer : Optimizer | None
        Set by ``setup_optimizer()``; can be overridden.
    scheduler : _LRScheduler | None
        Optional LR scheduler.
    global_step : int
        Monotonically increasing step counter across calls to ``train()``.
    """

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.device = self._resolve_device(config.device)

        # Subclasses should populate this before calling super().__init__
        # so that setup_optimizer() can find parameters.
        if not hasattr(self, "model_components"):
            self.model_components: Dict[str, nn.Module] = {}

        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.global_step: int = 0

        self._step_callbacks: List[Callable] = []
        self._accumulator = MetricAccumulator()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Process one mini-batch and return a dict of scalar metrics.
        The dict **must** contain the key ``"loss_total"`` (the value that
        will be backpropped).

        Returns
        -------
        dict
            e.g. ``{"loss_total": 0.42, "recon": 0.1, "sparse": 0.02}``
        """

    # ------------------------------------------------------------------
    # Hooks  (override as needed)
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Called once before the training loop begins."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_before_step(self, step: int, batch: Any) -> None:
        """Called just before ``train_step``."""

    def on_after_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Called just after ``train_step`` and the optimiser step."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch with averaged metrics."""

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        """Called once after the training loop finishes."""

    # ------------------------------------------------------------------
    # Optimiser setup  (can be overridden)
    # ------------------------------------------------------------------

    def setup_optimizer(self) -> None:
        """
        Build an AdamW optimiser over all trainable parameters in
        ``self.model_components``.  Override for custom optimiser logic.
        """
        params = []
        for mod in self.model_components.values():
            if isinstance(mod, nn.Module):
                params += [p for p in mod.parameters() if p.requires_grad]

        if not params:
            logger.warning(
                f"[{self.__class__.__name__}] No trainable parameters found."
            )
            self.optimizer = None
            return

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        *,
        resume_step: int = 0,
    ) -> Dict[str, float]:
        """
        Run the training loop.

        Parameters
        ----------
        dataloader : DataLoader
            Training data.
        resume_step : int
            Resume from this global step (useful after loading a checkpoint).

        Returns
        -------
        dict
            Final averaged metrics over the whole run.
        """
        if self.optimizer is None:
            self.setup_optimizer()

        # Move all modules to device
        for mod in self.model_components.values():
            if isinstance(mod, nn.Module):
                mod.to(self.device)

        self.global_step = resume_step
        self.on_train_start()

        cfg = self.config
        use_steps = cfg.max_steps > 0
        n_epochs = 1 if use_steps else cfg.max_epochs

        all_metrics = MetricAccumulator()
        done = False

        # ── outer progress bar ─────────────────────────────────────────
        trainer_name = type(self).__name__
        if use_steps:
            outer_iter = range(n_epochs)  # always 1 in steps mode
        else:
            outer_iter = tqdm(range(n_epochs), desc=f"{trainer_name}",
                              unit="epoch", dynamic_ncols=True)

        for epoch in outer_iter:
            if done:
                break
            self.on_epoch_start(epoch)
            self._accumulator.reset()

            # ── inner progress bar ─────────────────────────────────────
            if use_steps:
                remaining = cfg.max_steps - self.global_step
                batch_iter = tqdm(
                    dataloader, total=remaining,
                    desc=f"{trainer_name} steps", unit="step",
                    dynamic_ncols=True,
                )
            else:
                batch_iter = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch + 1}/{n_epochs}",
                    unit="batch", leave=False, dynamic_ncols=True,
                )

            for batch in batch_iter:
                if use_steps and self.global_step >= cfg.max_steps:
                    batch_iter.close()
                    done = True
                    break

                self.on_before_step(self.global_step, batch)

                # ---- forward + backward ----
                metrics = self._do_step(batch)

                self.on_after_step(self.global_step, metrics)
                self._accumulator.update(metrics)
                all_metrics.update(metrics)
                self.global_step += 1

                # ---- update progress bar postfix ----
                loss_val = metrics.get("loss_total", 0)
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                postfix = {"loss": f"{loss_val:.4f}", "step": self.global_step}
                n_slots = metrics.get("n_active_slots")
                if n_slots is not None:
                    postfix["slots"] = f"{n_slots:.1f}"
                batch_iter.set_postfix(postfix)

                # ---- logging ----
                if cfg.log_every_n_steps > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avg = self._accumulator.mean()
                    self._log_metrics(avg, self.global_step)
                    self._accumulator.reset()

                # ---- checkpoint ----
                if (
                    cfg.save_every_n_steps > 0
                    and self.global_step % cfg.save_every_n_steps == 0
                ):
                    self.save_checkpoint(tag=f"step{self.global_step}")

            epoch_metrics = self._accumulator.mean()
            self.on_epoch_end(epoch, epoch_metrics)
            if not use_steps and not isinstance(outer_iter, range):
                outer_iter.set_postfix(
                    {k: f"{v:.4f}" for k, v in epoch_metrics.items()
                     if isinstance(v, (int, float))})

        final_metrics = all_metrics.mean()
        self.on_train_end(final_metrics)

        # Save final checkpoint
        self.save_checkpoint(tag="final")

        return final_metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_step(self, batch: Any) -> Dict[str, float]:
        """Forward + backward + clip + optimiser step."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        metrics = self.train_step(batch)

        loss = metrics.get("loss_total")
        if loss is not None and isinstance(loss, torch.Tensor):
            loss.backward()

            if self.config.grad_clip > 0:
                for mod in self.model_components.values():
                    if isinstance(mod, nn.Module):
                        nn.utils.clip_grad_norm_(mod.parameters(), self.config.grad_clip)

            if self.optimizer is not None:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Detach so accumulator stores plain floats
            metrics["loss_total"] = loss.detach().item()

        # Detach any remaining tensors
        return {
            k: (v.detach().item() if isinstance(v, torch.Tensor) else v)
            for k, v in metrics.items()
        }

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        logger.info(f"[{self.__class__.__name__}] step={step}  {parts}")

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> str:
        """
        Save all model_components + optimizer state.

        Returns
        -------
        str
            Path to the saved file.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.__class__.__name__}_{tag}.pt",
        )
        state = {
            "global_step": self.global_step,
            "config": self.config,
            "models": {
                k: v.state_dict()
                for k, v in self.model_components.items()
                if isinstance(v, nn.Module)
            },
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, path)
        logger.info(f"[{self.__class__.__name__}] Checkpoint saved → {path}")
        return path

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        """
        Load checkpoint.

        Returns
        -------
        int
            global_step at which the checkpoint was saved.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = state.get("global_step", 0)

        for k, sd in state.get("models", {}).items():
            if k in self.model_components:
                self.model_components[k].load_state_dict(sd, strict=strict)

        if self.optimizer and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])

        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])

        logger.info(
            f"[{self.__class__.__name__}] Loaded checkpoint from {path} "
            f"(step {self.global_step})"
        )
        return self.global_step

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, batch: Any) -> Any:
        """Recursively move tensors in batch to self.device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            moved = [self.to_device(x) for x in batch]
            return type(batch)(moved)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        return batch

    def freeze(self, *names: str) -> None:
        """Freeze named model components (gradient = False)."""
        for name in names:
            mod = self.model_components.get(name)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)

    def unfreeze(self, *names: str) -> None:
        """Unfreeze named model components."""
        for name in names:
            mod = self.model_components.get(name)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
